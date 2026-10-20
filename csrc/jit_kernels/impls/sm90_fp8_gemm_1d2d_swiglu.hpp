#pragma once

#include <filesystem>
#include <format>
#include <vector>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"

#include "../../utils/math.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90FP8Gemm1D2DSwiGLURuntime final {
public:
    struct Args {
        int m, n, k, m_pad;
        int block_m, num_stages, num_multicast, num_sms;
        deep_jit::cuda::LaunchOptions options;

        void *sfb, *sd;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        CUtensorMap tensor_map_sfa;
    };

    static std::string generate(const Args& args) {
        return std::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d_swiglu.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_swiglu_impl<
        0, {}, {},
        {}, 128, 128,
        128, 128,
        {},
        128, 256,
        {},
        {}
    >);
}};
)", args.n, args.k, args.block_m, args.num_stages, args.num_multicast, args.num_sms);
    }

    static void compile_and_launch(const Args& args) {
        const auto kernel = jit->compile("sm90_fp8_gemm_1d2d_swiglu", generate(args));
        jit->launch(
            kernel, args.options,
            args.sfb, args.sd, args.m_pad,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_d, args.tensor_map_sfa
        );
    }
};

// FP8 [m, k] x [n, k]^T GEMM of a fused gate/up projection (B: gate rows then up rows, with
// 128x128 block scales `sfb`), written as the FP8 input of the next projection, SiLU(gate) *
// up: `qd` (m_pad, n / 2) and `sd`, its column-major 1x128 scales stored as (n / 256, m_pad).
// SFA is column-major with a K stride of m_pad, as `sd`
static void sm90_fp8_gemm_1d2d_swiglu(const torch::Tensor& a, const torch::Tensor& sfa,
                                      const torch::Tensor& b, const torch::Tensor& sfb,
                                      const torch::Tensor& qd, const torch::Tensor& sd,
                                      const int& m, const int& n, const int& k, const int& m_pad) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn and b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(qd.scalar_type() == torch::kFloat8_e4m3fn and sd.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat and sfb.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(n % 256 == 0 and k % 128 == 0 and m_pad % 4 == 0 and m <= m_pad);
    DG_HOST_ASSERT(qd.size(0) == m_pad and qd.size(1) == n / 2 and qd.is_contiguous());
    DG_HOST_ASSERT(sd.numel() >= static_cast<int64_t>(n / 256) * m_pad and sd.is_contiguous());
    DG_HOST_ASSERT(sfb.numel() == static_cast<int64_t>(n / 128) * (k / 128) and sfb.is_contiguous());
    DG_HOST_ASSERT(a.stride(1) == 1 and b.stride(1) == 1);

    // 128-row blocks of a 256-row B pair: A loads once per K block for both tiles
    // A cluster's two CTAs share B (TMA multicast) on adjacent M blocks, when there are two
    const int block_m = 128, num_stages = 4;
    const int num_multicast = m > block_m ? 2 : 1;
    const int num_sms = runtime->get_num_sms();
    const int smem_d = block_m * 128;
    const int smem_stage = block_m * 128 + 2 * 128 * 128 + align(block_m * 4, 128);
    const int smem_size = smem_d + num_stages * smem_stage + align(k / 128 * 2 * 4, 8) + num_stages * 2 * 8;

    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k, block_m, 128,
                                              static_cast<int>(a.stride(0)), 1, 128);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k, 64, 128,
                                              static_cast<int>(b.stride(0)), 1, 128);
    const auto tensor_map_d = make_tma_2d_desc(qd, n / 2, m_pad, 128, block_m,
                                               static_cast<int>(qd.stride(0)), 128);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k, block_m, 128, 1, 0,
                                                 0, false, 1, m_pad);

    SM90FP8Gemm1D2DSwiGLURuntime::compile_and_launch({
        .m = m, .n = n, .k = k, .m_pad = m_pad,
        .block_m = block_m, .num_stages = num_stages, .num_multicast = num_multicast, .num_sms = num_sms,
        .options = {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(128 + 256, 1, 1),
            .cluster_dim = dim3(num_multicast, 1, 1),
        },
        .sfb = sfb.data_ptr(),
        .sd = sd.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    });
}

// Compiles the kernel `sm90_fp8_gemm_1d2d_swiglu` would launch for an [n, k] gate/up weight into
// the disk cache, without a GPU: for `num_sms` SMs and `arch` (e.g., "90a"), with and without
// the B multicast (which prompts longer than one 128-row block use). Returns the cache entries.
static std::vector<std::filesystem::path> sm90_fp8_gemm_1d2d_swiglu_precompile(
        const int& n, const int& k, const int& num_sms, const std::string& arch) {
    std::vector<std::filesystem::path> entries;
    for (const int num_multicast: {1, 2}) {
        SM90FP8Gemm1D2DSwiGLURuntime::Args args = {};
        args.n = n, args.k = k;
        args.block_m = 128, args.num_stages = 4, args.num_multicast = num_multicast, args.num_sms = num_sms;
        deep_jit::cuda::CompilerOptions options;
        options.arch = arch;
        entries.push_back(jit->compile_without_load("sm90_fp8_gemm_1d2d_swiglu",
                                                    SM90FP8Gemm1D2DSwiGLURuntime::generate(args), options));
    }
    return entries;
}

} // namespace deep_gemm
