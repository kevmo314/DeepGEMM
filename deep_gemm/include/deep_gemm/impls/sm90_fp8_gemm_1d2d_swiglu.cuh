#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

// SiLU(gate) * up of the BF16-rounded accumulators, with the instructions (and so the
// rounding) of a Triton `gate * tl.sigmoid(gate) * up` in FP32: the unfused path writes
// the GEMM output in BF16, then computes this
CUTLASS_DEVICE float swiglu_of_bf16(const float& gate_accum, const float& up_accum) {
    const float g = __bfloat162float(__float2bfloat16_rn(gate_accum));
    const float u = __bfloat162float(__float2bfloat16_rn(up_accum));
    float t, e, d, sig, y;
    const float zero = 0.0f, one = 1.0f;
    asm volatile("sub.f32 %0, %1, %2;" : "=f"(t) : "f"(zero), "f"(g));
    asm volatile("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(t) : "f"(t));
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
    asm volatile("add.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(sig) : "f"(one), "f"(d));
    asm volatile("mul.f32 %0, %1, %2;" : "=f"(y) : "f"(sig), "f"(g));
    asm volatile("mul.f32 %0, %1, %2;" : "=f"(y) : "f"(y), "f"(u));
    return __bfloat162float(__float2bfloat16_rn(y));
}

CUTLASS_DEVICE float div_full(const float& a, const float& b) {
    float r;
    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// Two values quantized as `(y / scale).to(tl.float8e4nv)`: `lo` in the low byte
CUTLASS_DEVICE uint16_t quantize_e4m3x2(const float& lo, const float& hi, const float& scale) {
    const float q_lo = div_full(lo, scale), q_hi = div_full(hi, scale);
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(q_hi), "f"(q_lo));
    return r;
}

// FP8 [m, k] x [n, k]^T GEMM of a fused gate/up projection (B: the n / 2 gate rows, then
// the up rows; `sfb` its (n / 128, k / 128) 128x128 block scales), written as the FP8
// (1x128 scaled) input of the next projection: SiLU(gate) * up, quantized per 128
// outputs. A block is BLOCK_M rows by one quantization group, outputs [128p, 128p + 128):
// two 128-row tiles of 64 gate rows and their 64 up rows, loaded side by side, whose
// gate and up halves each lie in scale block p (gate) or n / 256 + p (up). Per K block,
// A loads once and the two tiles run as two 128-wide WGMMAs into their own accumulators. With kNumTMAMulticast 2, a
// cluster's two CTAs take adjacent M blocks of the same pair and share its B loads
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast,
          uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d2d_swiglu_impl(float* sfb, float* sd, uint32_t shape_m_pad,
                               uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(BLOCK_N == 128, "A tile is 64 gate and 64 up columns");

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M == WGMMA::M * 2 and kNumMathThreads == 256, "Two math warp-groups of one wave");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory: the FP8 output tile, 128 columns swizzled for TMA, and per stage
    // A and the pair's two B tiles
    static constexpr uint32_t kNumGroupOutputs = BLOCK_N;
    static constexpr uint32_t kNumTiles = 2;
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * kNumGroupOutputs, 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_TILE_SIZE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = kNumTiles * SMEM_B_TILE_SIZE;
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    const uint32_t shape_k_scales = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t smem_sfb_size = math::align<uint32_t>(shape_k_scales * 2 * sizeof(float), sizeof(Barrier));

    // Configs
    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_q = smem_buffer;
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size);
    auto full_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    // Initialize barriers
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }
        cutlass::arch::fence_barrier_init();
    }
    (kNumTMAMulticast > 1) ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler over pairs of tiles: a block is `BLOCK_N * 2` rows of B
    uint32_t m_block_idx, n_block_idx;
    DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");
    auto scheduler = sched::Scheduler<GemmType::Normal, BLOCK_M, BLOCK_N * kNumTiles, 1, kNumTMAMulticast, false, kNumSMs>(shape_m, shape_n, shape_k, nullptr);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                const uint32_t num_tma_multicast_b = scheduler.is_tma_multicast_valid(m_block_idx) ? kNumTMAMulticast : 1;
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3, false>(&tensor_map_a, &full_barrier,
                             smem_a[stage_idx], k_idx, m_block_idx * BLOCK_M, 1, 0);
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier,
                             smem_sfa[stage_idx], m_block_idx * BLOCK_M, k_block_idx, 1);
                    // A tile's gate rows, then their up rows: 64-row boxes, which with 128B
                    // swizzling land as one 128-row tile would
                    #pragma unroll
                    for (uint32_t t = 0; t < kNumTiles; ++ t) {
                        const uint32_t gate_row = (n_block_idx * kNumTiles + t) * (BLOCK_N / 2);
                        #pragma unroll
                        for (uint32_t u = 0; u < 2; ++ u) {
                            tma::copy<BLOCK_K, BLOCK_N / 2, kSwizzleBMode, __nv_fp8_e4m3, false>(&tensor_map_b, &full_barrier,
                                     smem_b[stage_idx] + t * SMEM_B_TILE_SIZE + u * (SMEM_B_TILE_SIZE / 2), k_idx,
                                     gate_row + u * (shape_n / 2), num_tma_multicast_b, 0);
                        }
                    }
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        auto a_desc = mma::sm90::make_smem_desc(smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K, 1);
        auto b_desc = mma::sm90::make_smem_desc(smem_b[0], 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Per thread and tile: 8 column pairs of 2 rows, gate in accumulator groups [0, 8)
        // and the matching up in [8, 16)
        constexpr uint32_t kNumGateGroups = WGMMA::kNumAccum / 8;
        DG_STATIC_ASSERT(kNumGateGroups == 8, "Invalid accumulator layout");

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // The group's gate and up block scales, the same for both of its tiles
            if (threadIdx.x >= 32) {
                const auto gate_sfb = sfb + n_block_idx * shape_k_scales;
                const auto up_sfb = sfb + (shape_n / (BLOCK_N * kNumTiles) + n_block_idx) * shape_k_scales;
                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < shape_k_scales * 2; i += kNumMathThreads - 32)
                    ptx::st_shared(smem_sfb + i, i < shape_k_scales ? gate_sfb[i] : up_sfb[i - shape_k_scales]);
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

            float accum[WGMMA::kNumAccum], final_accum[kNumTiles][WGMMA::kNumAccum] = {};

            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                // Gate and up B scales, then A's
                const float scale_b_0 = ptx::ld_shared(smem_sfb + k_block_idx);
                const float scale_b_1 = ptx::ld_shared(smem_sfb + k_block_idx + shape_k_scales);

                full_barriers[stage_idx]->wait(phase);
                const float scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + r_0);
                const float scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + r_1);
                const float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                const float scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;

                #pragma unroll
                for (uint32_t t = 0; t < kNumTiles; ++ t) {
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        a_desc.reg32_[0] = a_desc_base_lo + (k * WGMMA::K) / 16;
                        b_desc.reg32_[0] = b_desc_base_lo + (t * SMEM_B_TILE_SIZE + k * WGMMA::K) / 16;
                        WGMMA::wgmma(a_desc, b_desc, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    if (t == kNumTiles - 1) {
                        if constexpr (kNumTMAMulticast == 1) {
                            lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                        } else {
                            auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                            lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();
                        }
                    }

                    // Promote with scales, as `sm90_fp8_gemm_1d2d_impl` with its first
                    // 64 columns (gate) on one B scale and the rest (up) on the other
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        const bool predicate = i < kNumGateGroups;
                        final_accum[t][i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                        final_accum[t][i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                        final_accum[t][i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                        final_accum[t][i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                    }
                }
            }

            // SwiGLU of the pair's 128 outputs, into the gate accumulators, and each row's
            // largest |output|: over this thread's, then the 4 lanes sharing the row
            float row_max[2] = {0, 0};
            #pragma unroll
            for (uint32_t t = 0; t < kNumTiles; ++ t) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumGateGroups; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; ++ j) {
                        auto& y = final_accum[t][i * 4 + j];
                        y = swiglu_of_bf16(y, final_accum[t][(i + kNumGateGroups) * 4 + j]);
                        row_max[j / 2] = fmaxf(row_max[j / 2], fabsf(y));
                    }
                }
            }
            float row_scale[2];
            #pragma unroll
            for (uint32_t r = 0; r < 2; ++ r) {
                float m = row_max[r];
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
                row_scale[r] = fmaxf(div_full(m, 448.0f), 1e-12f);
            }

            // Wait for the last TMA store to finish reading the output tile
            if (threadIdx.x == 0)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);

            // Quantize into the swizzled FP8 tile: row `row`'s 16-byte chunk `c` sits at
            // chunk `c ^ (row % 8)`
            #pragma unroll
            for (uint32_t t = 0; t < kNumTiles; ++ t) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumGateGroups; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; j += 2) {
                        const uint32_t row = j == 0 ? r_0 : r_1;
                        const uint32_t col = t * (kNumGroupOutputs / kNumTiles) + i * 8 + (lane_idx % 4) * 2;
                        const uint32_t chunk = (col / 16) ^ (row % 8);
                        *reinterpret_cast<uint16_t*>(smem_q + row * kNumGroupOutputs + chunk * 16 + col % 16) =
                            quantize_e4m3x2(final_accum[t][i * 4 + j], final_accum[t][i * 4 + j + 1], row_scale[j / 2]);
                    }
                }
            }

            // Row scales, column-major: one lane per row
            if (lane_idx % 4 == 0) {
                #pragma unroll
                for (uint32_t r = 0; r < 2; ++ r) {
                    const uint32_t row = m_block_idx * BLOCK_M + (r == 0 ? r_0 : r_1);
                    if (row < shape_m_pad)
                        sd[n_block_idx * shape_m_pad + row] = row_scale[r];
                }
            }

            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
            if (threadIdx.x == 0) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_q, n_block_idx * kNumGroupOutputs, m_block_idx * BLOCK_M);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
