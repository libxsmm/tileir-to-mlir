// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | FileCheck %s
// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | mlir-opt --loop-invariant-code-motion -cse -canonicalize -cse > /dev/null

// CHECK: #map = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: gpu.module @gemm_kloop_module {

// An implementation of GEMM in cuda_tile.
//
// Kernel computes MxNxK with 128x128x64 Tile Size.
// Computes F32 += f16 * f16 + 0.0
//
// This implementation does tiling, and reduction over
// K for dynamic sizes.
// Input A, B, C are expected in (K x M), (N x K), (M x N), so the kernel needs to flip input dimensions.
cuda_tile.module @gemm_kloop_module {
    // CHECK-LABEL: gpu.func @gemm_kloop_kernel(
    // CHECK-SAME: %[[A_ARG:[a-zA-Z0-9_]+]]: memref<?x?xf16, strided<[?, 1]>>
    // CHECK-SAME: %[[B_ARG:[a-zA-Z0-9_]+]]: memref<?x?xf16, strided<[?, 1]>>
    // CHECK-SAME: %[[C_ARG:[a-zA-Z0-9_]+]]: memref<?x?xf32, strided<[?, 1]>>
    entry @gemm_kloop_kernel(
        %A_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %B_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %C_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %M: !cuda_tile.tile<i32>, %N: !cuda_tile.tile<i32>, %K: !cuda_tile.tile<i32>,
        %stride_ak: !cuda_tile.tile<i32>, %stride_bn: !cuda_tile.tile<i32>, %stride_cm: !cuda_tile.tile<i32>
    ) {
        // First we need to prepare the inputs for the actual computation.
        //
        // Assume the preconditions of this kernel (i.e., the stride are all divisible by 8)
        %A_ptr_assume = assume #cuda_tile.div_by<16>, %A_ptr : tile<ptr<f16>>
        %B_ptr_assume = assume #cuda_tile.div_by<16>, %B_ptr : tile<ptr<f16>>
        %C_ptr_assume = assume #cuda_tile.div_by<16>, %C_ptr : tile<ptr<f32>>
        %stride_ak_assume = assume #cuda_tile.div_by<8>, %stride_ak : tile<i32>
        %stride_bn_assume = assume #cuda_tile.div_by<8>, %stride_bn : tile<i32>
        %stride_cm_assume = assume #cuda_tile.div_by<8>, %stride_cm : tile<i32>

        // Constants must be allocated explicitly in the program, below we allocate scalar `0`, `1`,
        // and the zero'd tensor used for accumulation.
        %i0 = constant <i32: 0> : !cuda_tile.tile<i32>
        %i1 = constant <i32: 1> : !cuda_tile.tile<i32>
        %cst = constant <f32: 0.000000e+00> : !cuda_tile.tile<256x128xf32>

        // Convert the unstructured pointers `ptr` to `tensor_view`.
        //
        // A reference to the A tensor pointed to by A_ptr, (K x M)
        %A = make_tensor_view %A_ptr_assume, shape = [%K, %M], strides = [%stride_ak, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
        // A reference to the B tensor pointed to by B_ptr, (N x K)
        %B = make_tensor_view %B_ptr_assume, shape = [%N, %K], strides = [%stride_bn, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
        // A reference to the C tensor pointed to by C_ptr, (M x N)
        %C = make_tensor_view %C_ptr_assume, shape = [%M, %N], strides = [%stride_cm, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>

        // Now we have all the inputs as structured pointers each associated with layouts.
        //
        // Next we will tile the problem.
        //
        // Our matrix multiplication is (M*K) @ (K*N) = M*N but our input tensors are transposed.
        //
        // In order to handle this we create partition view where we flip the 0th and 1st dims.

        // We are blocking A (K x M) -> block_m x block_k.
        %A_block  = make_partition_view %A : partition_view<tile=(256x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
        // We are blocking B (N x K) -> block_k x block_n.
        %B_block  = make_partition_view %B : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
        // We are blocking C (M xN) -> block_m x block_n.
        %C_block  = make_partition_view %C : partition_view<tile=(256x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>

        // Read Tile block id's.
        %bidx, %bidy, %bidz = get_tile_block_id : tile<i32>

        // Because we allow for dynamic dimensions we must get the reduction dimension `K` dynamically.
        // CHECK: %[[D0:.*]] = memref.dim %[[A_ARG]], %{{.*}} : memref<?x?xf16, strided<[?, 1]>>
        // CHECK: %[[C256:.*]] = arith.constant 256 : index
        // CHECK: %[[CEIL0:.*]] = arith.ceildivui %[[D0]], %[[C256]] : index
        // CHECK: %[[D1:.*]] = memref.dim %[[A_ARG]], %{{.*}} : memref<?x?xf16, strided<[?, 1]>>
        // CHECK: %[[C64:.*]] = arith.constant 64 : index
        // CHECK: %[[CEIL1:.*]] = arith.ceildivui %[[D1]], %[[C64]] : index
        %mk_len_i32:2 = get_index_space_shape %A_block : partition_view<tile=(256x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]> -> tile<i32>

        // Now that we have done all the setup, we can finally perform the  computation itself.
        //
        // We simply loop over the K dimension computing: dot(A_block[0, k], B_block[k, 0]).
        // CHECK: scf.for %{{.*}} = %{{.*}} to %[[CEIL1]] step %{{.*}}
        %result = for %k in (%i0 to %mk_len_i32#1, step %i1) : tile<i32>
            iter_values(%acc_prev = %cst) -> (tile<256x128xf32>)
        {
            // Load a single 256x64 matrix from the tile.
            // CHECK: %[[A_FRAG:.*]] = vector.transfer_read %[[A_ARG]][%{{.*}}, %{{.*}}], %{{.*}} {permutation_map = #map} : memref<?x?xf16, strided<[?, 1]>>, vector<256x64xf16>
            %A_frag, %t1 = load_view_tko weak %A_block[%bidx, %k] : partition_view<tile=(256x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<256x64xf16>, token

            // Load a single 64x128 matrix from the tile.
            // CHECK: %[[B_FRAG:.*]] = vector.transfer_read %[[B_ARG]][%{{.*}}, %{{.*}}], %{{.*}} {permutation_map = #map} : memref<?x?xf16, strided<[?, 1]>>, vector<64x128xf16>
            %B_frag, %t2 = load_view_tko weak %B_block [%k, %bidy] : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<64x128xf16>, token

            // Compute the mma(A_frag, B_frag) + acc_prev.
            // CHECK: %[[ACC_NEXT:.*]] = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[A_FRAG]], %[[B_FRAG]], %{{.*}} : vector<256x64xf16>, vector<64x128xf16> into vector<256x128xf32>
            %acc = mmaf %A_frag, %B_frag, %acc_prev: tile<256x64xf16>, tile<64x128xf16>, tile<256x128xf32>
            // Store the partial sum to the 256x128 accumulator.
            // CHECK: scf.yield %[[ACC_NEXT]] : vector<256x128xf32>
            continue %acc : tile<256x128xf32>
        }

        // Finally store the complete 256x128 tile to the view of C.
        // CHECK: vector.transfer_write %{{.*}}, %[[C_ARG]][%{{.*}}, %{{.*}}] : vector<256x128xf32>, memref<?x?xf32, strided<[?, 1]>>
        // CHECK-NOT: permutation_map = #map
        // CHECK: gpu.return
        %t3 = store_view_tko weak %result, %C_block[%bidx, %bidy] : tile<256x128xf32>, partition_view<tile=(256x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> token
    }
}

// Section 11.2.2: https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-constant-0
// CHECK-LABEL: gpu.module @appendix_constant_0_module {
cuda_tile.module @appendix_constant_0_module {
    // CHECK-LABEL: gpu.func @appendix_constant_0_kernel
    entry @appendix_constant_0_kernel() {
        // CHECK: arith.constant 0 : index
        %c0 = constant <i32: 0> : tile<i32>

        // CHECK: arith.constant 1 : i64
        %c1 = constant <i64: 1> : tile<i64>

        // CHECK: arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
        %c2 = constant <i32: [0, 1, 2, 3]> : tile<4xi32>

        // CHECK: arith.constant 0.000000e+00 : f32
        // CHECK: vector.broadcast
        %c3 = constant <f32: 0.0> : tile<2x4xf32>

        // CHECK: arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : vector<4xf64>
        %c4 = constant <f64: [0.0, 1.0, 2.0, 3.0]> : tile<4xf64>

        // CHECK: gpu.return
        return
    }
}

// Section 11.2.16: https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-for-0
// CHECK-LABEL: gpu.module @appendix_for_0_module {
cuda_tile.module @appendix_for_0_module {
    // CHECK-LABEL: gpu.func @appendix_for_0_kernel
    entry @appendix_for_0_kernel() {
        %lowerBound = constant <i32: 0> : tile<i32>
        %upperBound = constant <i32: 10> : tile<i32>
        %step = constant <i32: 1> : tile<i32>

        // CHECK: %[[LB:.*]] = arith.constant 0 : index
        // CHECK: %[[UB:.*]] = arith.constant 10 : index
        // CHECK: %[[STEP:.*]] = arith.constant 1 : index
        // CHECK: scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] {
        for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
            continue
        }

        %initVal0 = constant <f32: 0.0> : tile<f32>
        // CHECK: arith.constant 0.000000e+00 : f32
        // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f32) {
        // CHECK: %[[VAL:.*]] = arith.constant 1.000000e+00 : f32
        // CHECK: scf.yield %[[VAL]] : f32
        %results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
                        iter_values(%val00 = %initVal0) -> (tile<f32>) {
            %loopVal0 = constant <f32: 1.0> : tile<f32>
            continue %loopVal0 : tile<f32>
        }

        // CHECK: gpu.return
        return
    }
}
