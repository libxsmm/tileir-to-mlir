// RUN: tileir-to-mlir --convert-tileir-to-mlir %s | FileCheck %s
// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=cpu append-grid-args=true' %s | FileCheck %s --check-prefix=CPU

// An implementation of GEMM in cuda_tile.
//
// Kernel computes MxNxK with 256x128x64 Tile Size.
// Computes F32 += f16 * f16 + 0.0
//
// This implementation does tiling, and reduction over
// K for dynamic sizes.

// CHECK-LABEL: gpu.func @gemm_kloop_kernel(
// CHECK-SAME:      %[[A:[a-z0-9]+]]: memref<*xf16>, %[[B:[a-z0-9]+]]: memref<*xf16>, %[[C:[a-z0-9]+]]: memref<*xf32>
// CHECK-DAG:     %[[AV:.*]] = memref.reinterpret_cast %[[A]]
// CHECK-DAG:     %[[BV:.*]] = memref.reinterpret_cast %[[B]]
// CHECK-DAG:     %[[CV:.*]] = memref.reinterpret_cast %[[C]]
// The loop over K tiles steps through elements, so it indexes the views directly.
// CHECK:         %[[RES:.*]] = scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (vector<256x128xf32>)
// CHECK:           %[[AT:.*]] = vector.transfer_read %[[AV]][%[[K]], %{{.*}}], %{{.*}} {permutation_map = #{{.*}}} : {{.*}}, vector<256x64xf16>
// CHECK:           %[[BT:.*]] = vector.transfer_read %[[BV]][%{{.*}}, %[[K]]], %{{.*}} {permutation_map = #{{.*}}} : {{.*}}, vector<64x128xf16>
// CHECK:           %[[D:.*]] = vector.contract {{.*}} %[[AT]], %[[BT]], %[[ACC]] : vector<256x64xf16>, vector<64x128xf16> into vector<256x128xf32>
// CHECK:           scf.yield %[[D]]
// CHECK:         vector.transfer_write %[[RES]], %[[CV]]

// CPU-LABEL: func.func @gemm_kloop_kernel(
// CPU-NOT:     gpu.
// CPU:         scf.for
// CPU:           vector.contract
// CPU:         vector.transfer_write

cuda_tile.module @gemm_kloop_module {
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
        %mk_len_i32:2 = get_index_space_shape %A_block : partition_view<tile=(256x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]> -> tile<i32>

        // Now that we have done all the setup, we can finally perform the  computation itself.
        //
        // We simply loop over the K dimension computing: dot(A_block[0, k], B_block[k, 0]).
        %result = for %k in (%i0 to %mk_len_i32#1, step %i1) : tile<i32>
            iter_values(%acc_prev = %cst) -> (tile<256x128xf32>)
        {
            // Load a single 256x64 matrix from the tile.
            %A_frag, %t1 = load_view_tko weak %A_block[%bidx, %k] : partition_view<tile=(256x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<256x64xf16>, token

            // Load a single 64x128 matrix from the tile.
            %B_frag, %t2 = load_view_tko weak %B_block [%k, %bidy] : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<64x128xf16>, token

            // Compute the mma(A_frag, B_frag) + acc_prev.
            %acc = mmaf %A_frag, %B_frag, %acc_prev: tile<256x64xf16>, tile<64x128xf16>, tile<256x128xf32>
            // Store the partial sum to the 256x128 accumulator.
            continue %acc : tile<256x128xf32>
        }

        // Finally store the complete 256x128 tile to the view of C.
        %t3 = store_view_tko weak %result, %C_block[%bidx, %bidy] : tile<256x128xf32>, partition_view<tile=(256x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> token
    }
}

