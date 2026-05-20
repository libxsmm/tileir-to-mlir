// An implementation of GEMM in cuda_tile.
//
// Kernel computes MxNxK with 128x128x64 Tile Size.
// Computes F32 += f16 * f16 + 0.0
//
// This implementation does tiling, and reduction over
// K for dynamic sizes.
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


// gpu.module @payload_kernel {
//   gpu.func @payload_kernel(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf16>, %arg2: memref<?x?xf16>) kernel {
//     %block_id_x = gpu.block_id x
//     %block_id_y = gpu.block_id y
//     %c256 = arith.constant 256 : index
//     %c128 = arith.constant 128 : index
//     %0 = ub.poison : f32
//     %1 = ub.poison : f16
//     %c0 = arith.constant 0 : index
//     %c512 = arith.constant 512 : index
//     %c64 = arith.constant 64 : index
//     %2 = arith.muli %block_id_x, %c256 overflow<nsw> : index
//     %3 = arith.muli %block_id_y, %c128 overflow<nsw> : index
//     %4 = vector.transfer_read %arg0[%2, %3], %0 {in_bounds = [true, true]} : memref<?x?xf32>, vector<256x128xf32>
//     %5 = scf.for %arg3 = %c0 to %c512 step %c64 iter_args(%arg4 = %4) -> (vector<256x128xf32>) {
//       %6 = vector.transfer_read %arg1[%2, %arg3], %1 {in_bounds = [true, true]} : memref<?x?xf16>, vector<256x64xf16>
//       %7 = vector.transfer_read %arg2[%arg3, %3], %1 {in_bounds = [true, true]} : memref<?x?xf16>, vector<64x128xf16>
//       %8 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %6, %7, %arg4 : vector<256x64xf16>, vector<64x128xf16> into vector<256x128xf32>
//       scf.yield %8 : vector<256x128xf32>
//     }
//     vector.transfer_write %5, %arg0[%2, %3] {in_bounds = [true, true]} : vector<256x128xf32>, memref<?x?xf32>
//     gpu.return
//   }
// }
