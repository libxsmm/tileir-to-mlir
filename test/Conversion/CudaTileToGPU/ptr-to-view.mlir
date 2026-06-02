// RUN: cudatile-to-gpu --tileir-ptr-to-view %s | FileCheck %s
// RUN: cudatile-to-gpu --tileir-ptr-to-view --convert-cuda-tile-to-gpu %s | FileCheck %s --check-prefix=CHECK-GPU

// CHECK-GPU-LABEL: gpu.module @cuda_tile_module {

module {
  cuda_tile.module @cuda_tile_module {

    // CHECK-LABEL: entry @load_1d
    // CHECK-GPU-LABEL: gpu.func @load_1d
    entry @load_1d(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %shape_1d = reshape %arg1 : tile<i32> -> tile<1xi32>
      %shape_bc = broadcast %shape_1d : tile<1xi32> -> tile<1024xi32>
      %mask = cmpi less_than %index, %shape_bc, signed : tile<1024xi32> -> tile<1024xi1>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[TV0:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PV0:.*]] = make_partition_view %[[TV0]] : partition_view<tile=(1024), padding_value = zero, tensor_view<?xf32, strides=[1]>>
      // CHECK: %[[LD0:.*]], %[[TOK0:.*]] = load_view_tko weak %[[PV0]][%{{.*}}] : partition_view<tile=(1024), padding_value = zero, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      // CHECK-GPU: %[[LOAD1D_VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%{{.*}}], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
      // CHECK-GPU: %[[LOAD1D_TILE:.*]] = vector.transfer_read %[[LOAD1D_VIEW]][%{{.*}}], %{{.*}} : memref<?xf32, strided<[1]>>, vector<1024xf32>
      %tile, %token = load_ptr_tko weak %ptr, %mask : tile<1024xptr<f32>>, tile<1024xi1> -> tile<1024xf32>, token
      return
    }

    // CHECK-LABEL: entry @store_1d
    // CHECK-GPU-LABEL: gpu.func @store_1d
    entry @store_1d(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %value = constant <f32: 1.000000e+00> : tile<1024xf32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %shape_1d = reshape %arg1 : tile<i32> -> tile<1xi32>
      %shape_bc = broadcast %shape_1d : tile<1xi32> -> tile<1024xi32>
      %mask = cmpi less_than %index, %shape_bc, signed : tile<1024xi32> -> tile<1024xi1>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[TV1:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PV1:.*]] = make_partition_view %[[TV1]] : partition_view<tile=(1024), padding_value = zero, tensor_view<?xf32, strides=[1]>>
      // CHECK: %[[ST1:.*]] = store_view_tko weak %{{.*}}, %[[PV1]][%{{.*}}] : tile<1024xf32>, partition_view<tile=(1024), padding_value = zero, tensor_view<?xf32, strides=[1]>>, tile<i32> -> token
      // CHECK-GPU: %[[STORE1D_VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%{{.*}}], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
      // CHECK-GPU: vector.transfer_write %{{.*}}, %[[STORE1D_VIEW]][%{{.*}}] : vector<1024xf32>, memref<?xf32, strided<[1]>>
      %token = store_ptr_tko weak %ptr, %value, %mask : tile<1024xptr<f32>>, tile<1024xf32>, tile<1024xi1> -> token
      return
    }

    // CHECK-LABEL: entry @store_2d
    // CHECK-GPU-LABEL: gpu.func @store_2d
    entry @store_2d(%arg0: tile<ptr<f16>>, %arg1: tile<i32>, %arg2: tile<i32>, %arg3: tile<i32>) {
      %cst_128 = constant <i32: 128> : tile<i32>
      %cst_64 = constant <i32: 64> : tile<i32>
      %value = constant <f16: 0.000000e+00> : tile<128x64xf16>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %row_start = muli %block_id_x, %cst_128 : tile<i32>
      %col_start = muli %block_id_y, %cst_64 : tile<i32>
      %row_lane = iota : tile<128xi32>
      %col_lane = iota : tile<64xi32>
      %row_start_1d = reshape %row_start : tile<i32> -> tile<1xi32>
      %row_start_bc = broadcast %row_start_1d : tile<1xi32> -> tile<128xi32>
      %rows = addi %row_start_bc, %row_lane : tile<128xi32>
      %col_start_1d = reshape %col_start : tile<i32> -> tile<1xi32>
      %col_start_bc = broadcast %col_start_1d : tile<1xi32> -> tile<64xi32>
      %cols = addi %col_start_bc, %col_lane : tile<64xi32>
      %rows_2d = reshape %rows : tile<128xi32> -> tile<128x1xi32>
      %cols_2d = reshape %cols : tile<64xi32> -> tile<1x64xi32>
      %stride_2d = reshape %arg3 : tile<i32> -> tile<1x1xi32>
      %stride_bc = broadcast %stride_2d : tile<1x1xi32> -> tile<128x1xi32>
      %linear_rows = muli %rows_2d, %stride_bc : tile<128x1xi32>
      %base_2d = reshape %arg0 : tile<ptr<f16>> -> tile<1x1xptr<f16>>
      %base_row_bc = broadcast %base_2d : tile<1x1xptr<f16>> -> tile<128x1xptr<f16>>
      %row_ptr = offset %base_row_bc, %linear_rows : tile<128x1xptr<f16>>, tile<128x1xi32> -> tile<128x1xptr<f16>>
      %base_col_bc = broadcast %row_ptr : tile<128x1xptr<f16>> -> tile<128x64xptr<f16>>
      %col_bc = broadcast %cols_2d : tile<1x64xi32> -> tile<128x64xi32>
      %ptr = offset %base_col_bc, %col_bc : tile<128x64xptr<f16>>, tile<128x64xi32> -> tile<128x64xptr<f16>>
      %shape_m = reshape %arg1 : tile<i32> -> tile<1x1xi32>
      %shape_m_bc = broadcast %shape_m : tile<1x1xi32> -> tile<128x1xi32>
      %row_mask = cmpi less_than %rows_2d, %shape_m_bc, signed : tile<128x1xi32> -> tile<128x1xi1>
      %shape_n = reshape %arg2 : tile<i32> -> tile<1x1xi32>
      %shape_n_bc = broadcast %shape_n : tile<1x1xi32> -> tile<1x64xi32>
      %col_mask = cmpi less_than %cols_2d, %shape_n_bc, signed : tile<1x64xi32> -> tile<1x64xi1>
      %row_mask_bc = broadcast %row_mask : tile<128x1xi1> -> tile<128x64xi1>
      %col_mask_bc = broadcast %col_mask : tile<1x64xi1> -> tile<128x64xi1>
      %row_mask_i16 = exti %row_mask_bc signed : tile<128x64xi1> -> tile<128x64xi16>
      %col_mask_i16 = exti %col_mask_bc signed : tile<128x64xi1> -> tile<128x64xi16>
      %mask_i16 = andi %row_mask_i16, %col_mask_i16 : tile<128x64xi16>
      %mask = trunci %mask_i16 : tile<128x64xi16> -> tile<128x64xi1>
      // CHECK: %[[TV2:.*]] = make_tensor_view %arg0, shape = [%arg1, %arg2], strides = [%arg3, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
      // CHECK: %[[PV2:.*]] = make_partition_view %[[TV2]] : partition_view<tile=(128x64), padding_value = zero, tensor_view<?x?xf16, strides=[?,1]>>
      // CHECK: %[[ST2:.*]] = store_view_tko weak %{{.*}}, %[[PV2]][%{{.*}}, %{{.*}}] : tile<128x64xf16>, partition_view<tile=(128x64), padding_value = zero, tensor_view<?x?xf16, strides=[?,1]>>, tile<i32> -> token
      // CHECK-GPU: %[[STORE2D_VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%{{.*}}, %{{.*}}], strides: [%{{.*}}, 1] : memref<*xf16> to memref<?x?xf16, strided<[?, 1]>>
      // CHECK-GPU: vector.transfer_write %{{.*}}, %[[STORE2D_VIEW]][%{{.*}}, %{{.*}}] : vector<128x64xf16>, memref<?x?xf16, strided<[?, 1]>>
      %token = store_ptr_tko weak %ptr, %value, %mask : tile<128x64xptr<f16>>, tile<128x64xf16>, tile<128x64xi1> -> token
      return
    }

    // CHECK-LABEL: entry @k_zero
    entry @k_zero(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %pad = constant <f32: 0.000000e+00> : tile<1024xf32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %shape_1d = reshape %arg1 : tile<i32> -> tile<1xi32>
      %shape_bc = broadcast %shape_1d : tile<1xi32> -> tile<1024xi32>
      %mask = cmpi less_than %index, %shape_bc, signed : tile<1024xi32> -> tile<1024xi1>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[TVZ:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PVZ:.*]] = make_partition_view %[[TVZ]] : partition_view<tile=(1024), padding_value = zero, tensor_view<?xf32, strides=[1]>>
      // CHECK: %[[LDZ:.*]], %[[TOKZ:.*]] = load_view_tko weak %[[PVZ]][%{{.*}}] : partition_view<tile=(1024), padding_value = zero, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, token
      return
    }

    // CHECK-LABEL: entry @k_neg_zero
    entry @k_neg_zero(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %pad = constant <f32: -0.000000e+00> : tile<1024xf32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %shape_1d = reshape %arg1 : tile<i32> -> tile<1xi32>
      %shape_bc = broadcast %shape_1d : tile<1xi32> -> tile<1024xi32>
      %mask = cmpi less_than %index, %shape_bc, signed : tile<1024xi32> -> tile<1024xi1>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[TVNZ:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PVNZ:.*]] = make_partition_view %[[TVNZ]] : partition_view<tile=(1024), padding_value = neg_zero, tensor_view<?xf32, strides=[1]>>
      // CHECK: %[[LDNZ:.*]], %[[TOKNZ:.*]] = load_view_tko weak %[[PVNZ]][%{{.*}}] : partition_view<tile=(1024), padding_value = neg_zero, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, token
      return
    }

    // CHECK-LABEL: entry @k_nan
    // CHECK-GPU-LABEL: gpu.func @k_nan
    entry @k_nan(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %pad = constant <f32: 0x7FC00000> : tile<1024xf32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %shape_1d = reshape %arg1 : tile<i32> -> tile<1xi32>
      %shape_bc = broadcast %shape_1d : tile<1xi32> -> tile<1024xi32>
      %mask = cmpi less_than %index, %shape_bc, signed : tile<1024xi32> -> tile<1024xi1>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[TVN:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PVN:.*]] = make_partition_view %[[TVN]] : partition_view<tile=(1024), padding_value = nan, tensor_view<?xf32, strides=[1]>>
      // CHECK: %[[LDN:.*]], %[[TOKN:.*]] = load_view_tko weak %[[PVN]][%{{.*}}] : partition_view<tile=(1024), padding_value = nan, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      // CHECK-GPU: %[[NANPAD:.*]] = arith.constant 0x7FC00000 : f32
      // CHECK-GPU: vector.transfer_read %{{.*}}[%{{.*}}], %[[NANPAD]] : memref<?xf32, strided<[1]>>, vector<1024xf32>
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, token
      return
    }

    // CHECK-LABEL: entry @k_pos_inf
    entry @k_pos_inf(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %pad = constant <f32: 0x7F800000> : tile<1024xf32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %shape_1d = reshape %arg1 : tile<i32> -> tile<1xi32>
      %shape_bc = broadcast %shape_1d : tile<1xi32> -> tile<1024xi32>
      %mask = cmpi less_than %index, %shape_bc, signed : tile<1024xi32> -> tile<1024xi1>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[TVPI:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PVPI:.*]] = make_partition_view %[[TVPI]] : partition_view<tile=(1024), padding_value = pos_inf, tensor_view<?xf32, strides=[1]>>
      // CHECK: %[[LDPI:.*]], %[[TOKPI:.*]] = load_view_tko weak %[[PVPI]][%{{.*}}] : partition_view<tile=(1024), padding_value = pos_inf, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, token
      return
    }

    // CHECK-LABEL: entry @k_neg_inf
    entry @k_neg_inf(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %pad = constant <f32: 0xFF800000> : tile<1024xf32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %shape_1d = reshape %arg1 : tile<i32> -> tile<1xi32>
      %shape_bc = broadcast %shape_1d : tile<1xi32> -> tile<1024xi32>
      %mask = cmpi less_than %index, %shape_bc, signed : tile<1024xi32> -> tile<1024xi1>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[TVNI:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PVNI:.*]] = make_partition_view %[[TVNI]] : partition_view<tile=(1024), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>
      // CHECK: %[[LDNI:.*]], %[[TOKNI:.*]] = load_view_tko weak %[[PVNI]][%{{.*}}] : partition_view<tile=(1024), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, token
      return
    }

    // Tests ptr-to-view rewriting of loop-carried pointer iter_args.
    // The pattern: a 2D ptr is built outside the loop (with one dim having a
    // static tile index and the other having just an iota), then the loop advances
    // the iota-only dimension each iteration via `offset(iterArg, constant_step)`.
    // CHECK-LABEL: entry @load_in_loop
    // CHECK-GPU-LABEL: gpu.func @load_in_loop
    entry @load_in_loop (
        %base: tile<ptr<f16>>,
        %M: tile<i32>, %K: tile<i32>,
        %stride: tile<i32>, %num_tiles: tile<i32>) {

      %cst_0_i32 = constant <i32: 0> : tile<i32>
      %cst_1_i32 = constant <i32: 1> : tile<i32>
      %cst_128_i32 = constant <i32: 128> : tile<i32>
      %cst_32_step = constant <i32: 32> : tile<128x32xi32>
      %cst_pad = constant <f16: 0.000000e+00> : tile<128x32xf16>
      %cst_0_f32 = constant <f32: 0.000000e+00> : tile<128x64xf32>

      // Compute tile index for dim-0: blockId * 128.
      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      %row_start = muli %blockId_x, %cst_128_i32 : tile<i32>

      // Build initial 2D ptr: dim-0 = row_start + iota (with stride),
      //                        dim-1 = iota (contiguous, no start).
      %iota_128 = iota : tile<128xi32>
      %rs_1d = reshape %row_start : tile<i32> -> tile<1xi32>
      %rs_bc = broadcast %rs_1d : tile<1xi32> -> tile<128xi32>
      %off0_1d = addi %rs_bc, %iota_128 : tile<128xi32>
      %off0_2d = reshape %off0_1d : tile<128xi32> -> tile<128x1xi32>
      %stride_2d = reshape %stride : tile<i32> -> tile<1x1xi32>
      %stride_bc = broadcast %stride_2d : tile<1x1xi32> -> tile<128x1xi32>
      %off0_strided = muli %off0_2d, %stride_bc : tile<128x1xi32>

      %base_2d = reshape %base : tile<ptr<f16>> -> tile<1x1xptr<f16>>
      %base_col = broadcast %base_2d : tile<1x1xptr<f16>> -> tile<128x1xptr<f16>>
      %ptr1 = offset %base_col, %off0_strided : tile<128x1xptr<f16>>, tile<128x1xi32> -> tile<128x1xptr<f16>>

      %iota_32 = iota : tile<32xi32>
      %iota_row = reshape %iota_32 : tile<32xi32> -> tile<1x32xi32>
      %ptr1_bc = broadcast %ptr1 : tile<128x1xptr<f16>> -> tile<128x32xptr<f16>>
      %iota_bc = broadcast %iota_row : tile<1x32xi32> -> tile<128x32xi32>
      %ptr_init = offset %ptr1_bc, %iota_bc : tile<128x32xptr<f16>>, tile<128x32xi32> -> tile<128x32xptr<f16>>

      // Mask: dim-0 < M, dim-1 < K.
      %M_2d = reshape %M : tile<i32> -> tile<1x1xi32>
      %M_bc = broadcast %M_2d : tile<1x1xi32> -> tile<128x1xi32>
      %cmp0 = cmpi less_than %off0_2d, %M_bc, signed : tile<128x1xi32> -> tile<128x1xi1>
      %K_2d = reshape %K : tile<i32> -> tile<1x1xi32>
      %K_bc = broadcast %K_2d : tile<1x1xi32> -> tile<1x32xi32>
      %cmp1 = cmpi less_than %iota_row, %K_bc, signed : tile<1x32xi32> -> tile<1x32xi1>
      %cmp0_bc = broadcast %cmp0 : tile<128x1xi1> -> tile<128x32xi1>
      %cmp1_bc = broadcast %cmp1 : tile<1x32xi1> -> tile<128x32xi1>
      %mask = andi %cmp0_bc, %cmp1_bc : tile<128x32xi1>

      // For loop: ptr advances dim-1 by 32 each iteration.  The view does not
      // depend on the loop induction variable, so it is hoisted out of the loop
      // (placed right after the last operand it uses) rather than rebuilt each
      // iteration.
      // CHECK: make_tensor_view %arg0
      // CHECK: make_partition_view
      // CHECK: for %[[IDX:.*]] in
      // CHECK:   load_view_tko weak %{{.*}}[%{{.*}}, %[[IDX]]]
      // CHECK-NOT: load_ptr_tko
      // CHECK-GPU: scf.for %[[IV:.*]] =
      // CHECK-GPU:   memref.reinterpret_cast
      // CHECK-GPU:   vector.transfer_read
      %for:2 = for %loopIdx in (%cst_0_i32 to %num_tiles, step %cst_1_i32) : tile<i32>
          iter_values(%iterPtr = %ptr_init, %iterAcc = %cst_0_f32)
          -> (tile<128x32xptr<f16>>, tile<128x64xf32>) {
        %tile, %tok = load_ptr_tko weak %iterPtr, %mask, %cst_pad
            : tile<128x32xptr<f16>>, tile<128x32xi1>, tile<128x32xf16>
            -> tile<128x32xf16>, token
        %next_ptr = offset %iterPtr, %cst_32_step
            : tile<128x32xptr<f16>>, tile<128x32xi32> -> tile<128x32xptr<f16>>
        continue %next_ptr, %iterAcc
            : tile<128x32xptr<f16>>, tile<128x64xf32>
      }
      return
    }

    // `assume` metadata that the source attached to operands the rewrite
    // re-uses (the base pointer and the shape/stride scalars) must be forwarded
    // onto the new make_tensor_view operands.  Here the base carries div_by<16>
    // and the shape carries a *chain* of assumes (div_by<8> then bounded),
    // which must be forwarded transitively and in order.
    // CHECK-LABEL: entry @forward_assume_1d
    entry @forward_assume_1d(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %a_base = assume #cuda_tile.div_by<16>, %arg0 : tile<ptr<f32>>
      %a_shape0 = assume #cuda_tile.div_by<8>, %arg1 : tile<i32>
      %a_shape1 = assume #cuda_tile.bounded<0, ?>, %a_shape0 : tile<i32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %shape_1d = reshape %a_shape1 : tile<i32> -> tile<1xi32>
      %shape_bc = broadcast %shape_1d : tile<1xi32> -> tile<1024xi32>
      %mask = cmpi less_than %index, %shape_bc, signed : tile<1024xi32> -> tile<1024xi1>
      %base_1d = reshape %a_base : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[AB:.*]] = assume div_by<16>, %arg0 : tile<ptr<f32>>
      // CHECK: %[[AS0:.*]] = assume div_by<8>, %arg1 : tile<i32>
      // CHECK: %[[AS1:.*]] = assume bounded<0, ?>, %[[AS0]] : tile<i32>
      // CHECK: %[[TVF:.*]] = make_tensor_view %[[AB]], shape = [%[[AS1]]], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PVF:.*]] = make_partition_view %[[TVF]] : partition_view<tile=(1024), padding_value = zero, tensor_view<?xf32, strides=[1]>>
      // CHECK: load_view_tko weak %[[PVF]][%{{.*}}]
      %tile, %token = load_ptr_tko weak %ptr, %mask : tile<1024xptr<f32>>, tile<1024xi1> -> tile<1024xf32>, token
      return
    }

    // Forwarding is *selective*: only operands the source actually annotated
    // get an assume.  Here the base (div_by<16>) and the stride (div_by<8>) are
    // annotated, but the two shape scalars are not, so they must appear bare in
    // the make_tensor_view.
    // CHECK-LABEL: entry @forward_assume_2d
    entry @forward_assume_2d(%arg0: tile<ptr<f16>>, %arg1: tile<i32>, %arg2: tile<i32>, %arg3: tile<i32>) {
      %cst_128 = constant <i32: 128> : tile<i32>
      %cst_64 = constant <i32: 64> : tile<i32>
      %value = constant <f16: 0.000000e+00> : tile<128x64xf16>
      %a_base = assume #cuda_tile.div_by<16>, %arg0 : tile<ptr<f16>>
      %a_stride = assume #cuda_tile.div_by<8>, %arg3 : tile<i32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %row_start = muli %block_id_x, %cst_128 : tile<i32>
      %col_start = muli %block_id_y, %cst_64 : tile<i32>
      %row_lane = iota : tile<128xi32>
      %col_lane = iota : tile<64xi32>
      %row_start_1d = reshape %row_start : tile<i32> -> tile<1xi32>
      %row_start_bc = broadcast %row_start_1d : tile<1xi32> -> tile<128xi32>
      %rows = addi %row_start_bc, %row_lane : tile<128xi32>
      %col_start_1d = reshape %col_start : tile<i32> -> tile<1xi32>
      %col_start_bc = broadcast %col_start_1d : tile<1xi32> -> tile<64xi32>
      %cols = addi %col_start_bc, %col_lane : tile<64xi32>
      %rows_2d = reshape %rows : tile<128xi32> -> tile<128x1xi32>
      %cols_2d = reshape %cols : tile<64xi32> -> tile<1x64xi32>
      %stride_2d = reshape %a_stride : tile<i32> -> tile<1x1xi32>
      %stride_bc = broadcast %stride_2d : tile<1x1xi32> -> tile<128x1xi32>
      %linear_rows = muli %rows_2d, %stride_bc : tile<128x1xi32>
      %base_2d = reshape %a_base : tile<ptr<f16>> -> tile<1x1xptr<f16>>
      %base_row_bc = broadcast %base_2d : tile<1x1xptr<f16>> -> tile<128x1xptr<f16>>
      %row_ptr = offset %base_row_bc, %linear_rows : tile<128x1xptr<f16>>, tile<128x1xi32> -> tile<128x1xptr<f16>>
      %base_col_bc = broadcast %row_ptr : tile<128x1xptr<f16>> -> tile<128x64xptr<f16>>
      %col_bc = broadcast %cols_2d : tile<1x64xi32> -> tile<128x64xi32>
      %ptr = offset %base_col_bc, %col_bc : tile<128x64xptr<f16>>, tile<128x64xi32> -> tile<128x64xptr<f16>>
      %shape_m = reshape %arg1 : tile<i32> -> tile<1x1xi32>
      %shape_m_bc = broadcast %shape_m : tile<1x1xi32> -> tile<128x1xi32>
      %row_mask = cmpi less_than %rows_2d, %shape_m_bc, signed : tile<128x1xi32> -> tile<128x1xi1>
      %shape_n = reshape %arg2 : tile<i32> -> tile<1x1xi32>
      %shape_n_bc = broadcast %shape_n : tile<1x1xi32> -> tile<1x64xi32>
      %col_mask = cmpi less_than %cols_2d, %shape_n_bc, signed : tile<1x64xi32> -> tile<1x64xi1>
      %row_mask_bc = broadcast %row_mask : tile<128x1xi1> -> tile<128x64xi1>
      %col_mask_bc = broadcast %col_mask : tile<1x64xi1> -> tile<128x64xi1>
      %row_mask_i16 = exti %row_mask_bc signed : tile<128x64xi1> -> tile<128x64xi16>
      %col_mask_i16 = exti %col_mask_bc signed : tile<128x64xi1> -> tile<128x64xi16>
      %mask_i16 = andi %row_mask_i16, %col_mask_i16 : tile<128x64xi16>
      %mask = trunci %mask_i16 : tile<128x64xi16> -> tile<128x64xi1>
      // CHECK: %[[AB2:.*]] = assume div_by<16>, %arg0 : tile<ptr<f16>>
      // CHECK: %[[AST2:.*]] = assume div_by<8>, %arg3 : tile<i32>
      // CHECK: %[[TVF2:.*]] = make_tensor_view %[[AB2]], shape = [%arg1, %arg2], strides = [%[[AST2]], 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
      // CHECK: make_partition_view %[[TVF2]]
      // CHECK: store_view_tko weak
      // The unannotated shape scalars must not gain a fabricated assume.
      // CHECK-NOT: assume {{.*}}, %arg1
      // CHECK-NOT: assume {{.*}}, %arg2
      %token = store_ptr_tko weak %ptr, %value, %mask : tile<128x64xptr<f16>>, tile<128x64xf16>, tile<128x64xi1> -> token
      return
    }

    // CHECK-LABEL: @test_barrier_layer_norm_bwd 
    // CHECK-SAME: [[varg0:%.*]]: tile<ptr<f16>>, [[varg1:%.*]]: tile<i32>, [[varg2:%.*]]: tile<i32>) {
    entry @test_barrier_layer_norm_bwd(%arg4: tile<ptr<f16>>, %arg9: tile<i32>, %arg10: tile<i32>) {
      
      // CHECK: [[vassume:%.*]] = assume div_by<16>, [[varg2]] : tile<i32>
      %assume = assume div_by<16>, %arg10 : tile<i32>
      // CHECK: [[vassume_0:%.*]] = assume div_by<16>, [[varg1]] : tile<i32>
      %assume_0 = assume div_by<16>, %arg9 : tile<i32>
      // CHECK: [[vassume_1:%.*]] = assume div_by<16>, [[varg0]] : tile<ptr<f16>>
      %assume_5 = assume div_by<16>, %arg4 : tile<ptr<f16>>
      // CHECK: [[vblockId_x:%.*]], [[vblockId_y:%.*]], [[vblockId_z:%.*]] = get_tile_block_id : tile<i32>
      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      
      // CHECK: [[v0:%.*]] = muli [[vblockId_x]], [[vassume_0]] : tile<i32>
      // CHECK: [[v1:%.*]] = offset [[vassume_1]], [[v0]] : tile<ptr<f16>>, tile<i32> -> tile<ptr<f16>>
      // CHECK: [[vtview:%.*]] = make_tensor_view [[v1]], shape = [[[vassume]]], strides = [1] : tile<i32> -> tensor_view<?xf16, strides=[1]>
      // CHECK: [[vpview:%.*]] = make_partition_view [[vtview]] : partition_view<tile=(1024), padding_value = zero, tensor_view<?xf16, strides=[1]>>
      // CHECK: [[vcst_0_i32:%.*]] = constant <i32: 0> : tile<i32>
      // CHECK: [[vtile:%.*]], [[vresult_token:%.*]] = load_view_tko weak [[vpview]][[[vcst_0_i32]]] : partition_view<tile=(1024), padding_value = zero, tensor_view<?xf16, strides=[1]>>, tile<i32> -> tile<1024xf16>, token

      %0 = iota : tile<1024xi32>
      %reshape = reshape %assume : tile<i32> -> tile<1xi32>
      %bcast = broadcast %reshape : tile<1xi32> -> tile<1024xi32>
      %1 = cmpi less_than %0, %bcast, signed : tile<1024xi32> -> tile<1024xi1>
      %2 = muli %blockId_x, %assume_0 : tile<i32>
      %3 = offset %assume_5, %2 : tile<ptr<f16>>, tile<i32> -> tile<ptr<f16>>
      %reshape_14 = reshape %3 : tile<ptr<f16>> -> tile<1xptr<f16>>
      %bcast_15 = broadcast %reshape_14 : tile<1xptr<f16>> -> tile<1024xptr<f16>>
      %14 = offset %bcast_15, %0 : tile<1024xptr<f16>>, tile<1024xi32> -> tile<1024xptr<f16>>
      %cst_0_f16 = constant <f16: 0.000000e+00> : tile<1024xf16>
      %result, %result_token = load_ptr_tko weak %14, %1, %cst_0_f16 : tile<1024xptr<f16>>, tile<1024xi1>, tile<1024xf16> -> tile<1024xf16>, token
    }
  }
}
