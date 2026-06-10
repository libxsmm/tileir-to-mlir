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
      return
    }

    // CHECK-LABEL: entry @load_loop_iv_base
    // CHECK-GPU-LABEL: gpu.func @load_loop_iv_base
    entry @load_loop_iv_base(%arg0: tile<ptr<f32>>, %N: tile<i32>) {
      %c0 = constant <i32: 0> : tile<i32>
      %c32 = constant <i32: 32> : tile<i32>
      %iota = iota : tile<32xi32>
      %N_1d = reshape %N : tile<i32> -> tile<1xi32>
      %N_bc = broadcast %N_1d : tile<1xi32> -> tile<32xi32>
      
      // CHECK: %[[TV:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1]
      // CHECK: %[[PV:.*]] = make_partition_view %[[TV]] : partition_view<tile=(32), {{.*}}>
      // CHECK: for %[[IV:.*]] in
      // CHECK:   %[[C32:.*]] = constant <i32: 32> : tile<i32>
      // CHECK:   %[[DIV:.*]] = divi %[[IV]], %[[C32]] unsigned
      // CHECK:   load_view_tko weak %[[PV]][%[[DIV]]]
      for %iv in (%c0 to %N, step %c32) : tile<i32> {
        %iv_1d = reshape %iv : tile<i32> -> tile<1xi32>
        %iv_bc = broadcast %iv_1d : tile<1xi32> -> tile<32xi32>
        %off = addi %iv_bc, %iota : tile<32xi32>
        %mask = cmpi less_than %off, %N_bc, signed : tile<32xi32> -> tile<32xi1>
        %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
        %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<32xptr<f32>>
        %ptr = offset %base_bc, %off : tile<32xptr<f32>>, tile<32xi32> -> tile<32xptr<f32>>
        %tile, %tok = load_ptr_tko weak %ptr, %mask : tile<32xptr<f32>>, tile<32xi1> -> tile<32xf32>, token
        continue
      }
      return
    }

    // -----------------------------------------------------------------------
    // 1-D load with addi(broadcast(start), iota) offset: PtrToView should
    // lift this even though flattenOffset splits the addi at the top level.
    // -----------------------------------------------------------------------
    // CHECK-LABEL: entry @load_1d_bias
    // CHECK-SAME: (%[[BASE_B:.*]]: tile<ptr<bf16>>, %[[N_B:.*]]: tile<i32>)
    // CHECK-GPU-LABEL: gpu.func @load_1d_bias
    // CHECK-GPU-SAME: (%[[GBASE:.*]]: memref<*xbf16>, %[[GN:.*]]: i32)
    entry @load_1d_bias(%base: tile<ptr<bf16>>, %N: tile<i32>) {
      %iota = iota : tile<32xi32>
      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      %c32 = constant <i32: 32> : tile<i32>
      %start = muli %blockId_x, %c32 : tile<i32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<32xi32>
      %index = addi %start_bc, %iota : tile<32xi32>
      %N_1d = reshape %N : tile<i32> -> tile<1xi32>
      %N_bc = broadcast %N_1d : tile<1xi32> -> tile<32xi32>
      %mask = cmpi less_than %index, %N_bc, signed : tile<32xi32> -> tile<32xi1>
      %base_1d = reshape %base : tile<ptr<bf16>> -> tile<1xptr<bf16>>
      %base_bc = broadcast %base_1d : tile<1xptr<bf16>> -> tile<32xptr<bf16>>
      %ptr = offset %base_bc, %index : tile<32xptr<bf16>>, tile<32xi32> -> tile<32xptr<bf16>>
      %cst_pad = constant <bf16: 0.000000e+00> : tile<32xbf16>
      // CHECK: %[[TV_B:.*]] = make_tensor_view %[[BASE_B]], shape = [%[[N_B]]], strides = [1] : tile<i32> -> tensor_view<?xbf16, strides=[1]>
      // CHECK: %[[PV_B:.*]] = make_partition_view %[[TV_B]] : partition_view<tile=(32), padding_value = zero, tensor_view<?xbf16, strides=[1]>>
      // CHECK: %[[BLKID:.*]], %[[BLKY:.*]], %[[BLKZ:.*]] = get_tile_block_id : tile<i32>
      // CHECK: %[[LD_B:.*]], %[[TOK_B:.*]] = load_view_tko weak %[[PV_B]][%[[BLKID]]] : partition_view<tile=(32), padding_value = zero, tensor_view<?xbf16, strides=[1]>>, tile<i32> -> tile<32xbf16>, token
      // The original pointer-arithmetic ops must be fully gone after lifting.
      // CHECK-NOT: load_ptr_tko
      // CHECK-NOT: offset
      // CHECK-GPU: %[[GSIZE:.*]] = arith.index_cast %[[GN]] : i32 to index
      // CHECK-GPU: %[[GVIEW:.*]] = memref.reinterpret_cast %[[GBASE]] to offset: [0], sizes: [%[[GSIZE]]], strides: [1] : memref<*xbf16> to memref<?xbf16, strided<[1]>>
      // CHECK-GPU: %[[GPAD:.*]] = arith.constant 0.000000e+00 : bf16
      // CHECK-GPU: %{{.*}} = vector.transfer_read %[[GVIEW]][%{{.*}}], %[[GPAD]] : memref<?xbf16, strided<[1]>>, vector<32xbf16>
      %v, %t = load_ptr_tko weak %ptr, %mask, %cst_pad : tile<32xptr<bf16>>, tile<32xi1>, tile<32xbf16> -> tile<32xbf16>, token
      return
    }

    // -----------------------------------------------------------------------
    // 2-D non-affine pointer arithmetic (conv2d-style): PtrToView cannot lift
    // this, so it falls through to the gather/scatter lowering.
    // -----------------------------------------------------------------------
    // CHECK-LABEL: entry @gather_2d_conv
    // CHECK-NOT: make_tensor_view
    // CHECK-GPU-LABEL: gpu.func @gather_2d_conv
    // CHECK-GPU-SAME: (%[[SBASE:.*]]: memref<*xbf16>,
    entry @gather_2d_conv(%base: tile<ptr<bf16>>, %stride0: tile<i32>, %stride1: tile<i32>, %N: tile<i32>, %M: tile<i32>) {
      %iota = iota : tile<32xi32>
      // Compute 2D index via divmod (non-affine — can't be lifted to a view)
      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      %c32 = constant <i32: 32> : tile<i32>
      %start = muli %blockId_x, %c32 : tile<i32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<32xi32>
      %linear = addi %start_bc, %iota : tile<32xi32>
      // row = linear / M, col = linear % M
      %M_1d = reshape %M : tile<i32> -> tile<1xi32>
      %M_bc = broadcast %M_1d : tile<1xi32> -> tile<32xi32>
      %row = divi %linear, %M_bc signed : tile<32xi32>
      %col = remi %linear, %M_bc signed : tile<32xi32>
      // Build 2D offset: row[i]*stride0 + col[j]*stride1
      %row_2d = reshape %row : tile<32xi32> -> tile<32x1xi32>
      %row_bc = broadcast %row_2d : tile<32x1xi32> -> tile<32x32xi32>
      %stride0_rs = reshape %stride0 : tile<i32> -> tile<1x1xi32>
      %stride0_bc = broadcast %stride0_rs : tile<1x1xi32> -> tile<32x32xi32>
      %off0 = muli %row_bc, %stride0_bc : tile<32x32xi32>
      %col_2d = reshape %col : tile<32xi32> -> tile<1x32xi32>
      %col_bc = broadcast %col_2d : tile<1x32xi32> -> tile<32x32xi32>
      %stride1_rs = reshape %stride1 : tile<i32> -> tile<1x1xi32>
      %stride1_bc = broadcast %stride1_rs : tile<1x1xi32> -> tile<32x32xi32>
      %off1 = muli %col_bc, %stride1_bc : tile<32x32xi32>
      %total_off = addi %off0, %off1 : tile<32x32xi32>
      // Build pointer tile
      %base_rs = reshape %base : tile<ptr<bf16>> -> tile<1x1xptr<bf16>>
      %base_bc = broadcast %base_rs : tile<1x1xptr<bf16>> -> tile<32x32xptr<bf16>>
      %ptr = offset %base_bc, %total_off : tile<32x32xptr<bf16>>, tile<32x32xi32> -> tile<32x32xptr<bf16>>
      // Mask
      %N_1d = reshape %N : tile<i32> -> tile<1xi32>
      %N_bc = broadcast %N_1d : tile<1xi32> -> tile<32xi32>
      %mask_row = cmpi less_than %row, %N_bc, signed : tile<32xi32> -> tile<32xi1>
      %mask_col = cmpi less_than %col, %M_bc, signed : tile<32xi32> -> tile<32xi1>
      %mr_2d = reshape %mask_row : tile<32xi1> -> tile<32x1xi1>
      %mr_bc = broadcast %mr_2d : tile<32x1xi1> -> tile<32x32xi1>
      %mc_2d = reshape %mask_col : tile<32xi1> -> tile<1x32xi1>
      %mc_bc = broadcast %mc_2d : tile<1x32xi1> -> tile<32x32xi1>
      %mr_ext = exti %mr_bc signed : tile<32x32xi1> -> tile<32x32xi16>
      %mc_ext = exti %mc_bc signed : tile<32x32xi1> -> tile<32x32xi16>
      %mask_and = andi %mr_ext, %mc_ext : tile<32x32xi16>
      %mask = trunci %mask_and : tile<32x32xi16> -> tile<32x32xi1>
      // Load (gather)
      %pad = constant <bf16: 0.000000e+00> : tile<32x32xbf16>
      // The pointer tile is kept as an explicit offset op (no view lifting).
      // CHECK: %[[OFF:.*]] = offset %{{.*}}, %{{.*}} : tile<32x32xptr<bf16>>, tile<32x32xi32> -> tile<32x32xptr<bf16>>
      // CHECK: %[[GV:.*]], %[[GTOK:.*]] = load_ptr_tko weak %[[OFF]], %[[GMASK:.*]], %{{.*}} : tile<32x32xptr<bf16>>, tile<32x32xi1>, tile<32x32xbf16> -> tile<32x32xbf16>, token
      // CHECK-GPU-DAG: %[[GBASE:.*]] = memref.cast %[[SBASE]] : memref<*xbf16> to memref<?xbf16>
      // CHECK-GPU-DAG: arith.constant 0 : index
      // CHECK-GPU: %[[GATHERED:.*]] = vector.gather %[[GBASE]][%{{.*}}] [%{{.*}}], %{{.*}}, %{{.*}} : memref<?xbf16>, vector<32x32xindex>, vector<32x32xi1>, vector<32x32xbf16> into vector<32x32xbf16>
      %v, %t = load_ptr_tko weak %ptr, %mask, %pad : tile<32x32xptr<bf16>>, tile<32x32xi1>, tile<32x32xbf16> -> tile<32x32xbf16>, token
      // Store (scatter)
      // CHECK: store_ptr_tko weak %[[OFF]], %[[GV]], %[[GMASK]] : tile<32x32xptr<bf16>>, tile<32x32xbf16>, tile<32x32xi1> -> token
      // CHECK-GPU: vector.scatter %{{.*}}[%{{.*}}] [%{{.*}}], %{{.*}}, %[[GATHERED]] : memref<?xbf16>, vector<32x32xindex>, vector<32x32xi1>, vector<32x32xbf16>
      %57 = store_ptr_tko weak %ptr, %v, %mask : tile<32x32xptr<bf16>>, tile<32x32xbf16>, tile<32x32xi1> -> token
      return
    }

    // -----------------------------------------------------------------------
    // Argmax-style reduction loop. The running-max iter_arg (#0) is never used
    // outside the loop, but it is internally live: it feeds the comparison that
    // drives the kept argmax-index iter_arg (#1). The dead-iter-arg cleanup in
    // ptr-to-view must NOT drop iter_arg #0 (doing so used to erase the shared
    // comparison and produce a null select operand). Both iter_args and both
    // selects must survive, and the load (non-liftable) becomes a gather.
    // -----------------------------------------------------------------------
    // CHECK-LABEL: entry @argmax_loop
    // CHECK: %[[FOR:.*]]:2 = for %{{.*}} in {{.*}} -> (tile<32xf32>, tile<32xi32>)
    // CHECK:   load_ptr_tko
    // CHECK:   %[[CMP:.*]] = cmpf greater_than ordered
    // CHECK:   %[[NMAX:.*]] = select %[[CMP]]
    // CHECK:   %[[NIDX:.*]] = select %[[CMP]]
    // CHECK:   continue %[[NMAX]], %[[NIDX]]
    // CHECK-GPU-LABEL: gpu.func @argmax_loop
    // CHECK-GPU: %[[GFOR:.*]]:2 = scf.for %{{.*}} iter_args(%[[M:.*]] = %{{.*}}, %[[I:.*]] = %{{.*}}) -> (vector<32xf32>, vector<32xi32>)
    // CHECK-GPU:   vector.gather
    // CHECK-GPU:   %[[GCMP:.*]] = arith.cmpf ogt
    // CHECK-GPU:   %[[GMAX:.*]] = arith.select %[[GCMP]]
    // CHECK-GPU:   %[[GIDX:.*]] = arith.select %[[GCMP]]
    // CHECK-GPU:   scf.yield %[[GMAX]], %[[GIDX]]
    entry @argmax_loop(%base: tile<ptr<f32>>, %out: tile<ptr<i32>>, %N: tile<i32>, %stride: tile<i32>) {
      %c0 = constant <i32: 0> : tile<i32>
      %c1 = constant <i32: 1> : tile<i32>
      %c4 = constant <i32: 4> : tile<i32>
      %c32 = constant <i32: 32> : tile<i32>
      %neg_inf = constant <f32: 0xFF800000> : tile<32xf32>
      %zero_idx = constant <i32: 0> : tile<32xi32>
      %iota = iota : tile<32xi32>
      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      %start = muli %blockId_x, %c32 : tile<i32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<32xi32>
      %index = addi %start_bc, %iota : tile<32xi32>
      %N_1d = reshape %N : tile<i32> -> tile<1xi32>
      %N_bc = broadcast %N_1d : tile<1xi32> -> tile<32xi32>
      %mask = cmpi less_than %index, %N_bc, signed : tile<32xi32> -> tile<32xi1>
      %base_1d = reshape %base : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<32xptr<f32>>
      %for:2 = for %i in (%c0 to %c4, step %c1) : tile<i32>
          iter_values(%curMax = %neg_inf, %curIdx = %zero_idx) -> (tile<32xf32>, tile<32xi32>) {
        %off_s = muli %i, %stride : tile<i32>
        %off_1d = reshape %off_s : tile<i32> -> tile<1xi32>
        %off_bc = broadcast %off_1d : tile<1xi32> -> tile<32xi32>
        %ptr = offset %base_bc, %off_bc : tile<32xptr<f32>>, tile<32xi32> -> tile<32xptr<f32>>
        %v, %t = load_ptr_tko weak %ptr, %mask, %neg_inf : tile<32xptr<f32>>, tile<32xi1>, tile<32xf32> -> tile<32xf32>, token
        %gt = cmpf greater_than ordered %v, %curMax : tile<32xf32> -> tile<32xi1>
        %newMax = select %gt, %v, %curMax : tile<32xi1>, tile<32xf32>
        %i_1d = reshape %i : tile<i32> -> tile<1xi32>
        %i_bc = broadcast %i_1d : tile<1xi32> -> tile<32xi32>
        %newIdx = select %gt, %i_bc, %curIdx : tile<32xi1>, tile<32xi32>
        continue %newMax, %newIdx : tile<32xf32>, tile<32xi32>
      }
      // Only the argmax index (result #1) is consumed; the running max (#0) is dead.
      %out_1d = reshape %out : tile<ptr<i32>> -> tile<1xptr<i32>>
      %out_bc = broadcast %out_1d : tile<1xptr<i32>> -> tile<32xptr<i32>>
      %sptr = offset %out_bc, %index : tile<32xptr<i32>>, tile<32xi32> -> tile<32xptr<i32>>
      %st = store_ptr_tko weak %sptr, %for#1, %mask : tile<32xptr<i32>>, tile<32xi32>, tile<32xi1> -> token
      return
    }

    // -----------------------------------------------------------------------
    // Matmul-style loop-carried pointer (Triton LHS access). The pointer is a
    // `for` iter_arg that advances along the contiguous K dimension by exactly
    // one tile each iteration. The load masks ONLY the K dimension, written in
    // residual form `K - loopIdx*16`; the M dimension is unmasked. PtrToView
    // must:
    //   * recover the absolute K extent (%K) from the residual bound,
    //   * give the unmasked M dimension a static (in-bounds) extent of 64,
    //   * use blockId for M and the induction var for K as partition indices,
    //   * hoist the now loop-invariant view out of the loop.
    // The advance (16 elements) equals step(1) * tileSize(16) * stride(1), so
    // the raw induction variable is a faithful K partition index.
    // -----------------------------------------------------------------------
    // CHECK-LABEL: entry @matmul_lhs_loop
    // CHECK-GPU-LABEL: gpu.func @matmul_lhs_loop
    entry @matmul_lhs_loop(%A: tile<ptr<f32>>, %M: tile<i32>, %K: tile<i32>, %stride_am: tile<i32>) {
      %c0 = constant <i32: 0> : tile<i32>
      %c1 = constant <i32: 1> : tile<i32>
      %c16 = constant <i32: 16> : tile<i32>
      %c64 = constant <i32: 64> : tile<i32>
      %c16_2d = constant <i32: 16> : tile<64x16xi32>
      %pad = constant <f32: 0.000000e+00> : tile<64x16xf32>

      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      %row_start = muli %blockId_x, %c64 : tile<i32>

      // dim0 = (row_start + iota) * stride_am.
      %iota64 = iota : tile<64xi32>
      %rs_1d = reshape %row_start : tile<i32> -> tile<1xi32>
      %rs_bc = broadcast %rs_1d : tile<1xi32> -> tile<64xi32>
      %rows = addi %rs_bc, %iota64 : tile<64xi32>
      %rows_2d = reshape %rows : tile<64xi32> -> tile<64x1xi32>
      %stride_2d = reshape %stride_am : tile<i32> -> tile<1x1xi32>
      %stride_bc = broadcast %stride_2d : tile<1x1xi32> -> tile<64x1xi32>
      %rows_strided = muli %rows_2d, %stride_bc : tile<64x1xi32>

      // dim1 = iota (contiguous, no start).
      %iota16 = iota : tile<16xi32>
      %cols_2d = reshape %iota16 : tile<16xi32> -> tile<1x16xi32>

      %rows_bc = broadcast %rows_strided : tile<64x1xi32> -> tile<64x16xi32>
      %cols_bc = broadcast %cols_2d : tile<1x16xi32> -> tile<64x16xi32>
      %off = addi %rows_bc, %cols_bc : tile<64x16xi32>
      %A_2d = reshape %A : tile<ptr<f32>> -> tile<1x1xptr<f32>>
      %A_bc = broadcast %A_2d : tile<1x1xptr<f32>> -> tile<64x16xptr<f32>>
      %ptr_init = offset %A_bc, %off : tile<64x16xptr<f32>>, tile<64x16xi32> -> tile<64x16xptr<f32>>

      // The view is loop-invariant: M is a static tile extent (unmasked), K is
      // the absolute extent recovered from the residual mask, row-major stride.
      // CHECK: %[[TV:.*]] = make_tensor_view %{{.*}}, shape = [64, %{{.*}}], strides = [%{{.*}}, 1] : tile<i32> -> tensor_view<64x?xf32, strides=[?,1]>
      // CHECK: %[[PV:.*]] = make_partition_view %[[TV]] : partition_view<tile=(64x16), {{.*}}>
      // CHECK: for %[[IV:.*]] in
      // CHECK:   load_view_tko weak %[[PV]][%{{.*}}, %[[IV]]]
      // CHECK-NOT: load_ptr_tko
      // CHECK-GPU: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [64, %{{.*}}], strides: [%{{.*}}, 1]
      // CHECK-GPU: scf.for
      // CHECK-GPU:   vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, false]}
      %for = for %loopIdx in (%c0 to %K, step %c1) : tile<i32>
          iter_values(%iterPtr = %ptr_init) -> (tile<64x16xptr<f32>>) {
        // residual mask: iota16 < (K - loopIdx*16).
        %koff = muli %loopIdx, %c16 : tile<i32>
        %resid = subi %K, %koff : tile<i32>
        %resid_2d = reshape %resid : tile<i32> -> tile<1x1xi32>
        %resid_bc = broadcast %resid_2d : tile<1x1xi32> -> tile<1x16xi32>
        %kcmp = cmpi less_than %cols_2d, %resid_bc, signed : tile<1x16xi32> -> tile<1x16xi1>
        %mask = broadcast %kcmp : tile<1x16xi1> -> tile<64x16xi1>
        %v, %t = load_ptr_tko weak %iterPtr, %mask, %pad : tile<64x16xptr<f32>>, tile<64x16xi1>, tile<64x16xf32> -> tile<64x16xf32>, token
        %next = offset %iterPtr, %c16_2d : tile<64x16xptr<f32>>, tile<64x16xi32> -> tile<64x16xptr<f32>>
        continue %next : tile<64x16xptr<f32>>
      }
      return
    }
  }
}
