// RUN: tileir-to-mlir --tileir-ptr-to-view %s | FileCheck %s
// RUN: tileir-to-mlir --tileir-ptr-to-view --convert-tileir-to-mlir %s | FileCheck %s --check-prefix=CHECK-GPU

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
      // CHECK-GPU: %[[LOAD1D_VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%{{.*}}], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1], offset: ?>>
      // CHECK-GPU: %[[LOAD1D_TILE:.*]] = vector.transfer_read %[[LOAD1D_VIEW]][%{{.*}}], %{{.*}} : memref<?xf32, strided<[1], offset: ?>>, vector<1024xf32>
      %tile, %token = load_ptr_tko weak %ptr, %mask : tile<1024xptr<f32>>, tile<1024xi1> -> tile<1024xf32>, !cuda_tile.token
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
      // CHECK-GPU: %[[STORE1D_VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%{{.*}}], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1], offset: ?>>
      // CHECK-GPU: vector.transfer_write %{{.*}}, %[[STORE1D_VIEW]][%{{.*}}] : vector<1024xf32>, memref<?xf32, strided<[1], offset: ?>>
      %token = store_ptr_tko weak %ptr, %value, %mask : tile<1024xptr<f32>>, tile<1024xf32>, tile<1024xi1> -> !cuda_tile.token
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
      // CHECK-GPU: %[[STORE2D_VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%{{.*}}, %{{.*}}], strides: [%{{.*}}, 1] : memref<*xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      // CHECK-GPU: vector.transfer_write %{{.*}}, %[[STORE2D_VIEW]][%{{.*}}, %{{.*}}] : vector<128x64xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
      %token = store_ptr_tko weak %ptr, %value, %mask : tile<128x64xptr<f16>>, tile<128x64xf16>, tile<128x64xi1> -> !cuda_tile.token
      return
    }

    // The per-dim bound is computed by comparing a *1-D* index and only then
    // reshaping the i1 *result* into tile space (the reshape sits between the
    // broadcast and the cmpi).  This is the canonical Triton 2-D mask; the
    // dimension each comparison constrains is carried by that result reshape,
    // not by the (dimensionless) comparison operand.  Recovering it lets the
    // access lower to a contiguous 2-D transfer instead of a scalarized gather.
    // CHECK-LABEL: entry @load_2d_mask_result_reshape
    // CHECK-GPU-LABEL: gpu.func @load_2d_mask_result_reshape
    entry @load_2d_mask_result_reshape(%arg0: tile<ptr<bf16>>, %arg1: tile<i32>, %arg2: tile<i32>, %arg3: tile<i32>) {
      %cst_32 = constant <i32: 32> : tile<i32>
      %pad = constant <bf16: 0.000000e+00> : tile<32x32xbf16>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %row_start = muli %block_id_x, %cst_32 : tile<i32>
      %col_start = muli %block_id_y, %cst_32 : tile<i32>
      %row_lane = iota : tile<32xi32>
      %col_lane = iota : tile<32xi32>
      %row_start_1d = reshape %row_start : tile<i32> -> tile<1xi32>
      %row_start_bc = broadcast %row_start_1d : tile<1xi32> -> tile<32xi32>
      %rows = addi %row_start_bc, %row_lane : tile<32xi32>
      %col_start_1d = reshape %col_start : tile<i32> -> tile<1xi32>
      %col_start_bc = broadcast %col_start_1d : tile<1xi32> -> tile<32xi32>
      %cols = addi %col_start_bc, %col_lane : tile<32xi32>
      // Masks compare 1-D indices, then reshape the i1 result to tile space.
      %shape_m_1d = reshape %arg1 : tile<i32> -> tile<1xi32>
      %shape_m_bc = broadcast %shape_m_1d : tile<1xi32> -> tile<32xi32>
      %row_mask = cmpi less_than %rows, %shape_m_bc, signed : tile<32xi32> -> tile<32xi1>
      %shape_n_1d = reshape %arg2 : tile<i32> -> tile<1xi32>
      %shape_n_bc = broadcast %shape_n_1d : tile<1xi32> -> tile<32xi32>
      %col_mask = cmpi less_than %cols, %shape_n_bc, signed : tile<32xi32> -> tile<32xi1>
      %row_mask_2d = reshape %row_mask : tile<32xi1> -> tile<32x1xi1>
      %col_mask_2d = reshape %col_mask : tile<32xi1> -> tile<1x32xi1>
      %row_mask_bc = broadcast %row_mask_2d : tile<32x1xi1> -> tile<32x32xi1>
      %col_mask_bc = broadcast %col_mask_2d : tile<1x32xi1> -> tile<32x32xi1>
      %row_mask_i16 = exti %row_mask_bc signed : tile<32x32xi1> -> tile<32x32xi16>
      %col_mask_i16 = exti %col_mask_bc signed : tile<32x32xi1> -> tile<32x32xi16>
      %mask_i16 = andi %row_mask_i16, %col_mask_i16 : tile<32x32xi16>
      %mask = trunci %mask_i16 : tile<32x32xi16> -> tile<32x32xi1>
      // Row-major pointers: rows carry the runtime stride, cols are contiguous.
      %rows_2d = reshape %rows : tile<32xi32> -> tile<32x1xi32>
      %cols_2d = reshape %cols : tile<32xi32> -> tile<1x32xi32>
      %stride_2d = reshape %arg3 : tile<i32> -> tile<1x1xi32>
      %stride_bc = broadcast %stride_2d : tile<1x1xi32> -> tile<32x1xi32>
      %linear_rows = muli %rows_2d, %stride_bc : tile<32x1xi32>
      %base_2d = reshape %arg0 : tile<ptr<bf16>> -> tile<1x1xptr<bf16>>
      %base_row_bc = broadcast %base_2d : tile<1x1xptr<bf16>> -> tile<32x1xptr<bf16>>
      %row_ptr = offset %base_row_bc, %linear_rows : tile<32x1xptr<bf16>>, tile<32x1xi32> -> tile<32x1xptr<bf16>>
      %base_col_bc = broadcast %row_ptr : tile<32x1xptr<bf16>> -> tile<32x32xptr<bf16>>
      %col_bc = broadcast %cols_2d : tile<1x32xi32> -> tile<32x32xi32>
      %ptr = offset %base_col_bc, %col_bc : tile<32x32xptr<bf16>>, tile<32x32xi32> -> tile<32x32xptr<bf16>>
      // CHECK: %[[TV2R:.*]] = make_tensor_view %arg0, shape = [%arg1, %arg2], strides = [%arg3, 1] : tile<i32> -> tensor_view<?x?xbf16, strides=[?,1]>
      // CHECK: %[[PV2R:.*]] = make_partition_view %[[TV2R]] : partition_view<tile=(32x32), padding_value = zero, tensor_view<?x?xbf16, strides=[?,1]>>
      // CHECK: %[[LD2R:.*]], %[[TOK2R:.*]] = load_view_tko weak %[[PV2R]][%{{.*}}, %{{.*}}] : partition_view<tile=(32x32), padding_value = zero, tensor_view<?x?xbf16, strides=[?,1]>>, tile<i32> -> tile<32x32xbf16>, token
      // CHECK-NOT: load_ptr_tko
      // CHECK-GPU: %[[LOAD2DR_VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%{{.*}}, %{{.*}}], strides: [%{{.*}}, 1] : memref<*xbf16> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
      // CHECK-GPU: %[[LOAD2DR_TILE:.*]] = vector.transfer_read %[[LOAD2DR_VIEW]][%{{.*}}, %{{.*}}], %{{.*}} : memref<?x?xbf16, strided<[?, 1], offset: ?>>, vector<32x32xbf16>
      // CHECK-GPU-NOT: vector.gather
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<32x32xptr<bf16>>, tile<32x32xi1>, tile<32x32xbf16> -> tile<32x32xbf16>, !cuda_tile.token
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
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, !cuda_tile.token
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
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, !cuda_tile.token
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
      // CHECK-GPU: vector.transfer_read %{{.*}}[%{{.*}}], %[[NANPAD]] : memref<?xf32, strided<[1], offset: ?>>, vector<1024xf32>
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, !cuda_tile.token
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
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, !cuda_tile.token
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
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, !cuda_tile.token
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
            -> tile<128x32xf16>, !cuda_tile.token
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
      %tile, %token = load_ptr_tko weak %ptr, %mask : tile<1024xptr<f32>>, tile<1024xi1> -> tile<1024xf32>, !cuda_tile.token
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
      %token = store_ptr_tko weak %ptr, %value, %mask : tile<128x64xptr<f16>>, tile<128x64xf16>, tile<128x64xi1> -> !cuda_tile.token
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
      %result, %result_token = load_ptr_tko weak %14, %1, %cst_0_f16 : tile<1024xptr<f16>>, tile<1024xi1>, tile<1024xf16> -> tile<1024xf16>, !cuda_tile.token
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
      // The lowered transfer initially contains the exact casted round trip
      // `muli(index_cast(divui(index_cast(iv), 32)), 32)`. The cleanup must
      // replace it with an unsigned widening of the original i32 loop IV.
      // CHECK-GPU: scf.for %[[LOOP_IV:.*]] =
      // CHECK-GPU:   %[[LOOP_IV_I32:.*]] = arith.index_cast %[[LOOP_IV]] : index to i32
      // CHECK-GPU:   %[[DIV_I32:.*]] = arith.divui %[[LOOP_IV_I32]], %{{.*}} : i32
      // CHECK-GPU:   %[[DIV_INDEX:.*]] = arith.index_cast %[[DIV_I32]] : i32 to index
      // CHECK-GPU-NOT: arith.muli %[[DIV_INDEX]], {{.*}} : index
      // CHECK-GPU:   %[[FOLDED_INDEX:.*]] = arith.index_castui %[[LOOP_IV_I32]] : i32 to index
      // CHECK-GPU:   vector.transfer_read %{{.*}}[%[[FOLDED_INDEX]]], %{{.*}} : memref<?xf32, strided<[1], offset: ?>>, vector<32xf32>
      for %iv in (%c0 to %N, step %c32) : tile<i32> {
        %iv_1d = reshape %iv : tile<i32> -> tile<1xi32>
        %iv_bc = broadcast %iv_1d : tile<1xi32> -> tile<32xi32>
        %off = addi %iv_bc, %iota : tile<32xi32>
        %mask = cmpi less_than %off, %N_bc, signed : tile<32xi32> -> tile<32xi1>
        %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
        %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<32xptr<f32>>
        %ptr = offset %base_bc, %off : tile<32xptr<f32>>, tile<32xi32> -> tile<32xptr<f32>>
        %tile, %tok = load_ptr_tko weak %ptr, %mask : tile<32xptr<f32>>, tile<32xi1> -> tile<32xf32>, !cuda_tile.token
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
      // CHECK-GPU: %[[GVIEW:.*]] = memref.reinterpret_cast %[[GBASE]] to offset: [0], sizes: [%[[GSIZE]]], strides: [1] : memref<*xbf16> to memref<?xbf16, strided<[1], offset: ?>>
      // CHECK-GPU: %[[GPAD:.*]] = arith.constant 0.000000e+00 : bf16
      // CHECK-GPU: %{{.*}} = vector.transfer_read %[[GVIEW]][%{{.*}}], %[[GPAD]] : memref<?xbf16, strided<[1], offset: ?>>, vector<32xbf16>
      %v, %t = load_ptr_tko weak %ptr, %mask, %cst_pad : tile<32xptr<bf16>>, tile<32xi1>, tile<32xbf16> -> tile<32xbf16>, !cuda_tile.token
      return
    }

    // A lane-invariant vector offset is part of the base address, not a
    // partition dimension.
    // CHECK-LABEL: entry @load_uniform_vector_shift
    // CHECK: %[[SHIFTED_BASE:.*]] = offset %{{.*}}, %{{.*}} : tile<ptr<f32>>, tile<i32> -> tile<ptr<f32>>
    // CHECK: make_tensor_view %[[SHIFTED_BASE]], shape = [{{.*}}], strides = [1]
    // CHECK: load_view_tko
    // CHECK-NOT: load_ptr_tko
    entry @load_uniform_vector_shift(%base: tile<ptr<f32>>, %N: tile<i32>, %shift: tile<i32>) {
      %c32 = constant <i32: 32> : tile<i32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %c32 : tile<i32>
      %lane = iota : tile<32xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<32xi32>
      %index = addi %start_bc, %lane : tile<32xi32>
      %shift_1d = reshape %shift : tile<i32> -> tile<1xi32>
      %shift_bc = broadcast %shift_1d : tile<1xi32> -> tile<32xi32>
      %offset = addi %shift_bc, %index : tile<32xi32>
      %N_1d = reshape %N : tile<i32> -> tile<1xi32>
      %N_bc = broadcast %N_1d : tile<1xi32> -> tile<32xi32>
      %mask = cmpi less_than %index, %N_bc, signed : tile<32xi32> -> tile<32xi1>
      %base_1d = reshape %base : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<32xptr<f32>>
      %ptr = offset %base_bc, %offset : tile<32xptr<f32>>, tile<32xi32> -> tile<32xptr<f32>>
      %pad = constant <f32: 0.000000e+00> : tile<32xf32>
      %value, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<32xptr<f32>>, tile<32xi1>, tile<32xf32> -> tile<32xf32>, !cuda_tile.token
      return
    }

    // The mask proves that the clamped index equals the original index.
    // CHECK-LABEL: entry @load_masked_clamp
    // CHECK: make_tensor_view %arg0, shape = [64], strides = [1] : tensor_view<64xf32, strides=[1]>
    // CHECK: load_view_tko
    // CHECK-NOT: load_ptr_tko
    entry @load_masked_clamp(%base: tile<ptr<f32>>) {
      %c8 = constant <i32: 8> : tile<i32>
      %c63 = constant <i32: 63> : tile<i32>
      %c64 = constant <i32: 64> : tile<i32>
      %c0 = constant <i32: 0> : tile<i32>
      %block, %block_y, %block_z = get_tile_block_id : tile<i32>
      %start = muli %block, %c8 : tile<i32>
      %lane = iota : tile<8xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<8xi32>
      %index = addi %start_bc, %lane : tile<8xi32>
      %zero_1d = reshape %c0 : tile<i32> -> tile<1xi32>
      %zero_bc = broadcast %zero_1d : tile<1xi32> -> tile<8xi32>
      %clamp_lower = maxi %index, %zero_bc signed : tile<8xi32>
      %upper_1d = reshape %c63 : tile<i32> -> tile<1xi32>
      %upper_bc = broadcast %upper_1d : tile<1xi32> -> tile<8xi32>
      %clamped_index = mini %clamp_lower, %upper_bc signed : tile<8xi32>
      %extent_1d = reshape %c64 : tile<i32> -> tile<1xi32>
      %extent_bc = broadcast %extent_1d : tile<1xi32> -> tile<8xi32>
      %lower_mask = cmpi greater_than_or_equal %index, %zero_bc, signed : tile<8xi32> -> tile<8xi1>
      %upper_mask = cmpi less_than %index, %extent_bc, signed : tile<8xi32> -> tile<8xi1>
      %mask = andi %lower_mask, %upper_mask : tile<8xi1>
      %base_1d = reshape %base : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<8xptr<f32>>
      %ptr = offset %base_bc, %clamped_index : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>
      %pad = constant <f32: 0.000000e+00> : tile<8xf32>
      %value, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<8xptr<f32>>, tile<8xi1>, tile<8xf32> -> tile<8xf32>, !cuda_tile.token
      return
    }

    // A uniform `start * stride` term belongs to its strided dimension, even
    // when the dimension is already represented as a singleton 2-D tile.
    // CHECK-LABEL: entry @load_2d_separate_strided_start
    // CHECK: %[[TV_SEPARATE_START:.*]] = make_tensor_view %arg0, shape = [%arg1, %arg2], strides = [8, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[8,1]>
    // CHECK: %[[PV_SEPARATE_START:.*]] = make_partition_view %[[TV_SEPARATE_START]] : partition_view<tile=(2x4), padding_value = zero, tensor_view<?x?xf32, strides=[8,1]>>
    // CHECK: load_view_tko weak %[[PV_SEPARATE_START]][%{{.*}}, %{{.*}}]
    // CHECK-NOT: load_ptr_tko
    entry @load_2d_separate_strided_start(%base: tile<ptr<f32>>, %rows: tile<i32>, %cols: tile<i32>) {
      %c2 = constant <i32: 2> : tile<i32>
      %c4 = constant <i32: 4> : tile<i32>
      %c8 = constant <i32: 8> : tile<i32>
      %block_x, %block_y, %block_z = get_tile_block_id : tile<i32>
      %row_start = muli %block_x, %c2 : tile<i32>
      %row_lane = iota : tile<2xi32>
      %row_start_1d = reshape %row_start : tile<i32> -> tile<1xi32>
      %row_start_bc = broadcast %row_start_1d : tile<1xi32> -> tile<2xi32>
      %row_index = addi %row_start_bc, %row_lane : tile<2xi32>
      %row_offset = muli %row_start, %c8 : tile<i32>
      %row_offset_2d = reshape %row_offset : tile<i32> -> tile<1x1xi32>
      %row_offset_bc = broadcast %row_offset_2d : tile<1x1xi32> -> tile<2x1xi32>
      %row_lane_2d = reshape %row_lane : tile<2xi32> -> tile<2x1xi32>
      %stride_2d = reshape %c8 : tile<i32> -> tile<1x1xi32>
      %stride_bc = broadcast %stride_2d : tile<1x1xi32> -> tile<2x1xi32>
      %row_elements = muli %row_lane_2d, %stride_bc : tile<2x1xi32>
      %row_offset_elements = addi %row_offset_bc, %row_elements : tile<2x1xi32>
      %row_offset_full = broadcast %row_offset_elements : tile<2x1xi32> -> tile<2x4xi32>
      %col_start = muli %block_y, %c4 : tile<i32>
      %col_lane = iota : tile<4xi32>
      %col_start_2d = reshape %col_start : tile<i32> -> tile<1x1xi32>
      %col_start_bc = broadcast %col_start_2d : tile<1x1xi32> -> tile<1x4xi32>
      %col_lane_2d = reshape %col_lane : tile<4xi32> -> tile<1x4xi32>
      %col_index = addi %col_start_bc, %col_lane_2d : tile<1x4xi32>
      %col_offset_full = broadcast %col_index : tile<1x4xi32> -> tile<2x4xi32>
      %offset = addi %row_offset_full, %col_offset_full : tile<2x4xi32>
      %rows_1d = reshape %rows : tile<i32> -> tile<1xi32>
      %rows_bc = broadcast %rows_1d : tile<1xi32> -> tile<2xi32>
      %row_mask = cmpi less_than %row_index, %rows_bc, signed : tile<2xi32> -> tile<2xi1>
      %row_mask_2d = reshape %row_mask : tile<2xi1> -> tile<2x1xi1>
      %row_mask_full = broadcast %row_mask_2d : tile<2x1xi1> -> tile<2x4xi1>
      %cols_2d = reshape %cols : tile<i32> -> tile<1x1xi32>
      %cols_bc = broadcast %cols_2d : tile<1x1xi32> -> tile<1x4xi32>
      %col_mask = cmpi less_than %col_index, %cols_bc, signed : tile<1x4xi32> -> tile<1x4xi1>
      %col_mask_full = broadcast %col_mask : tile<1x4xi1> -> tile<2x4xi1>
      %mask = andi %row_mask_full, %col_mask_full : tile<2x4xi1>
      %base_2d = reshape %base : tile<ptr<f32>> -> tile<1x1xptr<f32>>
      %base_bc = broadcast %base_2d : tile<1x1xptr<f32>> -> tile<2x4xptr<f32>>
      %ptr = offset %base_bc, %offset : tile<2x4xptr<f32>>, tile<2x4xi32> -> tile<2x4xptr<f32>>
      %pad = constant <f32: 0.000000e+00> : tile<2x4xf32>
      %value, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<2x4xptr<f32>>, tile<2x4xi1>, tile<2x4xf32> -> tile<2x4xf32>, !cuda_tile.token
      return
    }

    // A global-coordinate mask can be normalized against a uniform base shift.
    // CHECK-LABEL: entry @store_global_mask_base_shift
    // CHECK: %[[LOCAL_SIZE:.*]] = subi %{{.*}}, %{{.*}} : tile<i32>
    // CHECK: %[[NONNEG_SIZE:.*]] = maxi %[[LOCAL_SIZE]], %{{.*}} signed : tile<i32>
    // CHECK: %[[SHIFTED_BASE:.*]] = offset %{{.*}}, %{{.*}} : tile<ptr<f32>>, tile<i32> -> tile<ptr<f32>>
    // CHECK: make_tensor_view %[[SHIFTED_BASE]], shape = [%[[NONNEG_SIZE]]], strides = [1]
    // CHECK: store_view_tko
    // CHECK-NOT: store_ptr_tko
    entry @store_global_mask_base_shift(%base: tile<ptr<f32>>, %N: tile<i32>, %row: tile<i32>, %block: tile<i32>) {
      %c64 = constant <i32: 64> : tile<i32>
      %c255 = constant <i32: 255> : tile<i32>
      %row_shift = muli %row, %c255 : tile<i32>
      %block_start = muli %block, %c64 : tile<i32>
      %lane = iota : tile<64xi32>
      %block_start_1d = reshape %block_start : tile<i32> -> tile<1xi32>
      %block_start_bc = broadcast %block_start_1d : tile<1xi32> -> tile<64xi32>
      %local_index = addi %block_start_bc, %lane : tile<64xi32>
      %row_shift_1d = reshape %row_shift : tile<i32> -> tile<1xi32>
      %row_shift_bc = broadcast %row_shift_1d : tile<1xi32> -> tile<64xi32>
      %global_index = addi %row_shift_bc, %local_index : tile<64xi32>
      %N_1d = reshape %N : tile<i32> -> tile<1xi32>
      %N_bc = broadcast %N_1d : tile<1xi32> -> tile<64xi32>
      %global_index_i64 = exti %global_index signed : tile<64xi32> -> tile<64xi64>
      %N_i64 = exti %N_bc signed : tile<64xi32> -> tile<64xi64>
      %mask = cmpi less_than %global_index_i64, %N_i64, unsigned : tile<64xi64> -> tile<64xi1>
      %base_1d = reshape %base : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<64xptr<f32>>
      %ptr = offset %base_bc, %global_index : tile<64xptr<f32>>, tile<64xi32> -> tile<64xptr<f32>>
      %value = constant <f32: 1.000000e+00> : tile<64xf32>
      %token = store_ptr_tko weak %ptr, %value, %mask : tile<64xptr<f32>>, tile<64xf32>, tile<64xi1> -> !cuda_tile.token
      return
    }

    // A contiguous flat slice whose start combines a row offset and a block
    // offset. Both terms are multiples of the vector width, so PtrToView can
    // factor them into one partition index.
    // CHECK-LABEL: entry @load_flat_row_slice
    // CHECK-GPU-LABEL: gpu.func @load_flat_row_slice
    entry @load_flat_row_slice(%base: tile<ptr<f32>>, %N: tile<i32>, %row: tile<i32>, %block: tile<i32>) {
      %D = constant <i32: 1024> : tile<i32>
      %BLOCK_SIZE = constant <i32: 32> : tile<i32>
      %lane = iota : tile<32xi32>
      %row_offset = muli %row, %D : tile<i32>
      %block_offset = muli %block, %BLOCK_SIZE : tile<i32>
      %start = addi %row_offset, %block_offset : tile<i32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<32xi32>
      %index = addi %start_bc, %lane : tile<32xi32>
      %N_1d = reshape %N : tile<i32> -> tile<1xi32>
      %N_bc = broadcast %N_1d : tile<1xi32> -> tile<32xi32>
      %mask = cmpi less_than %index, %N_bc, signed : tile<32xi32> -> tile<32xi1>
      %base_1d = reshape %base : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<32xptr<f32>>
      %ptr = offset %base_bc, %index : tile<32xptr<f32>>, tile<32xi32> -> tile<32xptr<f32>>
      %pad = constant <f32: 0.000000e+00> : tile<32xf32>
      // CHECK: %[[TV_ROW:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PV_ROW:.*]] = make_partition_view %[[TV_ROW]] : partition_view<tile=(32), padding_value = zero, tensor_view<?xf32, strides=[1]>>
      // CHECK: %{{.*}}, %{{.*}} = load_view_tko weak %[[PV_ROW]][%{{.*}}] : partition_view<tile=(32), padding_value = zero, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<32xf32>, token
      // CHECK-NOT: load_ptr_tko
      // CHECK-GPU: %[[ROW_VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%{{.*}}], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1], offset: ?>>
      // CHECK-GPU: vector.transfer_read %[[ROW_VIEW]][%{{.*}}], %{{.*}} : memref<?xf32, strided<[1], offset: ?>>, vector<32xf32>
      // CHECK-GPU-NOT: vector.gather
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<32xptr<f32>>, tile<32xi1>, tile<32xf32> -> tile<32xf32>, !cuda_tile.token
      return
    }

    // A flat row slice can add an element-space loop induction variable to a
    // row offset. The loop starts at zero and advances by the vector width, so
    // the induction variable is a valid tile index after division by 32.
    // CHECK-LABEL: entry @load_flat_row_loop_slice
    // CHECK-GPU-LABEL: gpu.func @load_flat_row_loop_slice
    entry @load_flat_row_loop_slice(%base: tile<ptr<f32>>, %N: tile<i32>) {
      %c0 = constant <i32: 0> : tile<i32>
      %c32 = constant <i32: 32> : tile<i32>
      %c1024 = constant <i32: 1024> : tile<i32>
      %lane = iota : tile<32xi32>
      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      %row_offset = muli %blockId_x, %c1024 : tile<i32>
      %N_1d = reshape %N : tile<i32> -> tile<1xi32>
      %N_bc = broadcast %N_1d : tile<1xi32> -> tile<32xi32>
      %base_1d = reshape %base : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<32xptr<f32>>
      %pad = constant <f32: 0.000000e+00> : tile<32xf32>
      // CHECK: %[[TV_ROW_LOOP:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PV_ROW_LOOP:.*]] = make_partition_view %[[TV_ROW_LOOP]] : partition_view<tile=(32), padding_value = zero, tensor_view<?xf32, strides=[1]>>
      // CHECK: %[[ROW_BLOCK_ID:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
      // CHECK: for %[[ROW_LOOP_IV:.*]] in
      // CHECK:   %[[ROW_TILE_INDEX:.*]] = muli %[[ROW_BLOCK_ID]], %{{.*}} : tile<i32>
      // CHECK:   %[[LOOP_TILE_INDEX:.*]] = divi %[[ROW_LOOP_IV]], %{{.*}} unsigned : tile<i32>
      // CHECK:   %[[PARTITION_INDEX:.*]] = addi %[[ROW_TILE_INDEX]], %[[LOOP_TILE_INDEX]] : tile<i32>
      // CHECK:   %{{.*}}, %{{.*}} = load_view_tko weak %[[PV_ROW_LOOP]][%[[PARTITION_INDEX]]]
      // CHECK-NOT: load_ptr_tko
      // CHECK-GPU: scf.for %[[ROW_LOOP_INDEX:.*]] =
      // CHECK-GPU:   %[[ROW_LOOP_INDEX_I32:.*]] = arith.index_cast %[[ROW_LOOP_INDEX]] : index to i32
      // CHECK-GPU:   arith.divui %[[ROW_LOOP_INDEX_I32]], %{{.*}} : i32
      // CHECK-GPU:   vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<?xf32, strided<[1], offset: ?>>, vector<32xf32>
      for %loopIdx in (%c0 to %c1024, step %c32) : tile<i32> {
        %start = addi %row_offset, %loopIdx : tile<i32>
        %start_1d = reshape %start : tile<i32> -> tile<1xi32>
        %start_bc = broadcast %start_1d : tile<1xi32> -> tile<32xi32>
        %index = addi %start_bc, %lane : tile<32xi32>
        %mask = cmpi less_than %index, %N_bc, signed : tile<32xi32> -> tile<32xi1>
        %ptr = offset %base_bc, %index : tile<32xptr<f32>>, tile<32xi32> -> tile<32xptr<f32>>
        %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<32xptr<f32>>, tile<32xi1>, tile<32xf32> -> tile<32xf32>, !cuda_tile.token
        continue
      }
      return
    }

    // A row-local mask can exclude a scalar row-base shift from its index.
    // PtrToView moves that shift into the scalar base and retains the local
    // loop index for a static-width row view.
    // CHECK-LABEL: entry @load_row_local_loop_slice
    // CHECK-GPU-LABEL: gpu.func @load_row_local_loop_slice
    entry @load_row_local_loop_slice(%base: tile<ptr<f32>>) {
      %c0 = constant <i32: 0> : tile<i32>
      %c32 = constant <i32: 32> : tile<i32>
      %c1024 = constant <i32: 1024> : tile<i32>
      %lane = iota : tile<32xi32>
      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      %row_offset = muli %blockId_x, %c1024 : tile<i32>
      %width_1d = reshape %c1024 : tile<i32> -> tile<1xi32>
      %width_bc = broadcast %width_1d : tile<1xi32> -> tile<32xi32>
      %base_1d = reshape %base : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<32xptr<f32>>
      %pad = constant <f32: 0.000000e+00> : tile<32xf32>
      // CHECK: %[[ROW_BLOCK:.*]], %{{.*}}, %{{.*}} = get_tile_block_id : tile<i32>
      // CHECK: %[[ROW_OFFSET:.*]] = muli %[[ROW_BLOCK]], %{{.*}} : tile<i32>
      // CHECK: for %[[ROW_LOOP:.*]] in
      // CHECK: %[[ROW_BASE:.*]] = offset %arg0, %[[ROW_OFFSET]] : tile<ptr<f32>>, tile<i32> -> tile<ptr<f32>>
      // CHECK: %[[ROW_TV:.*]] = make_tensor_view %[[ROW_BASE]], shape = [1024], strides = [1] : tensor_view<1024xf32, strides=[1]>
      // CHECK: %[[ROW_PV:.*]] = make_partition_view %[[ROW_TV]] : partition_view<tile=(32), padding_value = zero, tensor_view<1024xf32, strides=[1]>>
      // CHECK:   %[[ROW_TILE:.*]] = divi %[[ROW_LOOP]], %{{.*}} unsigned : tile<i32>
      // CHECK:   %{{.*}}, %{{.*}} = load_view_tko weak %[[ROW_PV]][%[[ROW_TILE]]]
      // CHECK-NOT: load_ptr_tko
      // CHECK-GPU: scf.for
      // CHECK-GPU:   vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} {in_bounds = [true]} : memref<1024xf32, strided<[1], offset: ?>>, vector<32xf32>
      for %loopIdx in (%c0 to %c1024, step %c32) : tile<i32> {
        %local_1d = reshape %loopIdx : tile<i32> -> tile<1xi32>
        %local_bc = broadcast %local_1d : tile<1xi32> -> tile<32xi32>
        %local_index = addi %local_bc, %lane : tile<32xi32>
        %mask = cmpi less_than %local_index, %width_bc, signed : tile<32xi32> -> tile<32xi1>
        %global_start = addi %row_offset, %loopIdx : tile<i32>
        %global_1d = reshape %global_start : tile<i32> -> tile<1xi32>
        %global_bc = broadcast %global_1d : tile<1xi32> -> tile<32xi32>
        %global_index = addi %global_bc, %lane : tile<32xi32>
        %ptr = offset %base_bc, %global_index : tile<32xptr<f32>>, tile<32xi32> -> tile<32xptr<f32>>
        %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<32xptr<f32>>, tile<32xi1>, tile<32xf32> -> tile<32xf32>, !cuda_tile.token
        continue
      }
      return
    }

    // -----------------------------------------------------------------------
    // 2-D non-affine pointer arithmetic (conv2d-style): PtrToView cannot lift
    // this, so it falls through to the gather/scatter lowering.
    // -----------------------------------------------------------------------
    // CHECK-LABEL: entry @maskedload_2d_conv
    // CHECK-NOT: make_tensor_view
    // CHECK: load_ptr_tko
    // CHECK-GPU-LABEL: gpu.func @maskedload_2d_conv
    // CHECK-GPU: vector.maskedload
    // CHECK-GPU: vector.maskedload
    // CHECK-GPU-NOT: vector.gather
    entry @maskedload_2d_conv(%base: tile<ptr<f16>>, %shift: tile<i32>, %stride: tile<i32>) {
      %zero = constant <i32: 0> : tile<i32>
      %start = subi %zero, %shift : tile<i32>
      %rows = iota : tile<2xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<2xi32>
      %row_index = addi %start_bc, %rows : tile<2xi32>
      %row_index_2d = reshape %row_index : tile<2xi32> -> tile<2x1xi32>
      %stride_2d = reshape %stride : tile<i32> -> tile<1x1xi32>
      %stride_bc = broadcast %stride_2d : tile<1x1xi32> -> tile<2x1xi32>
      %row_offset = muli %row_index_2d, %stride_bc : tile<2x1xi32>
      %row_offset_bc = broadcast %row_offset : tile<2x1xi32> -> tile<2x4xi32>
      %cols = iota : tile<4xi32>
      %cols_2d = reshape %cols : tile<4xi32> -> tile<1x4xi32>
      %cols_bc = broadcast %cols_2d : tile<1x4xi32> -> tile<2x4xi32>
      %offset = addi %row_offset_bc, %cols_bc : tile<2x4xi32>
      %base_2d = reshape %base : tile<ptr<f16>> -> tile<1x1xptr<f16>>
      %base_bc = broadcast %base_2d : tile<1x1xptr<f16>> -> tile<2x4xptr<f16>>
      %ptr = offset %base_bc, %offset : tile<2x4xptr<f16>>, tile<2x4xi32> -> tile<2x4xptr<f16>>

      %row_zero = constant <i32: 0> : tile<2x1xi32>
      %row_size = constant <i32: 2> : tile<2x1xi32>
      %row_ge = cmpi greater_than_or_equal %row_index_2d, %row_zero, signed : tile<2x1xi32> -> tile<2x1xi1>
      %row_lt = cmpi less_than %row_index_2d, %row_size, signed : tile<2x1xi32> -> tile<2x1xi1>
      %row_ge_bc = broadcast %row_ge : tile<2x1xi1> -> tile<2x4xi1>
      %row_lt_bc = broadcast %row_lt : tile<2x1xi1> -> tile<2x4xi1>
      %row_ge_i16 = exti %row_ge_bc signed : tile<2x4xi1> -> tile<2x4xi16>
      %row_lt_i16 = exti %row_lt_bc signed : tile<2x4xi1> -> tile<2x4xi16>
      %mask_i16 = andi %row_ge_i16, %row_lt_i16 : tile<2x4xi16>
      %mask = trunci %mask_i16 : tile<2x4xi16> -> tile<2x4xi1>
      %pad = constant <f16: 0.000000e+00> : tile<2x4xf16>
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<2x4xptr<f16>>, tile<2x4xi1>, tile<2x4xf16> -> tile<2x4xf16>, !cuda_tile.token
      return
    }

    // CHECK-LABEL: entry @maskedload_3d_conv
    // CHECK-NOT: make_tensor_view
    // CHECK: load_ptr_tko
    // CHECK-GPU-LABEL: gpu.func @maskedload_3d_conv
    // CHECK-GPU-COUNT-8: vector.maskedload
    // CHECK-GPU-NOT: vector.gather
    entry @maskedload_3d_conv(%base: tile<ptr<f32>>) {
      %cols = iota : tile<4xi32>
      %cols_3d = reshape %cols : tile<4xi32> -> tile<1x1x4xi32>
      %offset = broadcast %cols_3d : tile<1x1x4xi32> -> tile<2x4x4xi32>
      %base_3d = reshape %base : tile<ptr<f32>> -> tile<1x1x1xptr<f32>>
      %base_bc = broadcast %base_3d : tile<1x1x1xptr<f32>> -> tile<2x4x4xptr<f32>>
      %ptr = offset %base_bc, %offset : tile<2x4x4xptr<f32>>, tile<2x4x4xi32> -> tile<2x4x4xptr<f32>>
      %tile, %token = load_ptr_tko weak %ptr : tile<2x4x4xptr<f32>> -> tile<2x4x4xf32>, !cuda_tile.token
      return
    }

    // The column shift sits *inside* the broadcast (`broadcast(start + iota)`),
    // which is how a loop-invariant column base is normally emitted.
    // CHECK-LABEL: entry @maskedload_shifted_cols
    // CHECK-NOT: make_tensor_view
    // CHECK: load_ptr_tko
    // CHECK-GPU-LABEL: gpu.func @maskedload_shifted_cols
    // CHECK-GPU-COUNT-2: vector.maskedload
    // CHECK-GPU-NOT: vector.gather
    entry @maskedload_shifted_cols(%base: tile<ptr<f16>>, %shift: tile<i32>, %stride: tile<i32>, %col_start: tile<i32>) {
      %zero = constant <i32: 0> : tile<i32>
      %start = subi %zero, %shift : tile<i32>
      %rows = iota : tile<2xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<2xi32>
      %row_index = addi %start_bc, %rows : tile<2xi32>
      %row_index_2d = reshape %row_index : tile<2xi32> -> tile<2x1xi32>
      %stride_2d = reshape %stride : tile<i32> -> tile<1x1xi32>
      %stride_bc = broadcast %stride_2d : tile<1x1xi32> -> tile<2x1xi32>
      %row_offset = muli %row_index_2d, %stride_bc : tile<2x1xi32>
      %row_offset_bc = broadcast %row_offset : tile<2x1xi32> -> tile<2x4xi32>
      %cols = iota : tile<4xi32>
      %col_start_1d = reshape %col_start : tile<i32> -> tile<1xi32>
      %col_start_bc = broadcast %col_start_1d : tile<1xi32> -> tile<4xi32>
      %cols_shifted = addi %col_start_bc, %cols : tile<4xi32>
      %cols_2d = reshape %cols_shifted : tile<4xi32> -> tile<1x4xi32>
      %cols_bc = broadcast %cols_2d : tile<1x4xi32> -> tile<2x4xi32>
      %offset = addi %row_offset_bc, %cols_bc : tile<2x4xi32>
      %base_2d = reshape %base : tile<ptr<f16>> -> tile<1x1xptr<f16>>
      %base_bc = broadcast %base_2d : tile<1x1xptr<f16>> -> tile<2x4xptr<f16>>
      %ptr = offset %base_bc, %offset : tile<2x4xptr<f16>>, tile<2x4xi32> -> tile<2x4xptr<f16>>

      %row_zero = constant <i32: 0> : tile<2x1xi32>
      %row_size = constant <i32: 2> : tile<2x1xi32>
      %row_ge = cmpi greater_than_or_equal %row_index_2d, %row_zero, signed : tile<2x1xi32> -> tile<2x1xi1>
      %row_lt = cmpi less_than %row_index_2d, %row_size, signed : tile<2x1xi32> -> tile<2x1xi1>
      %row_ge_bc = broadcast %row_ge : tile<2x1xi1> -> tile<2x4xi1>
      %row_lt_bc = broadcast %row_lt : tile<2x1xi1> -> tile<2x4xi1>
      %row_ge_i16 = exti %row_ge_bc signed : tile<2x4xi1> -> tile<2x4xi16>
      %row_lt_i16 = exti %row_lt_bc signed : tile<2x4xi1> -> tile<2x4xi16>
      %mask_i16 = andi %row_ge_i16, %row_lt_i16 : tile<2x4xi16>
      %mask = trunci %mask_i16 : tile<2x4xi16> -> tile<2x4xi1>
      %pad = constant <f16: 0.000000e+00> : tile<2x4xf16>
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<2x4xptr<f16>>, tile<2x4xi1>, tile<2x4xf16> -> tile<2x4xf16>, !cuda_tile.token
      return
    }

    // Offsets that are constant along the minor dimension address one element
    // per row, not a contiguous run, so they must stay a gather.
    // CHECK-LABEL: entry @gather_minor_invariant
    // CHECK-NOT: make_tensor_view
    // CHECK: load_ptr_tko
    // CHECK-GPU-LABEL: gpu.func @gather_minor_invariant
    // CHECK-GPU: vector.gather
    // CHECK-GPU-NOT: vector.maskedload
    entry @gather_minor_invariant(%base: tile<ptr<f32>>) {
      %rows = iota : tile<4xi32>
      %rows_2d = reshape %rows : tile<4xi32> -> tile<4x1xi32>
      %offset = broadcast %rows_2d : tile<4x1xi32> -> tile<4x4xi32>
      %base_2d = reshape %base : tile<ptr<f32>> -> tile<1x1xptr<f32>>
      %base_bc = broadcast %base_2d : tile<1x1xptr<f32>> -> tile<4x4xptr<f32>>
      %ptr = offset %base_bc, %offset : tile<4x4xptr<f32>>, tile<4x4xi32> -> tile<4x4xptr<f32>>
      %tile, %token = load_ptr_tko weak %ptr : tile<4x4xptr<f32>> -> tile<4x4xf32>, !cuda_tile.token
      return
    }

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
      // CHECK-GPU-DAG: %[[GBASE:.*]] = memref.cast %[[SBASE]] : memref<*xbf16> to memref<?xbf16, strided<[1], offset: ?>>
      // vector.gather only lowers to LLVM for rank-1 vectors, so the multi-dim
      // index/mask/passthrough are flattened to rank-1 with vector.shape_cast,
      // a single rank-1 gather is emitted, and the result is shape_cast back.
      // CHECK-GPU: %[[GIDX:.*]] = vector.shape_cast %{{.*}} : vector<32x32xindex> to vector<1024xindex>
      // CHECK-GPU: %[[GMASK1D:.*]] = vector.shape_cast %{{.*}} : vector<32x32xi1> to vector<1024xi1>
      // CHECK-GPU: %[[GPAD:.*]] = vector.shape_cast %{{.*}} : vector<32x32xbf16> to vector<1024xbf16>
      // CHECK-GPU: %[[GATHERED1D:.*]] = vector.gather %[[GBASE]][%{{.*}}] [%[[GIDX]]], %[[GMASK1D]], %[[GPAD]] : memref<?xbf16, strided<[1], offset: ?>>, vector<1024xindex>, vector<1024xi1>, vector<1024xbf16> into vector<1024xbf16>
      // CHECK-GPU: %[[GATHERED:.*]] = vector.shape_cast %[[GATHERED1D]] : vector<1024xbf16> to vector<32x32xbf16>
      %v, %t = load_ptr_tko weak %ptr, %mask, %pad : tile<32x32xptr<bf16>>, tile<32x32xi1>, tile<32x32xbf16> -> tile<32x32xbf16>, !cuda_tile.token
      // Store (scatter)
      // CHECK: store_ptr_tko weak %[[OFF]], %[[GV]], %[[GMASK]] : tile<32x32xptr<bf16>>, tile<32x32xbf16>, tile<32x32xi1> -> token
      // The value is likewise flattened to rank-1 before the rank-1 scatter.
      // CHECK-GPU: %[[SVAL:.*]] = vector.shape_cast %[[GATHERED]] : vector<32x32xbf16> to vector<1024xbf16>
      // CHECK-GPU: vector.scatter %{{.*}}[%{{.*}}] [%{{.*}}], %{{.*}}, %[[SVAL]] : memref<?xbf16, strided<[1], offset: ?>>, vector<1024xindex>, vector<1024xi1>, vector<1024xbf16>
      %57 = store_ptr_tko weak %ptr, %v, %mask : tile<32x32xptr<bf16>>, tile<32x32xbf16>, tile<32x32xi1> -> !cuda_tile.token
      return
    }

    // -----------------------------------------------------------------------
    // Corner case: 2-D gather/scatter with NO mask (and no padding value). The
    // helper materializes an all-true mask, which is flattened to rank-1 along
    // with the index and value so the gather/scatter still lower cleanly. The
    // distinct 16x16 -> 256 flatten also guards the row-major collapse.
    // -----------------------------------------------------------------------
    // CHECK-LABEL: entry @gather_2d_nomask
    // CHECK-NOT: make_tensor_view
    // CHECK-GPU-LABEL: gpu.func @gather_2d_nomask
    // CHECK-GPU-SAME: (%[[NMBASE:.*]]: memref<*xf32>,
    entry @gather_2d_nomask(%base: tile<ptr<f32>>, %stride0: tile<i32>, %stride1: tile<i32>, %M: tile<i32>) {
      %iota = iota : tile<16xi32>
      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      %c16 = constant <i32: 16> : tile<i32>
      %start = muli %blockId_x, %c16 : tile<i32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<16xi32>
      %linear = addi %start_bc, %iota : tile<16xi32>
      // row = linear / M, col = linear % M (non-affine — not liftable to a view)
      %M_1d = reshape %M : tile<i32> -> tile<1xi32>
      %M_bc = broadcast %M_1d : tile<1xi32> -> tile<16xi32>
      %row = divi %linear, %M_bc signed : tile<16xi32>
      %col = remi %linear, %M_bc signed : tile<16xi32>
      %row_2d = reshape %row : tile<16xi32> -> tile<16x1xi32>
      %row_bc = broadcast %row_2d : tile<16x1xi32> -> tile<16x16xi32>
      %stride0_rs = reshape %stride0 : tile<i32> -> tile<1x1xi32>
      %stride0_bc = broadcast %stride0_rs : tile<1x1xi32> -> tile<16x16xi32>
      %off0 = muli %row_bc, %stride0_bc : tile<16x16xi32>
      %col_2d = reshape %col : tile<16xi32> -> tile<1x16xi32>
      %col_bc = broadcast %col_2d : tile<1x16xi32> -> tile<16x16xi32>
      %stride1_rs = reshape %stride1 : tile<i32> -> tile<1x1xi32>
      %stride1_bc = broadcast %stride1_rs : tile<1x1xi32> -> tile<16x16xi32>
      %off1 = muli %col_bc, %stride1_bc : tile<16x16xi32>
      %total_off = addi %off0, %off1 : tile<16x16xi32>
      %base_rs = reshape %base : tile<ptr<f32>> -> tile<1x1xptr<f32>>
      %base_bc = broadcast %base_rs : tile<1x1xptr<f32>> -> tile<16x16xptr<f32>>
      %ptr = offset %base_bc, %total_off : tile<16x16xptr<f32>>, tile<16x16xi32> -> tile<16x16xptr<f32>>
      // Load (gather) — no mask operand.
      // CHECK: %[[NMOFF:.*]] = offset %{{.*}}, %{{.*}} : tile<16x16xptr<f32>>, tile<16x16xi32> -> tile<16x16xptr<f32>>
      // CHECK: %[[NMV:.*]], %{{.*}} = load_ptr_tko weak %[[NMOFF]] : tile<16x16xptr<f32>> -> tile<16x16xf32>, token
      // CHECK-GPU-DAG: %[[NMB:.*]] = memref.cast %[[NMBASE]] : memref<*xf32> to memref<?xf32, strided<[1], offset: ?>>
      // With no original mask, an all-true mask is materialized then flattened.
      // CHECK-GPU: %[[NMTRUE:.*]] = arith.constant true
      // CHECK-GPU: %[[NMMASK2D:.*]] = vector.broadcast %[[NMTRUE]] : i1 to vector<16x16xi1>
      // CHECK-GPU: %[[NMIDX:.*]] = vector.shape_cast %{{.*}} : vector<16x16xindex> to vector<256xindex>
      // CHECK-GPU: %[[NMMASK1D:.*]] = vector.shape_cast %[[NMMASK2D]] : vector<16x16xi1> to vector<256xi1>
      // CHECK-GPU: %[[NMPAD:.*]] = vector.shape_cast %{{.*}} : vector<16x16xf32> to vector<256xf32>
      // CHECK-GPU: %[[NMGATH1D:.*]] = vector.gather %[[NMB]][%{{.*}}] [%[[NMIDX]]], %[[NMMASK1D]], %[[NMPAD]] : memref<?xf32, strided<[1], offset: ?>>, vector<256xindex>, vector<256xi1>, vector<256xf32> into vector<256xf32>
      // CHECK-GPU: %[[NMGATH:.*]] = vector.shape_cast %[[NMGATH1D]] : vector<256xf32> to vector<16x16xf32>
      %v, %t = load_ptr_tko weak %ptr : tile<16x16xptr<f32>> -> tile<16x16xf32>, !cuda_tile.token
      // Store (scatter) — no mask operand.
      // CHECK: store_ptr_tko weak %[[NMOFF]], %[[NMV]] : tile<16x16xptr<f32>>, tile<16x16xf32> -> token
      // CHECK-GPU: %[[NMSVAL:.*]] = vector.shape_cast %[[NMGATH]] : vector<16x16xf32> to vector<256xf32>
      // CHECK-GPU: vector.scatter %{{.*}}[%{{.*}}] [%{{.*}}], %{{.*}}, %[[NMSVAL]] : memref<?xf32, strided<[1], offset: ?>>, vector<256xindex>, vector<256xi1>, vector<256xf32>
      %st = store_ptr_tko weak %ptr, %v : tile<16x16xptr<f32>>, tile<16x16xf32> -> !cuda_tile.token
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
    // The load is already rank-1, so the flatten guard is skipped: no shape_cast.
    // CHECK-GPU:   vector.gather
    // CHECK-GPU-NOT:   vector.shape_cast
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
        %v, %t = load_ptr_tko weak %ptr, %mask, %neg_inf : tile<32xptr<f32>>, tile<32xi1>, tile<32xf32> -> tile<32xf32>, !cuda_tile.token
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
      %st = store_ptr_tko weak %sptr, %for#1, %mask : tile<32xptr<i32>>, tile<32xi32>, tile<32xi1> -> !cuda_tile.token
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
        %v, %t = load_ptr_tko weak %iterPtr, %mask, %pad : tile<64x16xptr<f32>>, tile<64x16xi1>, tile<64x16xf32> -> tile<64x16xf32>, !cuda_tile.token
        %next = offset %iterPtr, %c16_2d : tile<64x16xptr<f32>>, tile<64x16xi32> -> tile<64x16xptr<f32>>
        continue %next : tile<64x16xptr<f32>>
      }
      return
    }

    // Regression test for dead-iter-arg elimination: the loop's only
    // iter_arg has an externally-unused result but feeds a store every
    // iteration.  It must be kept (and the store preserved) -- dropping it
    // would delete the store and leave an empty kernel.
    // CHECK-LABEL: entry @iter_arg_feeds_store
    // CHECK-GPU-LABEL: gpu.func @iter_arg_feeds_store
    entry @iter_arg_feeds_store(%out: tile<ptr<f32>>, %n: tile<i32>) {
      %c0 = constant <i32: 0> : tile<i32>
      %c32 = constant <i32: 32> : tile<i32>
      %f0 = constant <f32: 0.000000e+00> : tile<f32>
      %f1 = constant <f32: 1.000000e+00> : tile<f32>
      %iota = iota : tile<32xi32>
      %n_1d = reshape %n : tile<i32> -> tile<1xi32>
      %n_bc = broadcast %n_1d : tile<1xi32> -> tile<32xi32>
      %mask = cmpi less_than %iota, %n_bc, signed : tile<32xi32> -> tile<32xi1>
      %out_1d = reshape %out : tile<ptr<f32>> -> tile<1xptr<f32>>
      %out_bc = broadcast %out_1d : tile<1xptr<f32>> -> tile<32xptr<f32>>
      %ptr = offset %out_bc, %iota : tile<32xptr<f32>>, tile<32xi32> -> tile<32xptr<f32>>

      // CHECK: for %{{.*}} in {{.*}} iter_values(%{{.*}} = %{{.*}}) -> (tile<f32>)
      // CHECK: store_view_tko
      // CHECK-NOT: store_ptr_tko
      // CHECK-GPU: scf.for
      // CHECK-GPU: vector.transfer_write
      %for = for %i in (%c0 to %n, step %c32) : tile<i32>
          iter_values(%acc = %f0) -> (tile<f32>) {
        %acc_1d = reshape %acc : tile<f32> -> tile<1xf32>
        %val = broadcast %acc_1d : tile<1xf32> -> tile<32xf32>
        %t = store_ptr_tko weak %ptr, %val, %mask : tile<32xptr<f32>>, tile<32xf32>, tile<32xi1> -> !cuda_tile.token
        %nacc = addf %acc, %f1 : tile<f32>
        continue %nacc : tile<f32>
      }
      return
    }

    // -----------------------------------------------------------------------
    // cutile widens index arithmetic to i64 with `exti signed` before applying
    // it to the pointer, and spells the padding tile as a broadcast of a scalar
    // integer constant.  Signed widening is value-preserving, so the narrow
    // (i32) scalars are recovered and fed to the view ops unchanged; the
    // integer zero padding maps onto `padding_value = zero`.
    // -----------------------------------------------------------------------
    // CHECK-LABEL: entry @load_1d_widened_index
    // CHECK-GPU-LABEL: gpu.func @load_1d_widened_index
    entry @load_1d_widened_index(%arg0: tile<ptr<i32>>, %arg1: tile<i32>) {
      %cst_0 = constant <i32: 0> : tile<i32>
      %cst_128 = constant <i32: 128> : tile<i32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_128 : tile<i32>
      %lane = iota : tile<128xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<128xi32>
      %index = addi %start_bc, %lane : tile<128xi32>
      %index_i64 = exti %index signed : tile<128xi32> -> tile<128xi64>
      %shape_i64 = exti %arg1 signed : tile<i32> -> tile<i64>
      %shape_1d = reshape %shape_i64 : tile<i64> -> tile<1xi64>
      %shape_bc = broadcast %shape_1d : tile<1xi64> -> tile<128xi64>
      %mask = cmpi less_than %index_i64, %shape_bc, unsigned : tile<128xi64> -> tile<128xi1>
      %base_1d = reshape %arg0 : tile<ptr<i32>> -> tile<1xptr<i32>>
      %base_bc = broadcast %base_1d : tile<1xptr<i32>> -> tile<128xptr<i32>>
      %ptr = offset %base_bc, %index_i64 : tile<128xptr<i32>>, tile<128xi64> -> tile<128xptr<i32>>
      %pad_1d = reshape %cst_0 : tile<i32> -> tile<1xi32>
      %pad = broadcast %pad_1d : tile<1xi32> -> tile<128xi32>
      // CHECK: %[[TVW:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xi32, strides=[1]>
      // CHECK: %[[PVW:.*]] = make_partition_view %[[TVW]] : partition_view<tile=(128), padding_value = zero, tensor_view<?xi32, strides=[1]>>
      // CHECK: load_view_tko weak %[[PVW]][%{{.*}}]
      // CHECK-NOT: load_ptr_tko
      // CHECK-GPU: %[[WVIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%{{.*}}], strides: [1] : memref<*xi32> to memref<?xi32, strided<[1], offset: ?>>
      // CHECK-GPU: vector.transfer_read %[[WVIEW]][%{{.*}}], %{{.*}} : memref<?xi32, strided<[1], offset: ?>>, vector<128xi32>
      // CHECK-GPU-NOT: vector.gather
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<128xptr<i32>>, tile<128xi1>, tile<128xi32> -> tile<128xi32>, !cuda_tile.token
      return
    }

    // Same canonical access, but widened with `exti unsigned` and compared
    // unsigned.  Zero- and sign-extension agree on the non-negative,
    // non-wrapping index arithmetic this pattern already presupposes, so the
    // access lifts exactly like the signed-widened one above.
    // CHECK-LABEL: entry @load_1d_widened_index_unsigned
    // CHECK-GPU-LABEL: gpu.func @load_1d_widened_index_unsigned
    entry @load_1d_widened_index_unsigned(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_128 = constant <i32: 128> : tile<i32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_128 : tile<i32>
      %lane = iota : tile<128xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<128xi32>
      %index = addi %start_bc, %lane : tile<128xi32>
      %index_i64 = exti %index unsigned : tile<128xi32> -> tile<128xi64>
      %shape_i64 = exti %arg1 unsigned : tile<i32> -> tile<i64>
      %shape_1d = reshape %shape_i64 : tile<i64> -> tile<1xi64>
      %shape_bc = broadcast %shape_1d : tile<1xi64> -> tile<128xi64>
      %mask = cmpi less_than %index_i64, %shape_bc, unsigned : tile<128xi64> -> tile<128xi1>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<128xptr<f32>>
      %ptr = offset %base_bc, %index_i64 : tile<128xptr<f32>>, tile<128xi64> -> tile<128xptr<f32>>
      // CHECK: %[[TVU:.*]] = make_tensor_view %arg0, shape = [%arg1], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
      // CHECK: %[[PVU:.*]] = make_partition_view %[[TVU]] : partition_view<tile=(128), padding_value = zero, tensor_view<?xf32, strides=[1]>>
      // CHECK: load_view_tko weak %[[PVU]][%{{.*}}]
      // CHECK-NOT: load_ptr_tko
      // CHECK-GPU: vector.transfer_read
      // CHECK-GPU-NOT: vector.gather
      %tile, %token = load_ptr_tko weak %ptr, %mask : tile<128xptr<f32>>, tile<128xi1> -> tile<128xf32>, !cuda_tile.token
      return
    }

    // CHECK-LABEL: entry @load_static_splat_bounds
    // CHECK-GPU-LABEL: gpu.func @load_static_splat_bounds
    entry @load_static_splat_bounds(%arg0: tile<ptr<f16>>) {
      %rows = iota : tile<4xi32>
      %cols = iota : tile<8xi32>
      %rows_i64 = exti %rows signed : tile<4xi32> -> tile<4xi64>
      %cols_i64 = exti %cols signed : tile<8xi32> -> tile<8xi64>
      %rows_2d = reshape %rows_i64 : tile<4xi64> -> tile<4x1xi64>
      %cols_2d = reshape %cols_i64 : tile<8xi64> -> tile<1x8xi64>
      %stride = constant <i64: 8> : tile<4x1xi64>
      %row_offset = muli %rows_2d, %stride : tile<4x1xi64>
      %row_offset_bc = broadcast %row_offset : tile<4x1xi64> -> tile<4x8xi64>
      %cols_bc = broadcast %cols_2d : tile<1x8xi64> -> tile<4x8xi64>
      %offset = addi %row_offset_bc, %cols_bc : tile<4x8xi64>
      %base_2d = reshape %arg0 : tile<ptr<f16>> -> tile<1x1xptr<f16>>
      %base_bc = broadcast %base_2d : tile<1x1xptr<f16>> -> tile<4x8xptr<f16>>
      %ptr = offset %base_bc, %offset : tile<4x8xptr<f16>>, tile<4x8xi64> -> tile<4x8xptr<f16>>

      %row_zero = constant <i64: 0> : tile<4x1xi64>
      %row_size = constant <i64: 4> : tile<4x1xi64>
      %row_ge = cmpi greater_than_or_equal %rows_2d, %row_zero, signed : tile<4x1xi64> -> tile<4x1xi1>
      %row_lt = cmpi less_than %rows_2d, %row_size, signed : tile<4x1xi64> -> tile<4x1xi1>
      %col_zero = constant <i64: 0> : tile<1x8xi64>
      %col_size = constant <i64: 8> : tile<1x8xi64>
      %col_ge = cmpi greater_than_or_equal %cols_2d, %col_zero, signed : tile<1x8xi64> -> tile<1x8xi1>
      %col_lt = cmpi less_than %cols_2d, %col_size, signed : tile<1x8xi64> -> tile<1x8xi1>
      %row_ge_bc = broadcast %row_ge : tile<4x1xi1> -> tile<4x8xi1>
      %row_lt_bc = broadcast %row_lt : tile<4x1xi1> -> tile<4x8xi1>
      %col_ge_bc = broadcast %col_ge : tile<1x8xi1> -> tile<4x8xi1>
      %col_lt_bc = broadcast %col_lt : tile<1x8xi1> -> tile<4x8xi1>
      %row_ge_i16 = exti %row_ge_bc signed : tile<4x8xi1> -> tile<4x8xi16>
      %row_lt_i16 = exti %row_lt_bc signed : tile<4x8xi1> -> tile<4x8xi16>
      %col_ge_i16 = exti %col_ge_bc signed : tile<4x8xi1> -> tile<4x8xi16>
      %col_lt_i16 = exti %col_lt_bc signed : tile<4x8xi1> -> tile<4x8xi16>
      %row_mask_i16 = andi %row_ge_i16, %row_lt_i16 : tile<4x8xi16>
      %col_mask_i16 = andi %col_ge_i16, %col_lt_i16 : tile<4x8xi16>
      %mask_i16 = andi %row_mask_i16, %col_mask_i16 : tile<4x8xi16>
      %mask = trunci %mask_i16 : tile<4x8xi16> -> tile<4x8xi1>
      %pad = constant <f16: 0.000000e+00> : tile<4x8xf16>

      // CHECK: make_tensor_view %arg0, shape = [4, 8], strides = [8, 1] : tensor_view<4x8xf16, strides=[8,1]>
      // CHECK: load_view_tko
      // CHECK-NOT: load_ptr_tko
      // CHECK-GPU: vector.transfer_read
      // CHECK-GPU-NOT: vector.gather
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<4x8xptr<f16>>, tile<4x8xi1>, tile<4x8xf16> -> tile<4x8xf16>, !cuda_tile.token
      return
    }
  }
}
