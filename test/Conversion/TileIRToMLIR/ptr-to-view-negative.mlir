// RUN: tileir-to-mlir --tileir-ptr-to-view --verify-diagnostics %s | FileCheck %s

module {
  cuda_tile.module @cuda_tile_module {

    // CHECK-LABEL: entry @no_mask
    entry @no_mask(%arg0: tile<ptr<f32>>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[NM:.*]], %[[NMT:.*]] = load_ptr_tko weak %[[NMP:.*]] : tile<1024xptr<f32>> -> tile<1024xf32>, token
      // expected-remark @below {{tileir-ptr-to-view: load has no mask; skipping}}
      %tile, %token = load_ptr_tko weak %ptr : tile<1024xptr<f32>> -> tile<1024xf32>, !cuda_tile.token
      return
    }

    // CHECK-LABEL: entry @bad_padding
    entry @bad_padding(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %pad = constant <f32: 1.000000e+00> : tile<1024xf32>
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
      // CHECK: %[[BP:.*]], %[[BPT:.*]] = load_ptr_tko weak %[[BPP:.*]], %[[BPM:.*]], %{{.*}} : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, token
      // expected-remark @below {{tileir-ptr-to-view: padding value not recognised}}
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, !cuda_tile.token
      return
    }

    // CHECK-LABEL: entry @non_canonical_addend
    entry @non_canonical_addend(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %cst_5 = constant <i32: 5> : tile<1024xi32>
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
      %ptr = offset %base_bc, %cst_5 : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[NC:.*]], %[[NCT:.*]] = load_ptr_tko weak %[[NCP:.*]], %[[NCM:.*]] : tile<1024xptr<f32>>, tile<1024xi1> -> tile<1024xf32>, token
      // expected-remark @below {{tileir-ptr-to-view: pointer-arithmetic pattern not recognised; skipping}}
      %tile, %token = load_ptr_tko weak %ptr, %mask : tile<1024xptr<f32>>, tile<1024xi1> -> tile<1024xf32>, !cuda_tile.token
      return
    }

    // CHECK-LABEL: entry @store_no_mask
    entry @store_no_mask(%arg0: tile<ptr<f32>>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %value = constant <f32: 1.000000e+00> : tile<1024xf32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[SNM:.*]] = store_ptr_tko weak %[[SNMP:.*]], %{{.*}} : tile<1024xptr<f32>>, tile<1024xf32> -> token
      // expected-remark @below {{tileir-ptr-to-view: store has no mask; skipping}}
      %token = store_ptr_tko weak %ptr, %value : tile<1024xptr<f32>>, tile<1024xf32> -> !cuda_tile.token
      return
    }

    // MoE-style row gather: the dim-0 index is *loaded* from memory rather than
    // being `start + iota`, so the 2-D access cannot be modelled by a
    // partition view and must stay a gather.  The 1-D load of the row ids is
    // liftable and lifts.
    // CHECK-LABEL: entry @data_dependent_row
    entry @data_dependent_row(%ids: tile<ptr<i32>>, %base: tile<ptr<bf16>>, %n: tile<i32>, %m: tile<i32>, %stride: tile<i32>) {
      %cst_32 = constant <i32: 32> : tile<i32>
      %pad = constant <bf16: 0.000000e+00> : tile<32x32xbf16>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_32 : tile<i32>
      %lane = iota : tile<32xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<32xi32>
      %index = addi %start_bc, %lane : tile<32xi32>
      %n_1d = reshape %n : tile<i32> -> tile<1xi32>
      %n_bc = broadcast %n_1d : tile<1xi32> -> tile<32xi32>
      %id_mask = cmpi less_than %index, %n_bc, signed : tile<32xi32> -> tile<32xi1>
      %ids_1d = reshape %ids : tile<ptr<i32>> -> tile<1xptr<i32>>
      %ids_bc = broadcast %ids_1d : tile<1xptr<i32>> -> tile<32xptr<i32>>
      %ids_ptr = offset %ids_bc, %index : tile<32xptr<i32>>, tile<32xi32> -> tile<32xptr<i32>>
      // CHECK: make_tensor_view %arg0
      // CHECK: load_view_tko
      %rows, %id_token = load_ptr_tko weak %ids_ptr, %id_mask : tile<32xptr<i32>>, tile<32xi1> -> tile<32xi32>, !cuda_tile.token

      %rows_2d = reshape %rows : tile<32xi32> -> tile<32x1xi32>
      %stride_2d = reshape %stride : tile<i32> -> tile<1x1xi32>
      %stride_bc = broadcast %stride_2d : tile<1x1xi32> -> tile<32x1xi32>
      %row_off = muli %rows_2d, %stride_bc : tile<32x1xi32>
      %row_off_bc = broadcast %row_off : tile<32x1xi32> -> tile<32x32xi32>
      %cols = iota : tile<32xi32>
      %cols_2d = reshape %cols : tile<32xi32> -> tile<1x32xi32>
      %cols_bc = broadcast %cols_2d : tile<1x32xi32> -> tile<32x32xi32>
      %off = addi %row_off_bc, %cols_bc : tile<32x32xi32>
      %base_2d = reshape %base : tile<ptr<bf16>> -> tile<1x1xptr<bf16>>
      %base_bc = broadcast %base_2d : tile<1x1xptr<bf16>> -> tile<32x32xptr<bf16>>
      %ptr = offset %base_bc, %off : tile<32x32xptr<bf16>>, tile<32x32xi32> -> tile<32x32xptr<bf16>>
      %m_2d = reshape %m : tile<i32> -> tile<1x1xi32>
      %m_bc = broadcast %m_2d : tile<1x1xi32> -> tile<32x32xi32>
      %rows_bc = broadcast %rows_2d : tile<32x1xi32> -> tile<32x32xi32>
      %row_mask = cmpi less_than %rows_bc, %m_bc, signed : tile<32x32xi32> -> tile<32x32xi1>
      // CHECK: load_ptr_tko
      // expected-remark @below {{tileir-ptr-to-view: pointer-arithmetic pattern not recognised; skipping}}
      %tile, %token = load_ptr_tko weak %ptr, %row_mask, %pad : tile<32x32xptr<bf16>>, tile<32x32xi1>, tile<32x32xbf16> -> tile<32x32xbf16>, !cuda_tile.token
      return
    }
  }
}