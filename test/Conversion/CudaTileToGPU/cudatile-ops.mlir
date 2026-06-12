// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | FileCheck %s
// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | mlir-opt --loop-invariant-code-motion -canonicalize -cse > /dev/null

// Tests derived from cuda_tile IR op definition examples in Ops.td.
// Each entry exercises one or more supported ops.

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: gpu.module @ops_module {
cuda_tile.module @ops_module {

  // --- global / get_global ---
  // Derived from cuda_tile.global and cuda_tile.get_global mlirExamples in
  // Ops.td.
  // CHECK: memref.global @val : memref<4xf32> = dense<[1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01]> {alignment = 128 : i64}
  global @val alignment = 128 <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>

  // CHECK-LABEL: gpu.func @test_get_global
  entry @test_get_global() {
    // CHECK: %[[G:.*]] = memref.get_global @val : memref<4xf32>
    // CHECK: %[[P:.*]] = memref.cast %[[G]] : memref<4xf32> to memref<*xf32>
    %ptr = get_global @val : tile<ptr<f32>>
    return
  }

  // --- get_tensor_shape ---
  // Derived from cuda_tile.get_tensor_shape mlirExamples in Ops.td.
  // CHECK-LABEL: gpu.func @test_get_tensor_shape_example
  // CHECK-SAME: %[[TS_BASE:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_get_tensor_shape_example(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // CHECK: %[[TS_VIEW:.*]] = memref.reinterpret_cast %[[TS_BASE]] to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<*xf32> to memref<32x32xf32>
    %tensor_view = make_tensor_view %base, shape = [32, 32], strides = [32, 1] : tensor_view<32x32xf32, strides=[32,1]>
    // CHECK: arith.index_castui {{.*}} : index to i64
    // CHECK: arith.index_castui {{.*}} : index to i64
    %dim0, %dim1 = get_tensor_shape %tensor_view : tensor_view<32x32xf32, strides=[32,1]> -> tile<i64>
    return
  }

  // --- iota ---
  // Extracted from Ops.td mlirExamples usage (e.g. atomic examples using
  // `%offsets = iota : tile<8xi32>`).
  // CHECK-LABEL: gpu.func @test_iota_example
  entry @test_iota_example() {
    // CHECK: %[[STEP:.*]] = vector.step : vector<8xindex>
    // CHECK: %[[IOTA:.*]] = arith.index_castui %[[STEP]] : vector<8xindex> to vector<8xi32>
    %offsets = iota : tile<8xi32>
    return
  }

  // --- constant ---
  // CHECK-LABEL: gpu.func @test_constant
  entry @test_constant() {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    %c0 = constant <i32: 0> : tile<i32>
    // CHECK: %[[C1:.*]] = arith.constant 1 : i64
    %c1 = constant <i64: 1> : tile<i64>
    // CHECK: %[[C2:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %c2 = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
    // CHECK: %[[C3:.*]] = arith.constant dense<0.000000e+00> : vector<2x4xf32>
    %c3 = constant <f32: 0.0> : tile<2x4xf32>
    // CHECK: %[[C4:.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : vector<4xf64>
    %c4 = constant <f64: [0.0, 1.0, 2.0, 3.0]> : tile<4xf64>
    return
  }

  // --- alloca ---
  // Derived from cuda_tile.alloca mlirExample in Ops.td. The `global` variant
  // from the example has no equivalent in the unranked memref pointer model and
  // is exercised separately as a negative test in cudatile-alloca-negative.mlir.
  // CHECK-LABEL: gpu.func @test_alloca
  entry @test_alloca() {
    // CHECK: %[[A:.*]] = memref.alloca() {alignment = 128 : i64} : memref<64xf32>
    // CHECK: %[[P:.*]] = memref.cast %[[A]] : memref<64xf32> to memref<*xf32>
    %0 = alloca num_elem = 64, alignment = 128 : tile<ptr<f32>>
    return
  }

  // --- atan2 ---
  // CHECK-LABEL: gpu.func @test_atan2
  entry @test_atan2() {
    // CHECK: %[[X:.*]] = arith.constant dense<[1.000000e+00, -1.000000e+00, 0.000000e+00, 2.000000e+00]> : vector<4xf32>
    %x = constant <f32: [1.0, -1.0, 0.0, 2.0]> : tile<4xf32>
    // CHECK: %[[Y:.*]] = arith.constant dense<[1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00]> : vector<4xf32>
    %y = constant <f32: [1.0, 1.0, 1.0, 0.0]> : tile<4xf32>
    // CHECK: %[[RES:.*]] = math.atan2 %[[X]], %[[Y]] : vector<4xf32>
    %res = atan2 %x, %y : tile<4xf32>
    return
  }

  // --- ceil ---
  // CHECK-LABEL: gpu.func @test_ceil
  entry @test_ceil() {
    // CHECK: %[[SRC:.*]] = arith.constant 5.000000e-01 : f32
    %source = constant <f32: 0.5> : tile<f32>
    // CHECK: %[[RES:.*]] = math.ceil %[[SRC]] : f32
    %result = ceil %source : tile<f32>
    return
  }

  // --- cmpf ---
  // CHECK-LABEL: gpu.func @test_cmpf
  entry @test_cmpf() {
    // CHECK: %[[LHS0:.*]] = arith.constant 0.000000e+00 : f16
    %lhs0 = constant <f16: 0.0> : tile<f16>
    // CHECK: %[[RHS0:.*]] = arith.constant 0.000000e+00 : f16
    %rhs0 = constant <f16: 0.0> : tile<f16>
    // CHECK: %[[CMP0:.*]] = arith.cmpf oeq, %[[LHS0]], %[[RHS0]] : f16
    %x0 = cmpf equal ordered %lhs0, %rhs0 : tile<f16> -> tile<i1>

    // CHECK: %[[LHS1:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf16>
    %lhs1 = constant <f16: 0.0> : tile<2x2xf16>
    // CHECK: %[[RHS1:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf16>
    %rhs1 = constant <f16: 0.0> : tile<2x2xf16>
    // CHECK: %[[CMP1:.*]] = arith.cmpf ult, %[[LHS1]], %[[RHS1]] : vector<2x2xf16>
    %x2 = cmpf less_than unordered %lhs1, %rhs1 : tile<2x2xf16> -> tile<2x2xi1>
    return
  }

  // --- cmpi ---
  // CHECK-LABEL: gpu.func @test_cmpi
  entry @test_cmpi() {
    // CHECK: %[[ILHS0:.*]] = arith.constant 0 : i16
    %lhs0 = constant <i16: 0> : tile<i16>
    // CHECK: %[[IRHS0:.*]] = arith.constant 0 : i16
    %rhs0 = constant <i16: 0> : tile<i16>
    // CHECK: %[[ICMP0:.*]] = arith.cmpi slt, %[[ILHS0]], %[[IRHS0]] : i16
    %x0 = cmpi less_than %lhs0, %rhs0, signed : tile<i16> -> tile<i1>

    // CHECK: %[[ILHS1:.*]] = arith.constant dense<0> : vector<2x2xi64>
    %lhs1 = constant <i64: 0> : tile<2x2xi64>
    // CHECK: %[[IRHS1:.*]] = arith.constant dense<0> : vector<2x2xi64>
    %rhs1 = constant <i64: 0> : tile<2x2xi64>
    // CHECK: %[[ICMP1:.*]] = arith.cmpi eq, %[[ILHS1]], %[[IRHS1]] : vector<2x2xi64>
    %x1 = cmpi equal %lhs1, %rhs1, signed : tile<2x2xi64> -> tile<2x2xi1>
    return
  }

  // --- cos ---
  // CHECK-LABEL: gpu.func @test_cos
  entry @test_cos() {
    // CHECK: %[[COS_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: %[[COS_RES:.*]] = math.cos %[[COS_IN]] : vector<4xf32>
    %res = cos %in : tile<4xf32>
    return
  }

  // --- exp ---
  // CHECK-LABEL: gpu.func @test_exp
  entry @test_exp() {
    // CHECK: %[[EXP_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: %[[EXP_RES:.*]] = math.exp %[[EXP_IN]] : vector<4xf32>
    %res = exp %in : tile<4xf32>
    return
  }

  // --- exp2 ---
  // CHECK-LABEL: gpu.func @test_exp2
  entry @test_exp2() {
    // CHECK: %[[EXP2_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: %[[EXP2_RES:.*]] = math.exp2 %[[EXP2_IN]] : vector<4xf32>
    %res = exp2 %in : tile<4xf32>
    return
  }

  // --- floor ---
  // CHECK-LABEL: gpu.func @test_floor
  entry @test_floor() {
    // CHECK: %[[FLOOR_SRC:.*]] = arith.constant 1.500000e+00 : f32
    %source = constant <f32: 1.5> : tile<f32>
    // CHECK: %[[FLOOR_RES:.*]] = math.floor %[[FLOOR_SRC]] : f32
    %result = floor %source : tile<f32>
    return
  }

  // --- log2 ---
  // CHECK-LABEL: gpu.func @test_log2
  entry @test_log2() {
    // CHECK: %[[LOG2_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: %[[LOG2_RES:.*]] = math.log2 %[[LOG2_IN]] : vector<4xf32>
    %res = log2 %in : tile<4xf32>
    return
  }

  // --- maxf ---
  // CHECK-LABEL: gpu.func @test_maxf
  // CHECK-SAME: %[[MAXF_A0:[a-zA-Z0-9_]+]]: memref<*xf32>, %[[MAXF_A1:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_maxf(%arg0: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // CHECK: %[[MAXF_MR0:.*]] = memref.reinterpret_cast %[[MAXF_A0]] to offset: [0], sizes: [2, 4], strides: [4, 1] : memref<*xf32> to memref<2x4xf32>
    %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
    // CHECK: %[[MAXF_MR1:.*]] = memref.reinterpret_cast %[[MAXF_A1]] to offset: [0], sizes: [2, 4], strides: [4, 1] : memref<*xf32> to memref<2x4xf32>
    %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
    %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
    %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
    %c0 = constant <i32: 0> : tile<i32>
    // CHECK: %[[MAXF_T0:.*]] = vector.transfer_read %[[MAXF_MR0]]{{.*}} : memref<2x4xf32>, vector<2x4xf32>
    %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, !cuda_tile.token
    // CHECK: %[[MAXF_T1:.*]] = vector.transfer_read %[[MAXF_MR1]]{{.*}} : memref<2x4xf32>, vector<2x4xf32>
    %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, !cuda_tile.token
    // CHECK: %[[MAXF_R:.*]] = arith.maxnumf %[[MAXF_T0]], %[[MAXF_T1]] : vector<2x4xf32>
    %5 = maxf %2, %3 : tile<2x4xf32>
    return
  }

  // --- maxi ---
  // CHECK-LABEL: gpu.func @test_maxi
  // CHECK-SAME: %[[MAXI_A0:[a-zA-Z0-9_]+]]: memref<*xi32>, %[[MAXI_A1:[a-zA-Z0-9_]+]]: memref<*xi32>
  entry @test_maxi(%arg0: !cuda_tile.tile<!cuda_tile.ptr<i32>>, %arg1: !cuda_tile.tile<!cuda_tile.ptr<i32>>) {
    %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
    %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
    %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
    %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
    %c0 = constant <i32: 0> : tile<i32>
    // CHECK: %[[MAXI_T0:.*]] = vector.transfer_read %{{.*}} : memref<2x4xi32>, vector<2x4xi32>
    %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, !cuda_tile.token
    // CHECK: %[[MAXI_T1:.*]] = vector.transfer_read %{{.*}} : memref<2x4xi32>, vector<2x4xi32>
    %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, !cuda_tile.token
    // CHECK: %[[MAXI_U:.*]] = arith.maxui %[[MAXI_T0]], %[[MAXI_T1]] : vector<2x4xi32>
    %4 = maxi %2, %3 unsigned : tile<2x4xi32>
    // CHECK: %[[MAXI_S:.*]] = arith.maxsi %[[MAXI_T0]], %[[MAXI_T1]] : vector<2x4xi32>
    %5 = maxi %2, %3 signed : tile<2x4xi32>
    return
  }

  // --- minf ---
  // CHECK-LABEL: gpu.func @test_minf
  // CHECK-SAME: %[[MINF_A0:[a-zA-Z0-9_]+]]: memref<*xf32>, %[[MINF_A1:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_minf(%arg0: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // CHECK: %[[MINF_MR0:.*]] = memref.reinterpret_cast %[[MINF_A0]]{{.*}} : memref<*xf32> to memref<2x4xf32>
    %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
    // CHECK: %[[MINF_MR1:.*]] = memref.reinterpret_cast %[[MINF_A1]]{{.*}} : memref<*xf32> to memref<2x4xf32>
    %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
    %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
    %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
    %c0 = constant <i32: 0> : tile<i32>
    // CHECK: %[[MINF_T0:.*]] = vector.transfer_read %[[MINF_MR0]]{{.*}} : memref<2x4xf32>, vector<2x4xf32>
    %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, !cuda_tile.token
    // CHECK: %[[MINF_T1:.*]] = vector.transfer_read %[[MINF_MR1]]{{.*}} : memref<2x4xf32>, vector<2x4xf32>
    %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, !cuda_tile.token
    // CHECK: %[[MINF_R:.*]] = arith.minnumf %[[MINF_T0]], %[[MINF_T1]] : vector<2x4xf32>
    %5 = minf %2, %3 : tile<2x4xf32>
    return
  }

  // --- mini ---
  // CHECK-LABEL: gpu.func @test_mini
  // CHECK-SAME: %[[MINI_A0:[a-zA-Z0-9_]+]]: memref<*xi32>, %[[MINI_A1:[a-zA-Z0-9_]+]]: memref<*xi32>
  entry @test_mini(%arg0: !cuda_tile.tile<!cuda_tile.ptr<i32>>, %arg1: !cuda_tile.tile<!cuda_tile.ptr<i32>>) {
    %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
    %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
    %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
    %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
    %c0 = constant <i32: 0> : tile<i32>
    // CHECK: %[[MINI_T0:.*]] = vector.transfer_read %{{.*}} : memref<2x4xi32>, vector<2x4xi32>
    %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, !cuda_tile.token
    // CHECK: %[[MINI_T1:.*]] = vector.transfer_read %{{.*}} : memref<2x4xi32>, vector<2x4xi32>
    %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, !cuda_tile.token
    // CHECK: %[[MINI_U:.*]] = arith.minui %[[MINI_T0]], %[[MINI_T1]] : vector<2x4xi32>
    %4 = mini %2, %3 unsigned : tile<2x4xi32>
    // CHECK: %[[MINI_S:.*]] = arith.minsi %[[MINI_T0]], %[[MINI_T1]] : vector<2x4xi32>
    %5 = mini %2, %3 signed : tile<2x4xi32>
    return
  }

  // --- mmaf ---
  // CHECK-LABEL: gpu.func @test_mmaf
  entry @test_mmaf() {
    // CHECK: %[[MMAF_LHS0:.*]] = arith.constant dense<0.000000e+00> : vector<4x8xf16>
    %lhs0 = constant <f16: 0.0> : tile<4x8xf16>
    // CHECK: %[[MMAF_RHS0:.*]] = arith.constant dense<0.000000e+00> : vector<8x2xf16>
    %rhs0 = constant <f16: 0.0> : tile<8x2xf16>
    // CHECK: %[[MMAF_ACC0:.*]] = arith.constant dense<0.000000e+00> : vector<4x2xf32>
    %acc0 = constant <f32: 0.0> : tile<4x2xf32>
    // CHECK: %[[MMAF_R0:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[MMAF_LHS0]], %[[MMAF_RHS0]], %[[MMAF_ACC0]] : vector<4x8xf16>, vector<8x2xf16> into vector<4x2xf32>
    %0 = mmaf %lhs0, %rhs0, %acc0 : tile<4x8xf16>, tile<8x2xf16>, tile<4x2xf32>

    // CHECK: %[[MMAF_LHS1:.*]] = arith.constant dense<0.000000e+00> : vector<2x4x8xf16>
    %lhs1 = constant <f16: 0.0> : tile<2x4x8xf16>
    // CHECK: %[[MMAF_RHS1:.*]] = arith.constant dense<0.000000e+00> : vector<2x8x2xf16>
    %rhs1 = constant <f16: 0.0> : tile<2x8x2xf16>
    // CHECK: %[[MMAF_ACC1:.*]] = arith.constant dense<0.000000e+00> : vector<2x4x2xf32>
    %acc1 = constant <f32: 0.0> : tile<2x4x2xf32>
    // CHECK: %[[MMAF_R1:.*]] = vector.contract {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[MMAF_LHS1]], %[[MMAF_RHS1]], %[[MMAF_ACC1]] : vector<2x4x8xf16>, vector<2x8x2xf16> into vector<2x4x2xf32>
    %1 = mmaf %lhs1, %rhs1, %acc1 : tile<2x4x8xf16>, tile<2x8x2xf16>, tile<2x4x2xf32>
    return
  }

  // --- mmai ---
  // CHECK-LABEL: gpu.func @test_mmai
  entry @test_mmai() {
    // CHECK: %[[MMAI_LHS0:.*]] = arith.constant dense<0> : vector<4x8xi8>
    %lhs0 = constant <i8: 0> : tile<4x8xi8>
    // CHECK: %[[MMAI_RHS0:.*]] = arith.constant dense<0> : vector<8x2xi8>
    %rhs0 = constant <i8: 0> : tile<8x2xi8>
    // CHECK: %[[MMAI_ACC0:.*]] = arith.constant dense<0> : vector<4x2xi32>
    %acc0 = constant <i32: 0> : tile<4x2xi32>
    // CHECK: %[[MMAI_R0:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[MMAI_LHS0]], %[[MMAI_RHS0]], %[[MMAI_ACC0]] {"tir-dropped-signedness-lhs" = "signed", "tir-dropped-signedness-rhs" = "signed"} : vector<4x8xi8>, vector<8x2xi8> into vector<4x2xi32>
    %0 = mmai %lhs0, %rhs0, %acc0 signed signed : tile<4x8xi8>, tile<8x2xi8>, tile<4x2xi32>

    // CHECK: %[[MMAI_LHS1:.*]] = arith.constant dense<0> : vector<2x4x8xi8>
    %lhs1 = constant <i8: 0> : tile<2x4x8xi8>
    // CHECK: %[[MMAI_RHS1:.*]] = arith.constant dense<0> : vector<2x8x2xi8>
    %rhs1 = constant <i8: 0> : tile<2x8x2xi8>
    // CHECK: %[[MMAI_ACC1:.*]] = arith.constant dense<0> : vector<2x4x2xi32>
    %acc1 = constant <i32: 0> : tile<2x4x2xi32>
    // CHECK: %[[MMAI_R1:.*]] = vector.contract {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[MMAI_LHS1]], %[[MMAI_RHS1]], %[[MMAI_ACC1]] {"tir-dropped-signedness-lhs" = "unsigned", "tir-dropped-signedness-rhs" = "unsigned"} : vector<2x4x8xi8>, vector<2x8x2xi8> into vector<2x4x2xi32>
    %1 = mmai %lhs1, %rhs1, %acc1 unsigned unsigned : tile<2x4x8xi8>, tile<2x8x2xi8>, tile<2x4x2xi32>
    return
  }

  // --- mulhii ---
  // CHECK-LABEL: gpu.func @test_mulhii
  entry @test_mulhii() {
    // CHECK: %[[MULHI_A:.*]] = arith.constant -2147483648 : i32
    %a = constant <i32: 2147483648> : tile<i32>
    // CHECK: %[[MULHI_B:.*]] = arith.constant 2 : i32
    %b = constant <i32: 2> : tile<i32>
    // CHECK: %[[MULHI_LO:.*]], %[[MULHI_HI:.*]] = arith.mului_extended %[[MULHI_A]], %[[MULHI_B]] : i32
    %res_hi = mulhii %a, %b : tile<i32>
    return
  }

  // --- muli ---
  // CHECK-LABEL: gpu.func @test_muli
  entry @test_muli() {
    // CHECK: %[[MULI_A:.*]] = arith.constant 6 : i32
    %a = constant <i32: 6> : tile<i32>
    // CHECK: %[[MULI_B:.*]] = arith.constant 7 : i32
    %b = constant <i32: 7> : tile<i32>
    // CHECK: %[[MULI_R:.*]] = arith.muli %[[MULI_A]], %[[MULI_B]] : i32
    %res = muli %a, %b : tile<i32>
    return
  }

  // --- negf ---
  // CHECK-LABEL: gpu.func @test_negf
  entry @test_negf() {
    // CHECK: %[[NEGF_IN:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    %source = constant <f32: 0.0> : tile<4xf32>
    // CHECK: %[[NEGF_R:.*]] = arith.negf %[[NEGF_IN]] : vector<4xf32>
    %result = negf %source : tile<4xf32>
    return
  }

  // --- negi ---
  // CHECK-LABEL: gpu.func @test_negi
  entry @test_negi() {
    // CHECK: %[[NEGI_IN:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi16>
    %source = constant <i16: [0, 1, 2, 3]> : tile<4xi16>
    // CHECK: %[[NEGI_ZERO:.*]] = arith.constant dense<0> : vector<4xi16>
    // CHECK: %[[NEGI_R:.*]] = arith.subi %[[NEGI_ZERO]], %[[NEGI_IN]] {"tir-dropped-overflow" = "none"} : vector<4xi16>
    %result = negi %source : tile<4xi16>
    return
  }

  // --- pow ---
  // CHECK-LABEL: gpu.func @test_pow
  entry @test_pow() {
    // CHECK: %[[POW_BASE:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    %source = constant <f32: 0.0> : tile<4xf32>
    // CHECK: %[[POW_EXP:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
    %exponent = constant <f32: 2.0> : tile<4xf32>
    // CHECK: %[[POW_R:.*]] = math.powf %[[POW_BASE]], %[[POW_EXP]] : vector<4xf32>
    %result = pow %source, %exponent : tile<4xf32>
    return
  }

  // --- rsqrt ---
  // CHECK-LABEL: gpu.func @test_rsqrt
  entry @test_rsqrt() {
    // CHECK: %[[RSQRT_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: %[[RSQRT_R:.*]] = math.rsqrt %[[RSQRT_IN]] : vector<4xf32>
    %res = rsqrt %in : tile<4xf32>
    return
  }

  // --- sin ---
  // CHECK-LABEL: gpu.func @test_sin
  entry @test_sin() {
    // CHECK: %[[SIN_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: %[[SIN_R:.*]] = math.sin %[[SIN_IN]] : vector<4xf32>
    %res = sin %in : tile<4xf32>
    return
  }

  // --- tanh ---
  // CHECK-LABEL: gpu.func @test_tanh
  entry @test_tanh() {
    // CHECK: %[[TANH_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: %[[TANH_R:.*]] = math.tanh %[[TANH_IN]] {"tir-dropped-rounding" = "full"} : vector<4xf32>
    %res0 = tanh %in : tile<4xf32>
    return
  }

  // --- xori ---
  // CHECK-LABEL: gpu.func @test_xori
  entry @test_xori() {
    // CHECK: %[[XORI_LHS:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %lhs = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
    // CHECK: %[[XORI_RHS:.*]] = arith.constant dense<[4, 5, 6, 7]> : vector<4xi32>
    %rhs = constant <i32: [4, 5, 6, 7]> : tile<4xi32>
    // CHECK: %[[XORI_R:.*]] = arith.xori %[[XORI_LHS]], %[[XORI_RHS]] : vector<4xi32>
    %result = xori %lhs, %rhs : tile<4xi32>
    return
  }

  // --- assume ---
  // CHECK-LABEL: gpu.func @test_assume
  entry @test_assume() {
    // CHECK: %[[ASSUME_V:.*]] = arith.constant dense<[32, 64, 0, 0, 32, -32, 1024, 0]> : vector<8xi16>
    %int_tile = constant <i16: [32, 64, 0, 0, 32, -32, 1024, 0]> : tile<8xi16>
    // CHECK-NOT: assume
    // CHECK: gpu.return
    %div_by_1 = assume #cuda_tile.div_by<32>, %int_tile : tile<8xi16>
    return
  }

  // --- get_tile_block_id ---
  // CHECK-LABEL: gpu.func @test_get_tile_block_id
  entry @test_get_tile_block_id() {
    // CHECK: %[[BID_X:.*]] = gpu.block_id x
    // CHECK: %[[BID_XI:.*]] = arith.index_cast %[[BID_X]] : index to i32
    // CHECK: %[[BID_Y:.*]] = gpu.block_id y
    // CHECK: %[[BID_YI:.*]] = arith.index_cast %[[BID_Y]] : index to i32
    // CHECK: %[[BID_Z:.*]] = gpu.block_id z
    // CHECK: %[[BID_ZI:.*]] = arith.index_cast %[[BID_Z]] : index to i32
    %x, %y, %z = get_tile_block_id : tile<i32>
    return
  }

  // --- get_num_tile_blocks ---
  // CHECK-LABEL: gpu.func @test_get_num_tile_blocks
  entry @test_get_num_tile_blocks() {
    // CHECK: %[[GD_X:.*]] = gpu.grid_dim x
    // CHECK: %[[GD_XI:.*]] = arith.index_cast %[[GD_X]] : index to i32
    // CHECK: %[[GD_Y:.*]] = gpu.grid_dim y
    // CHECK: %[[GD_YI:.*]] = arith.index_cast %[[GD_Y]] : index to i32
    // CHECK: %[[GD_Z:.*]] = gpu.grid_dim z
    // CHECK: %[[GD_ZI:.*]] = arith.index_cast %[[GD_Z]] : index to i32
    %x, %y, %z = get_num_tile_blocks : tile<i32>
    return
  }

  // --- for / continue ---
  // CHECK-LABEL: gpu.func @test_for
  entry @test_for() {
    // CHECK: %[[FOR_LB:.*]] = arith.constant 0 : i32
    %lowerBound = constant <i32: 0> : tile<i32>
    // CHECK: %[[FOR_UB:.*]] = arith.constant 10 : i32
    %upperBound = constant <i32: 10> : tile<i32>
    // CHECK: %[[FOR_STEP:.*]] = arith.constant 1 : i32
    %step = constant <i32: 1> : tile<i32>

    // CHECK: %[[FOR_LBI:.*]] = arith.index_cast %[[FOR_LB]] : i32 to index
    // CHECK: %[[FOR_UBI:.*]] = arith.index_cast %[[FOR_UB]] : i32 to index
    // CHECK: %[[FOR_STEPI:.*]] = arith.index_cast %[[FOR_STEP]] : i32 to index
    // CHECK: scf.for %[[FOR_IV:.*]] = %[[FOR_LBI]] to %[[FOR_UBI]] step %[[FOR_STEPI]] {
    // CHECK:   arith.index_cast %[[FOR_IV]] : index to i32
    for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
        continue
    }

    // CHECK: %[[FOR_INIT:.*]] = arith.constant 0.000000e+00 : f32
    %initVal0 = constant <f32: 0.0> : tile<f32>
    // CHECK: %[[FOR2_LBI:.*]] = arith.index_cast %[[FOR_LB]] : i32 to index
    // CHECK: %[[FOR2_UBI:.*]] = arith.index_cast %[[FOR_UB]] : i32 to index
    // CHECK: %[[FOR2_STEPI:.*]] = arith.index_cast %[[FOR_STEP]] : i32 to index
    // CHECK: %[[FOR2_R:.*]] = scf.for %{{.*}} = %[[FOR2_LBI]] to %[[FOR2_UBI]] step %[[FOR2_STEPI]] iter_args(%[[FOR2_ACC:.*]] = %[[FOR_INIT]]) -> (f32) {
    // CHECK:   %[[FOR2_VAL:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK:   scf.yield %[[FOR2_VAL]] : f32
    %results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
                        iter_values(%val00 = %initVal0) -> (tile<f32>) {
        %loopVal0 = constant <f32: 1.0> : tile<f32>
        continue %loopVal0 : tile<f32>
    }
    return
  }

  // --- reshape ---
  // CHECK-LABEL: gpu.func @test_reshape
  entry @test_reshape() {
    // scalar -> vector (broadcast): tile<i8> -> tile<1x1x1xi8>
    // CHECK: %[[S0:.*]] = arith.constant 0 : i8
    %cst = constant <i8: 0> : tile<i8>
    // CHECK: %[[R0:.*]] = vector.broadcast %[[S0]] : i8 to vector<1x1x1xi8>
    %0 = reshape %cst : tile<i8> -> tile<1x1x1xi8>

    // vector -> vector (shape_cast): tile<8x2xf32> -> tile<2x2x4x1xf32>
    // CHECK: %[[S1:.*]] = arith.constant dense<0.000000e+00> : vector<8x2xf32>
    %t = constant <f32: 0.0> : tile<8x2xf32>
    // CHECK: %[[R1:.*]] = vector.shape_cast %[[S1]] : vector<8x2xf32> to vector<2x2x4x1xf32>
    %1 = reshape %t : tile<8x2xf32> -> tile<2x2x4x1xf32>

    // vector -> vector (shape_cast): tile<2x4xi32> -> tile<2x2x2xi32>
    // CHECK: %[[S2:.*]] = arith.constant dense<{{\[}}[0, 1, 2, 3], [4, 5, 6, 7]]> : vector<2x4xi32>
    %cst2 = constant <i32: [[0, 1, 2, 3], [4, 5, 6, 7]]> : tile<2x4xi32>
    // CHECK: %[[R2:.*]] = vector.shape_cast %[[S2]] : vector<2x4xi32> to vector<2x2x2xi32>
    %2 = reshape %cst2 : tile<2x4xi32> -> tile<2x2x2xi32>

    // vector -> vector (flatten): tile<2x4xi32> -> tile<8xi32>
    // CHECK: %[[R3:.*]] = vector.shape_cast %[[S2]] : vector<2x4xi32> to vector<8xi32>
    %3 = reshape %cst2 : tile<2x4xi32> -> tile<8xi32>

    // vector -> vector (unflatten): tile<8xi32> -> tile<2x2x2xi32>
    // CHECK: %[[R4:.*]] = vector.shape_cast %[[R3]] : vector<8xi32> to vector<2x2x2xi32>
    %4 = reshape %3 : tile<8xi32> -> tile<2x2x2xi32>

    return
  }

  // --- if ---
  // CHECK-LABEL: gpu.func @test_if
  entry @test_if() {
    // CHECK: %[[COND:.*]] = arith.constant true
    %condition = constant <i1: 1> : tile<i1>

    // Simple if with no results.
    // CHECK: scf.if %[[COND]] {
    // CHECK: }
    if %condition {
    }

    // If with else, no results.
    // CHECK: scf.if %[[COND]] {
    // CHECK: } else {
    // CHECK: }
    if %condition {
    } else {
    }

    // If with else, returning mixed types (f32, i32).
    // CHECK: %[[IF_RES:.*]]:2 = scf.if %[[COND]] -> (f32, i32) {
    // CHECK:   %[[XT:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK:   %[[YT:.*]] = arith.constant 2 : i32
    // CHECK:   scf.yield %[[XT]], %[[YT]] : f32, i32
    // CHECK: } else {
    // CHECK:   %[[XE:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK:   %[[YE:.*]] = arith.constant 42 : i32
    // CHECK:   scf.yield %[[XE]], %[[YE]] : f32, i32
    // CHECK: }
    %x, %y = if %condition -> (tile<f32>, tile<i32>) {
      %x_then = constant <f32: 1.0> : tile<f32>
      %y_then = constant <i32: 2> : tile<i32>
      yield %x_then, %y_then : tile<f32>, tile<i32>
    } else {
      %x_else = constant <f32: 1.0> : tile<f32>
      %y_else = constant <i32: 42> : tile<i32>
      yield %x_else, %y_else : tile<f32>, tile<i32>
    }

    return
  }

  // --- reduce (1D -> scalar, addf) ---
  // CHECK-LABEL: gpu.func @test_reduce_addf_1d
  entry @test_reduce_addf_1d() {
    // CHECK: %[[RED1_IN:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
    %input = constant <f32: 0.0> : tile<8xf32>
    // CHECK: %[[RED1_ACC:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[RED1_R:.*]] = vector.reduction <add>, %[[RED1_IN]], %[[RED1_ACC]] : vector<8xf32> into f32
    %0 = reduce %input dim=0 identities=[0.000000e+00 : f32] : tile<8xf32> -> tile<f32>
      (%input_arg: tile<f32>, %input_accum: tile<f32>) {
        %add_result = addf %input_arg, %input_accum : tile<f32>
        yield %add_result : tile<f32>
      }
    return
  }

  // --- reduce (2D -> 1D, addf along dim 0) ---
  // CHECK-LABEL: gpu.func @test_reduce_addf_2d
  entry @test_reduce_addf_2d() {
    // CHECK: %[[RED2_IN:.*]] = arith.constant dense<0.000000e+00> : vector<8x64xf32>
    %input = constant <f32: 0.0> : tile<8x64xf32>
    // CHECK: %[[RED2_ACC:.*]] = arith.constant dense<0.000000e+00> : vector<64xf32>
    // CHECK: %[[RED2_R:.*]] = vector.multi_reduction <add>, %[[RED2_IN]], %[[RED2_ACC]] [0] : vector<8x64xf32> to vector<64xf32>
    %0 = reduce %input dim=0 identities=[0.000000e+00 : f32] : tile<8x64xf32> -> tile<64xf32>
      (%input_arg: tile<f32>, %input_accum: tile<f32>) {
        %add_result = addf %input_arg, %input_accum : tile<f32>
        yield %add_result : tile<f32>
      }
    return
  }

  // --- scan (2D inclusive product along dim 1) ---
  // CHECK-LABEL: gpu.func @test_scan_mulf_2d
  entry @test_scan_mulf_2d() {
    // CHECK: %[[SCAN_IN:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    %input = constant <f32: 0.0> : tile<8x16xf32>
    // CHECK: %[[SCAN_INIT:.*]] = arith.constant dense<1.000000e+00> : vector<8xf32>
    // CHECK: %[[SCAN_R:.*]], %{{.*}} = vector.scan <mul>, %[[SCAN_IN]], %[[SCAN_INIT]] {inclusive = true, reduction_dim = 1 : i64} : vector<8x16xf32>, vector<8xf32>
    %result = scan %input dim=1 reverse=false identities=[1.0 : f32] : tile<8x16xf32> -> tile<8x16xf32>
      (%acc: tile<f32>, %elem: tile<f32>) {
        %prod = mulf %acc, %elem rounding<nearest_even>: tile<f32>
        yield %prod : tile<f32>
      }
    return
  }

  // --- select (element-wise) ---
  // cuda_tile.select has no mlirExamples; synthesized from the assembly
  // format and spec: result[i] = cond[i] ? val_if_true[i] : val_if_false[i].
  // CHECK-LABEL: gpu.func @test_select
  entry @test_select() {
    // CHECK: %[[SEL_COND:.*]] = arith.constant dense<[true, false, true, false]> : vector<4xi1>
    %cond = constant <i1: [1, 0, 1, 0]> : tile<4xi1>
    // CHECK: %[[SEL_T:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
    %t = constant <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>
    // CHECK: %[[SEL_F:.*]] = arith.constant dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : vector<4xf32>
    %f = constant <f32: [5.0, 6.0, 7.0, 8.0]> : tile<4xf32>
    // CHECK: %[[SEL_R:.*]] = arith.select %[[SEL_COND]], %[[SEL_T]], %[[SEL_F]] : vector<4xi1>, vector<4xf32>
    %r = select %cond, %t, %f : tile<4xi1>, tile<4xf32>
    return
  }

  // --- permute ---
  // CHECK-LABEL: gpu.func @test_permute
  entry @test_permute() {
    // CHECK: %[[PERM_IN:.*]] = arith.constant dense<0.000000e+00> : vector<2x4x8xf16>
    %arg0 = constant <f16: 0.0> : tile<2x4x8xf16>
    // CHECK: %[[PERM_R:.*]] = vector.transpose %[[PERM_IN]], [2, 0, 1] : vector<2x4x8xf16> to vector<8x2x4xf16>
    %0 = permute %arg0 [2, 0, 1] : tile<2x4x8xf16> -> tile<8x2x4xf16>
    return
  }

  // --- extract (from mlirExamples) ---
  // Extract subtile at slice indices [1, 2] from tile<32x8xf32> -> tile<4x2xf32>.
  // Offset = [1*4, 2*2] = [4, 4].
  // CHECK-LABEL: gpu.func @test_extract
  entry @test_extract() {
    // CHECK-DAG: %[[EXT_I:.*]] = arith.constant 1 : i32
    %c1 = constant <i32: 1> : tile<i32>
    // CHECK-DAG: %[[EXT_J:.*]] = arith.constant 2 : i32
    %c2 = constant <i32: 2> : tile<i32>
    // CHECK-DAG: %[[EXT_SRC:.*]] = arith.constant dense<0.000000e+00> : vector<32x8xf32>
    %t = constant <f32: 0.0> : tile<32x8xf32>
    // CHECK: %[[EXT_RESHAPE:.*]] = vector.shape_cast %[[EXT_SRC]] : vector<32x8xf32> to vector<8x4x4x2xf32>
    // CHECK: %[[EXT_TRANS:.*]] = vector.transpose %[[EXT_RESHAPE]], [0, 2, 1, 3] : vector<8x4x4x2xf32> to vector<8x4x4x2xf32>
    // CHECK: %[[EXT_IDX0:.*]] = arith.index_castui %[[EXT_I]] : i32 to index
    // CHECK: %[[EXT_IDX1:.*]] = arith.index_castui %[[EXT_J]] : i32 to index
    // CHECK: %[[EXT_R:.*]] = vector.extract %[[EXT_TRANS]][%[[EXT_IDX0]], %[[EXT_IDX1]]] : vector<4x2xf32> from vector<8x4x4x2xf32>
    %0 = extract %t[%c1, %c2] : tile<32x8xf32> -> tile<4x2xf32>
    return
  }

  // --- cat (from mlirExamples) ---
  // Concatenate two tiles along dim 1 and dim 0.
  // CHECK-LABEL: gpu.func @test_cat
  entry @test_cat() {
    // CHECK-DAG: %[[CAT_LHS:.*]] = arith.constant dense<0.000000e+00> : vector<2x4xf32>
    %arg0 = constant <f32: 0.0> : tile<2x4xf32>
    // CHECK-DAG: %[[CAT_RHS:.*]] = arith.constant dense<1.000000e+00> : vector<2x4xf32>
    %arg1 = constant <f32: 1.0> : tile<2x4xf32>

    // cat along dim=1: tile<2x4> ++ tile<2x4> -> tile<2x8>
    // CHECK: %[[CAT1_P:.*]] = ub.poison : vector<2x8xf32>
    // CHECK: %[[CAT1_L:.*]] = vector.insert_strided_slice %[[CAT_LHS]], %[[CAT1_P]] {offsets = [0, 0], strides = [1, 1]} : vector<2x4xf32> into vector<2x8xf32>
    // CHECK: %[[CAT1_R:.*]] = vector.insert_strided_slice %[[CAT_RHS]], %[[CAT1_L]] {offsets = [0, 4], strides = [1, 1]} : vector<2x4xf32> into vector<2x8xf32>
    %0 = cat %arg0, %arg1 dim = 1 : tile<2x4xf32>, tile<2x4xf32> -> tile<2x8xf32>

    // cat along dim=0: tile<2x4> ++ tile<2x4> -> tile<4x4>
    // CHECK: %[[CAT0_P:.*]] = ub.poison : vector<4x4xf32>
    // CHECK: %[[CAT0_L:.*]] = vector.insert_strided_slice %[[CAT_LHS]], %[[CAT0_P]] {offsets = [0, 0], strides = [1, 1]} : vector<2x4xf32> into vector<4x4xf32>
    // CHECK: %[[CAT0_R:.*]] = vector.insert_strided_slice %[[CAT_RHS]], %[[CAT0_L]] {offsets = [2, 0], strides = [1, 1]} : vector<2x4xf32> into vector<4x4xf32>
    %1 = cat %arg0, %arg1 dim = 0 : tile<2x4xf32>, tile<2x4xf32> -> tile<4x4xf32>
    return
  }

  // --- exp2: flush_to_zero dropped ---
  // CHECK-LABEL: gpu.func @test_exp2_ftz
  entry @test_exp2_ftz() {
    // CHECK: %[[EF_IN:.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: math.exp2 %[[EF_IN]] {"tir-dropped-flush-to-zero"} : vector<4xf32>
    %r = exp2 %in flush_to_zero : tile<4xf32>
    return
  }

  // --- rsqrt: flush_to_zero dropped ---
  // CHECK-LABEL: gpu.func @test_rsqrt_ftz
  entry @test_rsqrt_ftz() {
    // CHECK: %[[RF_IN:.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: math.rsqrt %[[RF_IN]] {"tir-dropped-flush-to-zero"} : vector<4xf32>
    %r = rsqrt %in flush_to_zero : tile<4xf32>
    return
  }

  // --- atomic_rmw_tko (scalar): lowered to memref.atomic_rmw ---
  // Adapted from Ops.td mlirExamples to rank-0, matching scalar ptr lowering.
  // CHECK-LABEL: gpu.func @test_atomic_rmw_tko_scalar
  // CHECK-SAME: %[[ARMW_UPTR:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_atomic_rmw_tko_scalar(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %vals = constant <f32: 7.000000e+00> : tile<f32>
    // CHECK: %[[ARMW_V:.*]] = arith.constant 7.000000e+00 : f32
    // CHECK: %[[ARMW_R0:.*]] = memref.reinterpret_cast %[[ARMW_UPTR]] to offset: [0], sizes: [], strides: [] : memref<*xf32> to memref<f32>
    // CHECK: %[[ARMW_OLD0:.*]] = memref.atomic_rmw addf %[[ARMW_V]], %[[ARMW_R0]][] {{.*dropped-memory-ordering.*relaxed.*dropped-memory-scope.*device.*}} : (f32, memref<f32>) -> f32
    %0, %res_token0 = atomic_rmw_tko relaxed device %ptr, addf, %vals : tile<ptr<f32>>, tile<f32> -> tile<f32>, !cuda_tile.token
    return
  }

}
