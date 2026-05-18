// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | FileCheck %s
// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | mlir-opt --loop-invariant-code-motion -canonicalize -cse > /dev/null

// Tests derived from cuda_tile IR op definition examples in Ops.td.
// Each entry exercises one or more supported ops.

// CHECK-LABEL: gpu.module @ops_module {
cuda_tile.module @ops_module {

  // --- constant ---
  // CHECK-LABEL: gpu.func @test_constant
  entry @test_constant() {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    %c0 = constant <i32: 0> : tile<i32>
    // CHECK: %[[C1:.*]] = arith.constant 1 : i64
    %c1 = constant <i64: 1> : tile<i64>
    // CHECK: %[[C2:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %c2 = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
    // CHECK: %[[C3_SCALAR:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[C3:.*]] = vector.broadcast %[[C3_SCALAR]] : f32 to vector<2x4xf32>
    %c3 = constant <f32: 0.0> : tile<2x4xf32>
    // CHECK: %[[C4:.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : vector<4xf64>
    %c4 = constant <f64: [0.0, 1.0, 2.0, 3.0]> : tile<4xf64>
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

    // CHECK: %[[LHS1_S:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK: %[[LHS1:.*]] = vector.broadcast %[[LHS1_S]] : f16 to vector<2x2xf16>
    %lhs1 = constant <f16: 0.0> : tile<2x2xf16>
    // CHECK: %[[RHS1_S:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK: %[[RHS1:.*]] = vector.broadcast %[[RHS1_S]] : f16 to vector<2x2xf16>
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

    // CHECK: %[[ILHS1_S:.*]] = arith.constant 0 : i64
    // CHECK: %[[ILHS1:.*]] = vector.broadcast %[[ILHS1_S]] : i64 to vector<2x2xi64>
    %lhs1 = constant <i64: 0> : tile<2x2xi64>
    // CHECK: %[[IRHS1_S:.*]] = arith.constant 0 : i64
    // CHECK: %[[IRHS1:.*]] = vector.broadcast %[[IRHS1_S]] : i64 to vector<2x2xi64>
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
    %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
    // CHECK: %[[MAXF_T1:.*]] = vector.transfer_read %[[MAXF_MR1]]{{.*}} : memref<2x4xf32>, vector<2x4xf32>
    %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
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
    %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
    // CHECK: %[[MAXI_T1:.*]] = vector.transfer_read %{{.*}} : memref<2x4xi32>, vector<2x4xi32>
    %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
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
    %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
    // CHECK: %[[MINF_T1:.*]] = vector.transfer_read %[[MINF_MR1]]{{.*}} : memref<2x4xf32>, vector<2x4xf32>
    %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
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
    %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
    // CHECK: %[[MINI_T1:.*]] = vector.transfer_read %{{.*}} : memref<2x4xi32>, vector<2x4xi32>
    %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
    // CHECK: %[[MINI_U:.*]] = arith.minui %[[MINI_T0]], %[[MINI_T1]] : vector<2x4xi32>
    %4 = mini %2, %3 unsigned : tile<2x4xi32>
    // CHECK: %[[MINI_S:.*]] = arith.minsi %[[MINI_T0]], %[[MINI_T1]] : vector<2x4xi32>
    %5 = mini %2, %3 signed : tile<2x4xi32>
    return
  }

  // --- mmaf ---
  // CHECK-LABEL: gpu.func @test_mmaf
  entry @test_mmaf() {
    // CHECK: %[[MMAF_LHS0:.*]] = vector.broadcast %{{.*}} : f16 to vector<4x8xf16>
    %lhs0 = constant <f16: 0.0> : tile<4x8xf16>
    // CHECK: %[[MMAF_RHS0:.*]] = vector.broadcast %{{.*}} : f16 to vector<8x2xf16>
    %rhs0 = constant <f16: 0.0> : tile<8x2xf16>
    // CHECK: %[[MMAF_ACC0:.*]] = vector.broadcast %{{.*}} : f32 to vector<4x2xf32>
    %acc0 = constant <f32: 0.0> : tile<4x2xf32>
    // CHECK: %[[MMAF_R0:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[MMAF_LHS0]], %[[MMAF_RHS0]], %[[MMAF_ACC0]] : vector<4x8xf16>, vector<8x2xf16> into vector<4x2xf32>
    %0 = mmaf %lhs0, %rhs0, %acc0 : tile<4x8xf16>, tile<8x2xf16>, tile<4x2xf32>

    // CHECK: %[[MMAF_LHS1:.*]] = vector.broadcast %{{.*}} : f16 to vector<2x4x8xf16>
    %lhs1 = constant <f16: 0.0> : tile<2x4x8xf16>
    // CHECK: %[[MMAF_RHS1:.*]] = vector.broadcast %{{.*}} : f16 to vector<2x8x2xf16>
    %rhs1 = constant <f16: 0.0> : tile<2x8x2xf16>
    // CHECK: %[[MMAF_ACC1:.*]] = vector.broadcast %{{.*}} : f32 to vector<2x4x2xf32>
    %acc1 = constant <f32: 0.0> : tile<2x4x2xf32>
    // CHECK: %[[MMAF_R1:.*]] = vector.contract {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[MMAF_LHS1]], %[[MMAF_RHS1]], %[[MMAF_ACC1]] : vector<2x4x8xf16>, vector<2x8x2xf16> into vector<2x4x2xf32>
    %1 = mmaf %lhs1, %rhs1, %acc1 : tile<2x4x8xf16>, tile<2x8x2xf16>, tile<2x4x2xf32>
    return
  }

  // --- mmai ---
  // CHECK-LABEL: gpu.func @test_mmai
  entry @test_mmai() {
    // CHECK: %[[MMAI_LHS0:.*]] = vector.broadcast %{{.*}} : i8 to vector<4x8xi8>
    %lhs0 = constant <i8: 0> : tile<4x8xi8>
    // CHECK: %[[MMAI_RHS0:.*]] = vector.broadcast %{{.*}} : i8 to vector<8x2xi8>
    %rhs0 = constant <i8: 0> : tile<8x2xi8>
    // CHECK: %[[MMAI_ACC0:.*]] = vector.broadcast %{{.*}} : i32 to vector<4x2xi32>
    %acc0 = constant <i32: 0> : tile<4x2xi32>
    // CHECK: %[[MMAI_R0:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[MMAI_LHS0]], %[[MMAI_RHS0]], %[[MMAI_ACC0]] : vector<4x8xi8>, vector<8x2xi8> into vector<4x2xi32>
    %0 = mmai %lhs0, %rhs0, %acc0 signed signed : tile<4x8xi8>, tile<8x2xi8>, tile<4x2xi32>

    // CHECK: %[[MMAI_LHS1:.*]] = vector.broadcast %{{.*}} : i8 to vector<2x4x8xi8>
    %lhs1 = constant <i8: 0> : tile<2x4x8xi8>
    // CHECK: %[[MMAI_RHS1:.*]] = vector.broadcast %{{.*}} : i8 to vector<2x8x2xi8>
    %rhs1 = constant <i8: 0> : tile<2x8x2xi8>
    // CHECK: %[[MMAI_ACC1:.*]] = vector.broadcast %{{.*}} : i32 to vector<2x4x2xi32>
    %acc1 = constant <i32: 0> : tile<2x4x2xi32>
    // CHECK: %[[MMAI_R1:.*]] = vector.contract {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[MMAI_LHS1]], %[[MMAI_RHS1]], %[[MMAI_ACC1]] : vector<2x4x8xi8>, vector<2x8x2xi8> into vector<2x4x2xi32>
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
    // CHECK: %[[NEGF_S:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[NEGF_IN:.*]] = vector.broadcast %[[NEGF_S]] : f32 to vector<4xf32>
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
    // CHECK: %[[NEGI_R:.*]] = arith.subi %[[NEGI_ZERO]], %[[NEGI_IN]] : vector<4xi16>
    %result = negi %source : tile<4xi16>
    return
  }

  // --- pow ---
  // CHECK-LABEL: gpu.func @test_pow
  entry @test_pow() {
    // CHECK: %[[POW_S:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[POW_BASE:.*]] = vector.broadcast %[[POW_S]] : f32 to vector<4xf32>
    %source = constant <f32: 0.0> : tile<4xf32>
    // CHECK: %[[POW_E:.*]] = arith.constant 2.000000e+00 : f32
    // CHECK: %[[POW_EXP:.*]] = vector.broadcast %[[POW_E]] : f32 to vector<4xf32>
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
    // CHECK: %[[TANH_R:.*]] = math.tanh %[[TANH_IN]] : vector<4xf32>
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
}