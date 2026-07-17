// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=gpu append-grid-args=false drop-rounding-modes=true' %s | FileCheck %s

// Verifies drop-rounding-modes=true forces rounding to be preserved only as
// tir-dropped-rounding, even for modes that are otherwise representable.

// CHECK-LABEL: gpu.func @drop_rounding_modes
// CHECK: arith.addf %{{.*}}, %{{.*}} {"tir-dropped-rounding" = "nearest_even"}
// CHECK: arith.divf %{{.*}}, %{{.*}} {"tir-dropped-rounding" = "approx"}
// CHECK-NOT: fastmath<arcp>
// CHECK: arith.divsi %{{.*}}, %{{.*}} {"tir-dropped-rounding" = "zero"}
// CHECK: arith.truncf %{{.*}} {"tir-dropped-rounding" = "nearest_even"}
// CHECK: arith.fptosi %{{.*}} {"tir-dropped-rounding" = "nearest_int_to_zero"}
// CHECK: arith.sitofp %{{.*}} {"tir-dropped-rounding" = "nearest_even"}
// CHECK: math.sqrt %{{.*}} {"tir-dropped-rounding" = "approx"}
// CHECK: math.tanh %{{.*}} {"tir-dropped-rounding" = "approx"}
// CHECK-NOT: fastmath<afn>
// CHECK: math.fma %{{.*}}, %{{.*}}, %{{.*}} {"tir-dropped-rounding" = "zero"}

cuda_tile.module @m {
  entry @drop_rounding_modes(%x: tile<f32>, %y: tile<f32>, %ix: tile<i32>, %iy: tile<i32>) {
    %a = addf %x, %y : tile<f32>
    %b = divf %x, %y rounding<approx> : tile<f32>
    %c = divi %ix, %iy signed : tile<i32>
    %d = ftof %x rounding<nearest_even> : tile<f32> -> tile<f16>
    %e = ftoi %x signed rounding<nearest_int_to_zero> : tile<f32> -> tile<i32>
    %f = itof %ix signed rounding<nearest_even> : tile<i32> -> tile<f32>
    %g = sqrt %x rounding<approx> : tile<f32>
    %h = tanh %x rounding<approx> : tile<f32>
    %i = fma %x, %y, %x rounding<zero> : tile<f32>
    return
  }
}
