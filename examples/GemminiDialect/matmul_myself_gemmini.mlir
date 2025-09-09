// RUN: buddy-opt %s -verify-diagnostics | buddy-opt | FileCheck %s

module {
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%d \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() -> i8 {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(4295032833 : i64) : i64
    %2 = llvm.mlir.constant(34360262664 : i64) : i64
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(4575657221409472785 : i64) : i64
    %5 = llvm.mlir.constant(32 : i64) : i64
    %6 = llvm.mlir.constant(4575657221409472777 : i64) : i64
    %7 = llvm.mlir.constant(4575657221409472769 : i64) : i64
    %8 = llvm.mlir.constant(4575657221408423944 : i64) : i64
    %9 = llvm.mlir.constant(2 : i64) : i64
    %10 = llvm.mlir.constant(8 : i64) : i64
    %11 = llvm.mlir.constant(281474976710656 : i64) : i64
    %12 = llvm.mlir.constant(4575657221408489476 : i64) : i64
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(0 : i8) : i8
    %16 = llvm.mlir.constant(2 : i8) : i8
    %17 = llvm.mlir.constant(8 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.mlir.zero : !llvm.ptr
    %20 = llvm.getelementptr %19[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
    %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr
    %23 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %14, %25[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %17, %26[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %17, %27[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %17, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %18, %29[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = builtin.unrealized_conversion_cast %30 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<8x8xi8>
    %32 = llvm.mlir.zero : !llvm.ptr
    %33 = llvm.getelementptr %32[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.call @malloc(%34) : (i64) -> !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.insertvalue %35, %36[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %14, %38[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %17, %39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %17, %40[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.insertvalue %17, %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %18, %42[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = builtin.unrealized_conversion_cast %43 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<8x8xi8>
    %45 = llvm.mlir.zero : !llvm.ptr
    %46 = llvm.getelementptr %45[64] : (!llvm.ptr) -> !llvm.ptr, i8
    %47 = llvm.ptrtoint %46 : !llvm.ptr to i64
    %48 = llvm.call @malloc(%47) : (i64) -> !llvm.ptr
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        memref.store %16, %31[%arg0, %arg1] : memref<8x8xi8>
      }
    }
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        memref.store %16, %44[%arg0, %arg1] : memref<8x8xi8>
      }
    }
    %49 = llvm.mlir.zero : !llvm.ptr
    %50 = llvm.getelementptr %49[64] : (!llvm.ptr) -> !llvm.ptr, i32
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.call @malloc(%51) : (i64) -> !llvm.ptr
    %53 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.insertvalue %52, %53[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %52, %54[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %14, %55[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %17, %56[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %17, %57[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %17, %58[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.insertvalue %18, %59[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = builtin.unrealized_conversion_cast %60 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<8x8xi32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        memref.store %13, %61[%arg0, %arg1] : memref<8x8xi32>
      }
    }
    %62 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %63 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %64 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %65 = llvm.ptrtoint %52 : !llvm.ptr to i64
    "gemmini.intr.config_ex"(%12, %11) : (i64, i64) -> ()
    "gemmini.intr.config_st"(%9, %8) : (i64, i64) -> ()
    "gemmini.intr.config_ld"(%7, %10) : (i64, i64) -> ()
    "gemmini.intr.config_ld"(%6, %10) : (i64, i64) -> ()
    "gemmini.intr.config_ld"(%4, %5) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_bounds"(%2, %1) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_ab"(%62, %63) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_dc"(%65, %64) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_strides_ab"(%10, %10) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_strides_dc"(%10, %10) : (i64, i64) -> ()
    "gemmini.intr.loop_ws"(%0, %3) : (i64, i64) -> ()
    "gemmini.intr.flush"(%3, %3) : (i64, i64) -> ()
    llvm.call @free(%52) : (!llvm.ptr) -> ()
    %66 = llvm.mlir.addressof @frmt_spec : !llvm.ptr
    %67 = llvm.getelementptr %66[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    %68 = llvm.mlir.addressof @nl : !llvm.ptr
    %69 = llvm.getelementptr %68[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    llvm.br ^bb1(%14 : i64)
  ^bb1(%70: i64):  // 2 preds: ^bb0, ^bb5
    %71 = llvm.icmp "slt" %70, %17 : i64
    llvm.cond_br %71, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%14 : i64)
  ^bb3(%72: i64):  // 2 preds: ^bb2, ^bb4
    %73 = llvm.icmp "slt" %72, %17 : i64
    llvm.cond_br %73, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %74 = llvm.mul %70, %17  : i64
    %75 = llvm.add %74, %72  : i64
    %76 = llvm.getelementptr %48[%75] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %77 = llvm.load %76 : !llvm.ptr -> i8
    %78 = llvm.sext %77 : i8 to i32
    %79 = llvm.call @printf(%67, %78) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %80 = llvm.add %72, %18  : i64
    llvm.br ^bb3(%80 : i64)
  ^bb5:  // pred: ^bb3
    %81 = llvm.call @printf(%69) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %82 = llvm.add %70, %18  : i64
    llvm.br ^bb1(%82 : i64)
  ^bb6:  // pred: ^bb1
    llvm.call @free(%48) : (!llvm.ptr) -> ()
    llvm.call @free(%35) : (!llvm.ptr) -> ()
    llvm.call @free(%22) : (!llvm.ptr) -> ()
    llvm.return %15 : i8
  }
}

// CHECK: module

