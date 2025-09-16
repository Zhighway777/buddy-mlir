// RUN: buddy-opt %s -cocmh-increment | FileCheck %s

func.func @test_addi_one(%arg0: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %0 = arith.addi %arg0, %c1 : i32
  func.return %0 : i32
}

// CHECK-LABEL: func.func @test_addi_one(%arg0: i32) -> i32 {
// CHECK:         %0 = cocmh.inc %arg0 : i32
// CHECK-NEXT:    func.return %0 : i32
// CHECK-NEXT:  }

func.func @test_addi_one_reverse(%arg0: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %0 = arith.addi %c1, %arg0 : i32
  func.return %0 : i32
}

// CHECK-LABEL: func.func @test_addi_one_reverse(%arg0: i32) -> i32 {
// CHECK:         %0 = cocmh.inc %arg0 : i32
// CHECK-NEXT:    func.return %0 : i32
// CHECK-NEXT:  }

func.func @test_addi_other_constant(%arg0: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %0 = arith.addi %arg0, %c2 : i32
  func.return %0 : i32
}

// CHECK-LABEL: func.func @test_addi_other_constant(%arg0: i32) -> i32 {
// CHECK:         %[[RESULT:.*]] = arith.addi %arg0, %c2 : i32
// CHECK-NEXT:    func.return %[[RESULT]] : i32
// CHECK-NEXT:  }

func.func @test_addi_no_constant(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  func.return %0 : i32
}

// CHECK-LABEL: func.func @test_addi_no_constant(%arg0: i32, %arg1: i32) -> i32 {
// CHECK:         %[[RESULT:.*]] = arith.addi %arg0, %arg1 : i32
// CHECK-NEXT:    func.return %[[RESULT]] : i32
// CHECK-NEXT:  }
