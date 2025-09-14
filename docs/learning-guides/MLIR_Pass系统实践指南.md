# MLIR Pass系统实践指南

## 实践目标

通过这个实践指南，您将：
1. **逐步实现各种类型的Pass**
2. **深入理解Pattern匹配系统**
3. **掌握Pass开发和调试技巧**
4. **学会构建完整的Pass Pipeline**
5. **获得实际的MLIR开发经验**

## 实践环境准备

### 1. 环境设置

```bash
# 确保在正确的目录
cd /zhw/buddy-mlir

# 设置环境变量
export PYTHONPATH=/zhw/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/zhw/buddy-mlir/build/python_packages:${PYTHONPATH}
export BUDDY_MLIR_BUILD_DIR=/zhw/buddy-mlir/build
export LLVM_MLIR_BUILD_DIR=/zhw/buddy-mlir/llvm/build
```

### 2. 工具准备

```bash
# 确保buddy-opt可用
which buddy-opt

# 确保mlir-opt可用
which mlir-opt

# 确保mlir-cpu-runner可用
which mlir-cpu-runner
```

## 练习1：基础Pass开发 - 常量折叠Pass

### 1.1 理解目标

实现一个Pass，将常量表达式折叠为单个常量：
- `1.0 + 2.0` → `3.0`
- `3.0 * 3.0` → `9.0`
- `9.0 - 1.0` → `8.0`

### 1.2 实现步骤

#### 步骤1：创建Pass文件结构
```bash
mkdir -p /zhw/my-passes
cd /zhw/my-passes
```

#### 步骤2：实现常量折叠Pattern
```cpp
// ConstantFoldingPattern.cpp
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

class ConstantFoldingPattern : public RewritePattern {
public:
  ConstantFoldingPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // 检查是否是算术操作
    if (!isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(op)) {
      return failure();
    }
    
    // 检查操作数是否都是常量
    auto lhs = op->getOperand(0).getDefiningOp<arith::ConstantOp>();
    auto rhs = op->getOperand(1).getDefiningOp<arith::ConstantOp>();
    
    if (!lhs || !rhs) {
      return failure();
    }
    
    // 执行常量折叠
    return foldConstants(op, lhs, rhs, rewriter);
  }
  
private:
  LogicalResult foldConstants(Operation *op, arith::ConstantOp lhs,
                              arith::ConstantOp rhs,
                              PatternRewriter &rewriter) const {
    auto lhsValue = lhs.getValue().cast<FloatAttr>().getValueAsDouble();
    auto rhsValue = rhs.getValue().cast<FloatAttr>().getValueAsDouble();
    
    double result;
    if (isa<arith::AddFOp>(op)) {
      result = lhsValue + rhsValue;
    } else if (isa<arith::SubFOp>(op)) {
      result = lhsValue - rhsValue;
    } else if (isa<arith::MulFOp>(op)) {
      result = lhsValue * rhsValue;
    } else if (isa<arith::DivFOp>(op)) {
      if (rhsValue == 0.0) {
        return failure(); // 除零错误
      }
      result = lhsValue / rhsValue;
    }
    
    // 创建新的常量操作
    auto resultAttr = FloatAttr::get(op->getResult(0).getType(), result);
    auto newConst = rewriter.create<arith::ConstantOp>(op->getLoc(), resultAttr);
    
    // 替换原操作
    rewriter.replaceOp(op, newConst.getResult());
    return success();
  }
};
```

#### 步骤3：实现Pass
```cpp
// ConstantFoldingPass.cpp
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

class ConstantFoldingPass : public PassWrapper<ConstantFoldingPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "constant-folding"; }
  StringRef getDescription() const final { return "Fold constant expressions"; }
  
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    patterns.add<ConstantFoldingPattern>(ctx);
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};
```

#### 步骤4：测试Pass
```bash
# 创建测试文件
cat > test_constant_folding.mlir << 'EOF'
module {
  func.func @test() -> f32 {
    %c1 = arith.constant 1.0 : f32
    %c2 = arith.constant 2.0 : f32
    %add = arith.addf %c1, %c2 : f32
    return %add : f32
  }
}
EOF

# 运行Pass（假设已经集成到buddy-opt中）
buddy-opt test_constant_folding.mlir -constant-folding
```

### 1.3 验证结果

预期输出：
```mlir
module {
  func.func @test() -> f32 {
    %c3 = arith.constant 3.0 : f32
    return %c3 : f32
  }
}
```

## 练习2：Pattern匹配 - 算术表达式优化

### 2.1 理解目标

实现Pattern，优化常见的算术表达式：
- `x + 0` → `x`
- `x * 1` → `x`
- `x * 0` → `0`
- `x - x` → `0`

### 2.2 实现步骤

#### 步骤1：实现算术优化Pattern
```cpp
class ArithmeticOptimizationPattern : public RewritePattern {
public:
  ArithmeticOptimizationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // 处理加法优化
    if (auto addOp = dyn_cast<arith::AddFOp>(op)) {
      return optimizeAdd(addOp, rewriter);
    }
    
    // 处理乘法优化
    if (auto mulOp = dyn_cast<arith::MulFOp>(op)) {
      return optimizeMul(mulOp, rewriter);
    }
    
    // 处理减法优化
    if (auto subOp = dyn_cast<arith::SubFOp>(op)) {
      return optimizeSub(subOp, rewriter);
    }
    
    return failure();
  }
  
private:
  LogicalResult optimizeAdd(arith::AddFOp op, PatternRewriter &rewriter) const {
    // x + 0 = x
    if (auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = rhs.getValue().dyn_cast<FloatAttr>()) {
        if (attr.getValueAsDouble() == 0.0) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }
    
    // 0 + x = x
    if (auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = lhs.getValue().dyn_cast<FloatAttr>()) {
        if (attr.getValueAsDouble() == 0.0) {
          rewriter.replaceOp(op, op.getRhs());
          return success();
        }
      }
    }
    
    return failure();
  }
  
  LogicalResult optimizeMul(arith::MulFOp op, PatternRewriter &rewriter) const {
    // x * 1 = x
    if (auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = rhs.getValue().dyn_cast<FloatAttr>()) {
        if (attr.getValueAsDouble() == 1.0) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }
    
    // 1 * x = x
    if (auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = lhs.getValue().dyn_cast<FloatAttr>()) {
        if (attr.getValueAsDouble() == 1.0) {
          rewriter.replaceOp(op, op.getRhs());
          return success();
        }
      }
    }
    
    // x * 0 = 0
    if (auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = rhs.getValue().dyn_cast<FloatAttr>()) {
        if (attr.getValueAsDouble() == 0.0) {
          auto zero = rewriter.create<arith::ConstantOp>(
              op.getLoc(), FloatAttr::get(op.getType(), 0.0));
          rewriter.replaceOp(op, zero.getResult());
          return success();
        }
      }
    }
    
    // 0 * x = 0
    if (auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = lhs.getValue().dyn_cast<FloatAttr>()) {
        if (attr.getValueAsDouble() == 0.0) {
          auto zero = rewriter.create<arith::ConstantOp>(
              op.getLoc(), FloatAttr::get(op.getType(), 0.0));
          rewriter.replaceOp(op, zero.getResult());
          return success();
        }
      }
    }
    
    return failure();
  }
  
  LogicalResult optimizeSub(arith::SubFOp op, PatternRewriter &rewriter) const {
    // x - x = 0
    if (op.getLhs() == op.getRhs()) {
      auto zero = rewriter.create<arith::ConstantOp>(
          op.getLoc(), FloatAttr::get(op.getType(), 0.0));
      rewriter.replaceOp(op, zero.getResult());
      return success();
    }
    
    return failure();
  }
};
```

#### 步骤2：测试Pattern
```bash
# 创建测试文件
cat > test_arithmetic_optimization.mlir << 'EOF'
module {
  func.func @test(%arg0: f32, %arg1: f32) -> f32 {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    
    %add_zero = arith.addf %arg0, %c0 : f32
    %mul_one = arith.mulf %arg1, %c1 : f32
    %mul_zero = arith.mulf %arg0, %c0 : f32
    %sub_self = arith.subf %arg0, %arg0 : f32
    
    %result = arith.addf %add_zero, %mul_one : f32
    return %result : f32
  }
}
EOF

# 运行Pass
buddy-opt test_arithmetic_optimization.mlir -arithmetic-optimization
```

### 2.3 验证结果

预期输出：
```mlir
module {
  func.func @test(%arg0: f32, %arg1: f32) -> f32 {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    
    // %add_zero 被优化为 %arg0
    // %mul_one 被优化为 %arg1
    // %mul_zero 被优化为 %c0
    // %sub_self 被优化为 %c0
    
    %result = arith.addf %arg0, %arg1 : f32
    return %result : f32
  }
}
```

## 练习3：转换Pass - Linalg到循环转换

### 3.1 理解目标

实现ConversionPattern，将Linalg操作转换为循环：
- `linalg.matmul` → 三层嵌套循环
- 正确处理索引和边界

### 3.2 实现步骤

#### 步骤1：实现转换Pattern
```cpp
class LinalgToLoopsPattern : public ConversionPattern {
public:
  LinalgToLoopsPattern(MLIRContext *context)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), /*benefit=*/1, context) {}
  
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto matmulOp = cast<linalg::MatmulOp>(op);
    
    // 获取操作数
    Value lhs = operands[0];
    Value rhs = operands[1];
    Value result = operands[2];
    
    // 转换为循环
    return convertToLoops(matmulOp, lhs, rhs, result, rewriter);
  }
  
private:
  LogicalResult convertToLoops(linalg::MatmulOp op, Value lhs, Value rhs, Value result,
                               ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    
    // 获取维度信息
    auto lhsType = lhs.getType().cast<MemRefType>();
    auto rhsType = rhs.getType().cast<MemRefType>();
    auto resultType = result.getType().cast<MemRefType>();
    
    int64_t m = resultType.getDimSize(0);
    int64_t n = resultType.getDimSize(1);
    int64_t k = lhsType.getDimSize(1);
    
    // 创建常量
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto mConst = rewriter.create<arith::ConstantIndexOp>(loc, m);
    auto nConst = rewriter.create<arith::ConstantIndexOp>(loc, n);
    auto kConst = rewriter.create<arith::ConstantIndexOp>(loc, k);
    
    // 创建嵌套循环
    auto iLoop = rewriter.create<scf::ForOp>(loc, c0, mConst, c1, ValueRange{});
    rewriter.setInsertionPointToStart(iLoop.getBody());
    
    auto jLoop = rewriter.create<scf::ForOp>(loc, c0, nConst, c1, ValueRange{});
    rewriter.setInsertionPointToStart(jLoop.getBody());
    
    auto kLoop = rewriter.create<scf::ForOp>(loc, c0, kConst, c1, ValueRange{});
    rewriter.setInsertionPointToStart(kLoop.getBody());
    
    // 循环体：C[i][j] += A[i][k] * B[k][j]
    auto i = iLoop.getInductionVar();
    auto j = jLoop.getInductionVar();
    auto k = kLoop.getInductionVar();
    
    auto a = rewriter.create<memref::LoadOp>(loc, lhs, ValueRange{i, k});
    auto b = rewriter.create<memref::LoadOp>(loc, rhs, ValueRange{k, j});
    auto c = rewriter.create<memref::LoadOp>(loc, result, ValueRange{i, j});
    
    auto mul = rewriter.create<arith::MulFOp>(loc, a, b);
    auto add = rewriter.create<arith::AddFOp>(loc, c, mul);
    
    rewriter.create<memref::StoreOp>(loc, add, result, ValueRange{i, j});
    
    // 替换原操作
    rewriter.replaceOp(op, ValueRange{});
    return success();
  }
};
```

#### 步骤2：实现转换Pass
```cpp
class LinalgToLoopsPass : public PassWrapper<LinalgToLoopsPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "linalg-to-loops"; }
  StringRef getDescription() const final { return "Convert Linalg operations to loops"; }
  
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    
    // 设置合法和非法的操作
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, memref::MemRefDialect>();
    target.addIllegalDialect<linalg::LinalgDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();
    
    // 设置转换模式
    RewritePatternSet patterns(ctx);
    patterns.add<LinalgToLoopsPattern>(ctx);
    
    // 应用转换
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, memref::MemRefDialect>();
  }
};
```

#### 步骤3：测试转换Pass
```bash
# 创建测试文件
cat > test_linalg_to_loops.mlir << 'EOF'
module {
  func.func @test(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) {
    linalg.matmul ins(%arg0, %arg1: memref<4x4xf32>, memref<4x4xf32>)
                  outs(%arg2: memref<4x4xf32>)
    return
  }
}
EOF

# 运行Pass
buddy-opt test_linalg_to_loops.mlir -linalg-to-loops
```

### 3.3 验证结果

预期输出应该包含三层嵌套循环：
```mlir
module {
  func.func @test(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    
    scf.for %arg3 = %c0 to %c4 step %c1 {
      scf.for %arg4 = %c0 to %c4 step %c1 {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<4x4xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<4x4xf32>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<4x4xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<4x4xf32>
        }
      }
    }
    return
  }
}
```

## 练习4：优化Pass - 矩阵乘法优化

### 4.1 理解目标

实现优化Pass，应用分块、向量化等优化：
- 根据矩阵大小选择合适的优化策略
- 应用分块优化
- 应用向量化优化

### 4.2 实现步骤

#### 步骤1：实现矩阵乘法优化Pattern
```cpp
class MatmulOptimizationPattern : public RewritePattern {
public:
  MatmulOptimizationPattern(MLIRContext *context, int64_t tileSize)
      : RewritePattern(linalg::MatmulOp::getOperationName(), /*benefit=*/1, context),
        tileSize(tileSize) {}
  
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto matmulOp = cast<linalg::MatmulOp>(op);
    
    // 检查矩阵大小
    if (!shouldOptimize(matmulOp)) {
      return failure();
    }
    
    // 应用分块优化
    return applyTiling(matmulOp, rewriter);
  }
  
private:
  int64_t tileSize;
  
  bool shouldOptimize(linalg::MatmulOp op) const {
    auto lhsType = op.getInputs()[0].getType().cast<MemRefType>();
    auto rhsType = op.getInputs()[1].getType().cast<MemRefType>();
    
    int64_t m = lhsType.getDimSize(0);
    int64_t k = lhsType.getDimSize(1);
    int64_t n = rhsType.getDimSize(1);
    
    // 如果矩阵足够大，适合优化
    return m > 32 && k > 32 && n > 32;
  }
  
  LogicalResult applyTiling(linalg::MatmulOp op, PatternRewriter &rewriter) const {
    // 使用LinalgTilingOptions配置分块参数
    LinalgTilingOptions tilingOptions;
    tilingOptions.setTileSizes({tileSize, tileSize, tileSize});
    
    // 应用分块
    auto tiledOp = tileLinalgOp(rewriter, op, tilingOptions);
    if (failed(tiledOp)) {
      return failure();
    }
    
    // 替换原操作
    rewriter.replaceOp(op, tiledOp->tiledOps);
    return success();
  }
};
```

#### 步骤2：实现优化Pass
```cpp
class MatmulOptimizationPass : public PassWrapper<MatmulOptimizationPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "matmul-optimize"; }
  StringRef getDescription() const final { return "Optimize matrix multiplication operations"; }
  
  MatmulOptimizationPass() = default;
  MatmulOptimizationPass(int64_t tileSize) : tileSize(tileSize) {}
  
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    patterns.add<MatmulOptimizationPattern>(ctx, tileSize);
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect>();
  }
  
  Option<int64_t> tileSize{*this, "tile-size",
                           llvm::cl::desc("Tile size for matrix multiplication"),
                           llvm::cl::init(64)};
};
```

#### 步骤3：测试优化Pass
```bash
# 创建测试文件
cat > test_matmul_optimization.mlir << 'EOF'
module {
  func.func @test(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) {
    linalg.matmul ins(%arg0, %arg1: memref<128x128xf32>, memref<128x128xf32>)
                  outs(%arg2: memref<128x128xf32>)
    return
  }
}
EOF

# 运行Pass
buddy-opt test_matmul_optimization.mlir -matmul-optimize -tile-size=32
```

### 4.3 验证结果

预期输出应该包含分块后的操作，可能包括：
- 分块循环
- 向量化操作
- 优化后的内存访问模式

## 练习5：Pass Pipeline - 完整优化流程

### 5.1 理解目标

构建完整的Pass Pipeline：
1. 常量折叠Pass
2. 算术优化Pass
3. Linalg优化Pass
4. 转换Pass
5. 低级优化Pass

### 5.2 实现步骤

#### 步骤1：构建Pass Pipeline
```cpp
void buildOptimizationPipeline(PassManager &pm) {
  // 阶段1：高级优化
  pm.addPass(std::make_unique<ConstantFoldingPass>());
  pm.addPass(std::make_unique<ArithmeticOptimizationPass>());
  pm.addPass(std::make_unique<MatmulOptimizationPass>());
  
  // 阶段2：转换
  pm.addPass(std::make_unique<LinalgToLoopsPass>());
  
  // 阶段3：低级优化
  pm.addPass(std::make_unique<SCFOptimizationPass>());
  pm.addPass(std::make_unique<ArithOptimizationPass>());
}
```

#### 步骤2：测试Pass Pipeline
```bash
# 创建测试文件
cat > test_pass_pipeline.mlir << 'EOF'
module {
  func.func @test(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    %c1 = arith.constant 1.0 : f32
    %c2 = arith.constant 2.0 : f32
    %add = arith.addf %c1, %c2 : f32
    
    linalg.matmul ins(%arg0, %arg1: memref<64x64xf32>, memref<64x64xf32>)
                  outs(%arg2: memref<64x64xf32>)
    return
  }
}
EOF

# 运行Pass Pipeline
buddy-opt test_pass_pipeline.mlir -constant-folding -arithmetic-optimization -matmul-optimize -linalg-to-loops
```

### 5.3 验证结果

预期输出应该展示完整的优化流程：
1. 常量折叠：`1.0 + 2.0` → `3.0`
2. 算术优化：应用各种算术优化规则
3. 矩阵乘法优化：应用分块等优化
4. 转换：将Linalg操作转换为循环
5. 低级优化：优化循环和算术操作

## 调试和验证技巧

### 1. 调试Pass执行

```bash
# 查看Pass执行前后的IR
buddy-opt input.mlir -your-pass -mlir-print-ir-after-all

# 查看特定Pass执行后的IR
buddy-opt input.mlir -your-pass -mlir-print-ir-after=your-pass

# 查看Pass执行时间
buddy-opt input.mlir -your-pass -mlir-timing

# 查看Pass统计信息
buddy-opt input.mlir -your-pass -mlir-stats
```

### 2. 验证Pass正确性

```bash
# 在错误时打印IR
buddy-opt input.mlir -your-pass -mlir-print-on-error

# 打印堆栈跟踪
buddy-opt input.mlir -your-pass -mlir-print-stacktrace-on-diagnostic

# 验证IR
buddy-opt input.mlir -your-pass -verify-diagnostics
```

### 3. 性能测试

```bash
# 使用mlir-cpu-runner测试性能
buddy-opt input.mlir -your-pass | mlir-cpu-runner -e main -entry-point-result=void
```

## 常见问题和解决方案

### 1. Pattern匹配失败

**问题：** Pattern没有匹配到预期的操作
**解决方案：**
- 检查操作类型是否正确
- 验证匹配条件是否过于严格
- 使用调试输出查看操作结构

### 2. 转换失败

**问题：** 转换过程中出现错误
**解决方案：**
- 检查类型转换是否正确
- 验证操作数数量是否匹配
- 确保所有必要的方言都已加载

### 3. Pass执行失败

**问题：** Pass执行过程中出现异常
**解决方案：**
- 添加错误处理逻辑
- 使用try-catch包装关键代码
- 验证输入IR的正确性

## 进阶练习

### 1. 实现死代码消除Pass

实现一个Pass，消除未使用的操作：
- 分析操作的使用情况
- 识别死代码
- 安全地删除未使用的操作

### 2. 实现向量化Pass

实现一个Pass，将标量操作转换为向量操作：
- 识别可向量化的操作
- 处理向量化约束
- 生成高效的向量代码

### 3. 实现融合Pass

实现一个Pass，融合相邻的操作：
- 识别可融合的操作对
- 处理融合约束
- 优化内存访问模式

## 总结

通过这个实践指南，您应该能够：

1. ✅ **实现各种类型的Pass**：从简单的常量折叠到复杂的优化Pass
2. ✅ **掌握Pattern匹配系统**：理解如何编写有效的Pattern
3. ✅ **构建Pass Pipeline**：学会组合多个Pass实现完整的优化流程
4. ✅ **调试和验证Pass**：掌握Pass开发和调试的技巧
5. ✅ **解决常见问题**：能够处理Pass开发中的各种问题

继续实践这些练习，您将获得深入的MLIR Pass开发经验！


