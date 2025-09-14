# MLIR Pass系统深入学习指南

## 学习目标

通过这个指南，您将深入掌握MLIR的Pass系统：
1. **Pass系统架构**：理解MLIR Pass的设计理念和架构
2. **Pass类型详解**：掌握不同类型的Pass及其应用场景
3. **Pattern匹配系统**：学习RewritePattern和ConversionPattern
4. **Pass管理**：理解PassManager和PassPipeline
5. **自定义Pass开发**：从零开始编写自定义优化Pass
6. **实际项目实践**：基于buddy-mlir项目进行实践

## 第一部分：Pass系统架构深入理解

### 1.1 Pass系统的设计哲学

MLIR的Pass系统基于以下核心设计理念：

#### 1.1.1 模块化设计
```cpp
// Pass是独立的优化单元
class MyOptimizationPass : public PassWrapper<MyOptimizationPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    // 独立的优化逻辑
  }
};
```

#### 1.1.2 类型安全
```cpp
// 编译时类型检查
template<typename OpT>
class OperationPass : public Pass {
  // 确保Pass只能操作特定类型的操作
};
```

#### 1.1.3 可组合性
```cpp
// Pass可以组合成Pipeline
PassManager pm(ctx);
pm.addPass(std::make_unique<PassA>());
pm.addPass(std::make_unique<PassB>());
pm.addPass(std::make_unique<PassC>());
```

### 1.2 Pass系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    MLIR Pass System                        │
├─────────────────────────────────────────────────────────────┤
│  PassManager                                                │
│  ├── PassPipeline                                          │
│  │   ├── PassA (OperationPass<ModuleOp>)                   │
│  │   ├── PassB (OperationPass<FuncOp>)                     │
│  │   └── PassC (FunctionPass)                              │
│  └── PassRegistry                                          │
├─────────────────────────────────────────────────────────────┤
│  Pattern Matching System                                   │
│  ├── RewritePattern                                        │
│  │   ├── match() -> bool                                   │
│  │   └── rewrite() -> LogicalResult                        │
│  ├── ConversionPattern                                     │
│  │   ├── matchAndRewrite() -> LogicalResult                │
│  │   └── ConversionTarget                                  │
│  └── GreedyPatternRewriteDriver                            │
├─────────────────────────────────────────────────────────────┤
│  IR Transformation                                         │
│  ├── PatternRewriter                                       │
│  │   ├── replaceOp()                                       │
│  │   ├── create<Op>()                                      │
│  │   └── eraseOp()                                         │
│  └── IRMapping                                             │
└─────────────────────────────────────────────────────────────┘
```

## 第二部分：Pass类型详解

### 2.1 OperationPass - 操作级Pass

#### 2.1.1 基本结构
```cpp
class MyOperationPass : public PassWrapper<MyOperationPass, OperationPass<ModuleOp>> {
public:
  // Pass标识
  StringRef getArgument() const final { return "my-operation-pass"; }
  StringRef getDescription() const final { return "My operation pass description"; }
  
  // 核心执行逻辑
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // 遍历模块中的所有操作
    module.walk([&](Operation *op) {
      // 处理每个操作
      processOperation(op);
    });
  }
  
  // 依赖的方言
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect>();
  }
};
```

#### 2.1.2 实际应用示例
```cpp
class LinalgOptimizationPass : public PassWrapper<LinalgOptimizationPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // 遍历所有Linalg操作
    module.walk([&](linalg::LinalgOp linalgOp) {
      // 检查操作类型
      if (auto matmulOp = dyn_cast<linalg::MatmulOp>(linalgOp)) {
        optimizeMatmul(matmulOp);
      } else if (auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp)) {
        optimizeGeneric(genericOp);
      }
    });
  }
  
private:
  void optimizeMatmul(linalg::MatmulOp op) {
    // 矩阵乘法优化逻辑
    // 1. 检查是否可以分块
    // 2. 检查是否可以向量化
    // 3. 应用优化
  }
  
  void optimizeGeneric(linalg::GenericOp op) {
    // 通用操作优化逻辑
  }
};
```

### 2.2 FunctionPass - 函数级Pass

#### 2.2.1 基本结构
```cpp
class MyFunctionPass : public PassWrapper<MyFunctionPass, FunctionPass> {
public:
  void runOnFunction() override {
    FuncOp func = getFunction();
    // 处理单个函数
    processFunction(func);
  }
};
```

#### 2.2.2 实际应用示例
```cpp
class FunctionInliningPass : public PassWrapper<FunctionInliningPass, FunctionPass> {
public:
  void runOnFunction() override {
    FuncOp func = getFunction();
    
    // 内联小函数
    func.walk([&](func::CallOp callOp) {
      if (shouldInline(callOp)) {
        inlineCall(callOp);
      }
    });
  }
  
private:
  bool shouldInline(func::CallOp callOp) {
    // 判断是否应该内联
    auto callee = callOp.getCallee();
    // 检查函数大小、调用频率等
    return isSmallFunction(callee);
  }
  
  void inlineCall(func::CallOp callOp) {
    // 执行内联操作
  }
};
```

### 2.3 ModulePass - 模块级Pass

#### 2.3.1 基本结构
```cpp
class MyModulePass : public PassWrapper<MyModulePass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // 处理整个模块
    processModule(module);
  }
};
```

#### 2.3.2 实际应用示例
```cpp
class GlobalOptimizationPass : public PassWrapper<GlobalOptimizationPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // 全局数据流分析
    performGlobalAnalysis(module);
    
    // 全局优化
    applyGlobalOptimizations(module);
  }
  
private:
  void performGlobalAnalysis(ModuleOp module) {
    // 1. 构建调用图
    // 2. 分析数据流
    // 3. 识别优化机会
  }
  
  void applyGlobalOptimizations(ModuleOp module) {
    // 1. 死代码消除
    // 2. 常量传播
    // 3. 函数内联
  }
};
```

## 第三部分：Pattern匹配系统深入理解

### 3.1 RewritePattern - 重写模式

#### 3.1.1 基本结构
```cpp
class MyRewritePattern : public RewritePattern {
public:
  MyRewritePattern(MLIRContext *context)
      : RewritePattern(/*rootName=*/"my.op", /*benefit=*/1, context) {}
  
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // 1. 匹配条件检查
    if (!match(op)) {
      return failure();
    }
    
    // 2. 执行重写
    return rewrite(op, rewriter);
  }
  
private:
  bool match(Operation *op) const {
    // 匹配逻辑
    return true;
  }
  
  LogicalResult rewrite(Operation *op, PatternRewriter &rewriter) const {
    // 重写逻辑
    return success();
  }
};
```

#### 3.1.2 实际应用示例
```cpp
class ConstantFoldingPattern : public RewritePattern {
public:
  ConstantFoldingPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // 检查是否是算术操作
    if (!isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op)) {
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

### 3.2 ConversionPattern - 转换模式

#### 3.2.1 基本结构
```cpp
class MyConversionPattern : public ConversionPattern {
public:
  MyConversionPattern(MLIRContext *context)
      : ConversionPattern(/*rootName=*/"source.op", /*benefit=*/1, context) {}
  
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 匹配条件检查
    if (!match(op)) {
      return failure();
    }
    
    // 2. 执行转换
    return convert(op, operands, rewriter);
  }
  
private:
  bool match(Operation *op) const {
    // 匹配逻辑
    return true;
  }
  
  LogicalResult convert(Operation *op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const {
    // 转换逻辑
    return success();
  }
};
```

#### 3.2.2 实际应用示例
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
    
    // 创建嵌套循环
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    // 外层循环：i
    auto iLoop = rewriter.create<scf::ForOp>(loc, c0, c0, c1, result);
    rewriter.setInsertionPointToStart(iLoop.getBody());
    
    // 中层循环：j
    auto jLoop = rewriter.create<scf::ForOp>(loc, c0, c0, c1, iLoop.getRegionIterArgs());
    rewriter.setInsertionPointToStart(jLoop.getBody());
    
    // 内层循环：k
    auto kLoop = rewriter.create<scf::ForOp>(loc, c0, c0, c1, jLoop.getRegionIterArgs());
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
    
    // 设置循环边界
    auto mConst = rewriter.create<arith::ConstantIndexOp>(loc, m);
    auto nConst = rewriter.create<arith::ConstantIndexOp>(loc, n);
    auto kConst = rewriter.create<arith::ConstantIndexOp>(loc, k);
    
    iLoop.setUpperBound(mConst);
    jLoop.setUpperBound(nConst);
    kLoop.setUpperBound(kConst);
    
    // 替换原操作
    rewriter.replaceOp(op, iLoop.getResults());
    return success();
  }
};
```

### 3.3 Pattern匹配驱动

#### 3.3.1 GreedyPatternRewriteDriver
```cpp
class MyOptimizationPass : public PassWrapper<MyOptimizationPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    // 添加模式
    patterns.add<ConstantFoldingPattern>(ctx);
    patterns.add<DeadCodeEliminationPattern>(ctx);
    patterns.add<AlgebraicSimplificationPattern>(ctx);
    
    // 应用模式
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

#### 3.3.2 ConversionTarget
```cpp
class MyConversionPass : public PassWrapper<MyConversionPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    
    // 设置合法和非法的操作
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect>();
    target.addIllegalDialect<linalg::LinalgDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();
    
    // 设置转换模式
    RewritePatternSet patterns(ctx);
    patterns.add<LinalgToLoopsPattern>(ctx);
    patterns.add<LinalgToAffinePattern>(ctx);
    
    // 应用转换
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

## 第四部分：Pass管理深入理解

### 4.1 PassManager

#### 4.1.1 基本使用
```cpp
void runPassPipeline(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  PassManager pm(ctx);
  
  // 添加Pass
  pm.addPass(std::make_unique<LinalgOptimizationPass>());
  pm.addPass(std::make_unique<VectorizationPass>());
  pm.addPass(std::make_unique<LoweringPass>());
  
  // 运行Pass Pipeline
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass pipeline failed\n";
  }
}
```

#### 4.1.2 高级配置
```cpp
void runAdvancedPassPipeline(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  PassManager pm(ctx);
  
  // 启用Pass统计
  pm.enableStatistics();
  
  // 启用Pass计时
  pm.enableTiming();
  
  // 设置Pass选项
  pm.getContext()->getOrLoadDialect<linalg::LinalgDialect>();
  
  // 添加Pass
  pm.addPass(std::make_unique<LinalgOptimizationPass>());
  
  // 嵌套Pass Manager
  OpPassManager &nestedPM = pm.nest<func::FuncOp>();
  nestedPM.addPass(std::make_unique<FunctionInliningPass>());
  
  // 运行Pass Pipeline
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass pipeline failed\n";
  }
  
  // 打印统计信息
  pm.printStatistics();
}
```

### 4.2 Pass Pipeline

#### 4.2.1 构建Pipeline
```cpp
void buildOptimizationPipeline(PassManager &pm) {
  // 阶段1：高级优化
  pm.addPass(std::make_unique<LinalgOptimizationPass>());
  pm.addPass(std::make_unique<VectorizationPass>());
  
  // 阶段2：中间表示转换
  pm.addPass(std::make_unique<LinalgToLoopsPass>());
  pm.addPass(std::make_unique<AffineOptimizationPass>());
  
  // 阶段3：低级优化
  pm.addPass(std::make_unique<SCFOptimizationPass>());
  pm.addPass(std::make_unique<ArithOptimizationPass>());
  
  // 阶段4：代码生成
  pm.addPass(std::make_unique<ConvertToLLVMPass>());
}
```

#### 4.2.2 条件Pipeline
```cpp
void buildConditionalPipeline(PassManager &pm, bool enableVectorization) {
  // 基础优化
  pm.addPass(std::make_unique<LinalgOptimizationPass>());
  
  // 条件优化
  if (enableVectorization) {
    pm.addPass(std::make_unique<VectorizationPass>());
  }
  
  // 后续优化
  pm.addPass(std::make_unique<LoweringPass>());
}
```

## 第五部分：自定义Pass开发实践

### 5.1 完整的Pass开发流程

#### 5.1.1 项目结构
```
MyPass/
├── CMakeLists.txt
├── MyPass.cpp
├── MyPass.h
└── MyPass.td
```

#### 5.1.2 Pass定义文件 (MyPass.td)
```cpp
#ifndef MY_PASS
#define MY_PASS

include "mlir/Pass/PassBase.td"

def MyOptimizationPass : Pass<"my-optimization", "ModuleOp"> {
  let summary = "My custom optimization pass";
  let description = [{
    This pass performs custom optimizations on the input module.
  }];
  
  let dependentDialects = ["linalg::LinalgDialect", "arith::ArithDialect"];
}

#endif // MY_PASS
```

#### 5.1.3 Pass实现文件 (MyPass.cpp)
```cpp
#include "MyPass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mypass {

// Pattern定义
class MyOptimizationPattern : public RewritePattern {
public:
  MyOptimizationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // 实现优化逻辑
    return success();
  }
};

// Pass实现
class MyOptimizationPass : public PassWrapper<MyOptimizationPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "my-optimization"; }
  StringRef getDescription() const final { return "My custom optimization pass"; }
  
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    patterns.add<MyOptimizationPattern>(ctx);
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect>();
  }
};

} // namespace mypass
} // namespace mlir

// Pass注册
void mlir::mypass::registerMyOptimizationPass() {
  PassRegistration<MyOptimizationPass>();
}
```

#### 5.1.4 头文件 (MyPass.h)
```cpp
#ifndef MY_PASS_H
#define MY_PASS_H

namespace mlir {
namespace mypass {

void registerMyOptimizationPass();

} // namespace mypass
} // namespace mlir

#endif // MY_PASS_H
```

#### 5.1.5 CMakeLists.txt
```cmake
add_mlir_library(MyPass
  MyPass.cpp
  LINK_LIBS PUBLIC
  MLIRCore
  MLIRLinalg
  MLIRArith
)
```

### 5.2 实际项目中的Pass开发

#### 5.2.1 基于buddy-mlir的Pass开发
```cpp
// 参考LinalgToVIRPass的实现
class LinalgToVIRPass : public PassWrapper<LinalgToVIRPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToVIRPass)
  
  StringRef getArgument() const final { return "lower-linalg-to-vir"; }
  StringRef getDescription() const final {
    return "Lower Linalg Dialect to VIR Dialect (dynamic vectors).";
  }
  
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LinalgGenericToVIRPattern>(ctx);
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, buddy::vir::VIRDialect,
                    vector::VectorDialect>();
  }
};
```

#### 5.2.2 Pass注册
```cpp
// 在buddy-opt.cpp中注册
namespace mlir {
namespace buddy {
void registerLinalgToVIRPass();
} // namespace buddy
} // namespace mlir

int main(int argc, char **argv) {
  // ... 其他代码 ...
  mlir::buddy::registerLinalgToVIRPass();
  // ... 其他代码 ...
}
```

## 第六部分：实际项目实践

### 6.1 分析现有Pass

#### 6.1.1 LinalgToVIRPass分析
```cpp
// 核心转换逻辑
struct LinalgGenericToVIRPattern : public RewritePattern {
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return rewriter.notifyMatchFailure(op, "expected linalg op");
    }
    
    // 1. 计算VIR向量形状
    SmallVector<int64_t> virShape;
    SmallVector<OpFoldResult> commonShape;
    Value vlVal;
    if (failed(computeShapeAndVL(linalgOp, rewriter, virShape, commonShape, vlVal))) {
      return failure();
    }
    
    // 2. 创建VIR区域
    createSetVLRegion(rewriter, linalgOp.getLoc(), vlVal);
    
    // 3. 转换操作数
    DenseMap<Value, Value> transformedMemRefs;
    if (failed(transformProjectedPermutationOperands(linalgOp, rewriter, commonShape, transformedMemRefs))) {
      return failure();
    }
    
    // 4. 转换操作体
    IRMapping valueMap;
    DenseMap<Value, Value> vm;
    if (failed(convertBodyToVIR(linalgOp, rewriter, virShape, valueMap, vm))) {
      return failure();
    }
    
    // 5. 存储结果
    if (failed(storeYieldValues(linalgOp, rewriter, transformedMemRefs, valueMap, vm))) {
      return failure();
    }
    
    // 6. 删除原操作
    rewriter.eraseOp(linalgOp);
    return success();
  }
};
```

### 6.2 创建自定义Pass

#### 6.2.1 矩阵乘法优化Pass
```cpp
class MatmulOptimizationPass : public PassWrapper<MatmulOptimizationPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "matmul-optimize"; }
  StringRef getDescription() const final { return "Optimize matrix multiplication operations"; }
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    module.walk([&](linalg::MatmulOp matmulOp) {
      optimizeMatmul(matmulOp);
    });
  }
  
private:
  void optimizeMatmul(linalg::MatmulOp op) {
    // 1. 检查是否可以分块
    if (canTile(op)) {
      tileMatmul(op);
    }
    
    // 2. 检查是否可以向量化
    if (canVectorize(op)) {
      vectorizeMatmul(op);
    }
    
    // 3. 检查是否可以融合
    if (canFuse(op)) {
      fuseMatmul(op);
    }
  }
  
  bool canTile(linalg::MatmulOp op) {
    // 检查矩阵大小是否适合分块
    auto lhsType = op.getInputs()[0].getType().cast<MemRefType>();
    auto rhsType = op.getInputs()[1].getType().cast<MemRefType>();
    
    int64_t m = lhsType.getDimSize(0);
    int64_t k = lhsType.getDimSize(1);
    int64_t n = rhsType.getDimSize(1);
    
    // 如果矩阵足够大，适合分块
    return m > 64 && k > 64 && n > 64;
  }
  
  void tileMatmul(linalg::MatmulOp op) {
    // 实现分块逻辑
    // 使用LinalgTilingOptions配置分块参数
  }
  
  bool canVectorize(linalg::MatmulOp op) {
    // 检查是否支持向量化
    return true;
  }
  
  void vectorizeMatmul(linalg::MatmulOp op) {
    // 实现向量化逻辑
  }
  
  bool canFuse(linalg::MatmulOp op) {
    // 检查是否可以与相邻操作融合
    return true;
  }
  
  void fuseMatmul(linalg::MatmulOp op) {
    // 实现融合逻辑
  }
};
```

### 6.3 Pass测试

#### 6.3.1 单元测试
```cpp
TEST(MatmulOptimizationPass, BasicOptimization) {
  MLIRContext context;
  context.loadDialect<linalg::LinalgDialect, arith::ArithDialect>();
  
  // 创建测试模块
  auto module = createTestModule(&context);
  
  // 运行Pass
  PassManager pm(&context);
  pm.addPass(std::make_unique<MatmulOptimizationPass>());
  
  EXPECT_TRUE(succeeded(pm.run(module)));
  
  // 验证结果
  verifyOptimizationResults(module);
}
```

#### 6.3.2 集成测试
```mlir
// RUN: buddy-opt %s -matmul-optimize -convert-linalg-to-loops | FileCheck %s

func.func @test_matmul(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) {
  linalg.matmul ins(%arg0, %arg1: memref<128x128xf32>, memref<128x128xf32>)
                outs(%arg2: memref<128x128xf32>)
  return
}

// CHECK: func.func @test_matmul
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
```

## 第七部分：性能优化和调试

### 7.1 Pass性能优化

#### 7.1.1 避免重复计算
```cpp
class OptimizedPass : public PassWrapper<OptimizedPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // 预计算分析结果
    auto analysis = performAnalysis(module);
    
    // 使用分析结果进行优化
    module.walk([&](Operation *op) {
      optimizeOperation(op, analysis);
    });
  }
  
private:
  AnalysisResult performAnalysis(ModuleOp module) {
    // 一次性分析，避免重复计算
    AnalysisResult result;
    // ... 分析逻辑 ...
    return result;
  }
  
  void optimizeOperation(Operation *op, const AnalysisResult &analysis) {
    // 使用预计算的分析结果
  }
};
```

#### 7.1.2 增量更新
```cpp
class IncrementalPass : public PassWrapper<IncrementalPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // 使用增量更新策略
    SmallVector<Operation *> worklist;
    module.walk([&](Operation *op) {
      if (needsUpdate(op)) {
        worklist.push_back(op);
      }
    });
    
    // 处理工作列表
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (optimizeOperation(op)) {
        // 添加受影响的邻居操作
        addAffectedOperations(op, worklist);
      }
    }
  }
};
```

### 7.2 Pass调试

#### 7.2.1 调试信息输出
```cpp
class DebugPass : public PassWrapper<DebugPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // 启用调试输出
    if (debugMode) {
      llvm::errs() << "=== Pass Debug Information ===\n";
      llvm::errs() << "Module: " << module.getName() << "\n";
    }
    
    module.walk([&](Operation *op) {
      if (debugMode) {
        llvm::errs() << "Processing operation: " << op->getName() << "\n";
      }
      
      processOperation(op);
    });
  }
  
private:
  bool debugMode = true;
};
```

#### 7.2.2 验证和错误处理
```cpp
class SafePass : public PassWrapper<SafePass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // 保存原始状态
    auto originalModule = module.clone();
    
    try {
      // 执行优化
      performOptimization(module);
      
      // 验证结果
      if (!verifyModule(module)) {
        llvm::errs() << "Optimization failed verification\n";
        // 恢复原始状态
        module.getBody()->getOperations().clear();
        module.getBody()->getOperations().splice(
          module.getBody()->end(),
          originalModule.getBody()->getOperations());
        signalPassFailure();
        return;
      }
    } catch (const std::exception &e) {
      llvm::errs() << "Pass failed with exception: " << e.what() << "\n";
      signalPassFailure();
    }
  }
  
private:
  void performOptimization(ModuleOp module) {
    // 优化逻辑
  }
  
  bool verifyModule(ModuleOp module) {
    // 验证逻辑
    return true;
  }
};
```

## 实践练习

### 练习1：基础Pass开发
**任务：** 创建一个简单的常量折叠Pass

### 练习2：Pattern匹配
**任务：** 实现一个重写模式，将简单的算术表达式转换为更高效的形式

### 练习3：转换Pass
**任务：** 创建一个Pass，将Linalg操作转换为循环

### 练习4：优化Pass
**任务：** 实现一个矩阵乘法优化Pass

### 练习5：Pass Pipeline
**任务：** 构建一个完整的优化Pipeline

## 学习检查点

完成这些学习后，您应该能够：

1. ✅ **深入理解Pass系统架构**：掌握MLIR Pass的设计理念和架构
2. ✅ **掌握不同Pass类型**：理解OperationPass、FunctionPass、ModulePass的应用场景
3. ✅ **熟练使用Pattern匹配**：掌握RewritePattern和ConversionPattern的使用
4. ✅ **管理Pass Pipeline**：理解PassManager和PassPipeline的配置
5. ✅ **开发自定义Pass**：从零开始编写自定义优化Pass
6. ✅ **进行Pass调试**：掌握Pass的调试和性能优化技巧

## 下一步学习方向

1. **高级Pass技术**：学习更复杂的Pass技术，如数据流分析、别名分析等
2. **Pass组合优化**：学习如何组合多个Pass实现更好的优化效果
3. **硬件特定优化**：学习针对特定硬件的Pass开发
4. **Pass性能调优**：深入学习Pass的性能优化技术

---

通过这个深入的学习指南，您将获得对MLIR Pass系统的全面理解，为进行实际的MLIR优化工作打下坚实的基础！


