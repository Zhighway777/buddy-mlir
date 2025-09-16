#include "Cocmh/CocmhDialect.h"
#include "Cocmh/CocmhOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace buddy::cocmh;

namespace {
// 定义一个模式，用于将 arith.addi 转换为 cocmh.inc
struct ConvertAddiToOne : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    // 检查右操作数是否为常量1
    if (auto rhsConst = dyn_cast_or_null<arith::ConstantOp>(op.getRhs().getDefiningOp())) {
      if (auto valueAttr = dyn_cast<IntegerAttr>(rhsConst.getValue())) {
        if (valueAttr.getValue().isOne()) {
          // 创建一个新的 cocmh.inc 操作
          rewriter.replaceOpWithNewOp<buddy::cocmh::IncOp>(op, op.getLhs());
          return success();
        }
      }
    }

    // 检查左操作数是否为常量1
    if (auto lhsConst = dyn_cast_or_null<arith::ConstantOp>(op.getLhs().getDefiningOp())) {
      if (auto valueAttr = dyn_cast<IntegerAttr>(lhsConst.getValue())) {
        if (valueAttr.getValue().isOne()) {
          // 创建一个新的 cocmh.inc 操作
          rewriter.replaceOpWithNewOp<buddy::cocmh::IncOp>(op, op.getRhs());
          return success();
        }
      }
    }

    return failure();
  }
};

// 定义 Pass
struct CocmhIncrementPass : public PassWrapper<CocmhIncrementPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CocmhIncrementPass)

  StringRef getArgument() const override { return "cocmh-increment"; }
  StringRef getDescription() const override { return "Converts arith.addi with constant 1 to cocmh.inc"; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertAddiToOne>(context); // 添加转换模式

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buddy::cocmh::CocmhDialect, arith::ArithDialect>(); // 声明 Pass 依赖的方言
  }
};
} // end anonymous namespace

// 注册 Pass
namespace mlir {
namespace buddy {
void registerCocmhIncrementPass() {
  PassRegistration<CocmhIncrementPass>();
}
} // namespace buddy
} // namespace mlir