# Gemmini CI测试指南

本文档说明如何在spike模拟器上运行Gemmini的CI测试，以验证Gemmini方言在RISC-V环境下的正确性。

## 概述

CI测试包含以下几个部分：
- **基础功能测试**: 测试mvin-mvout、matrix-add等基本操作
- **矩阵乘法测试**: 测试不同尺寸的矩阵乘法
- **卷积测试**: 测试不同尺寸的卷积操作
- **激活函数测试**: 测试ReLU等激活函数

## 前置要求

在运行CI测试之前，请确保以下工具已安装：

### 1. RISC-V工具链
```bash
# 安装RISC-V GCC编译器
sudo apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# 或者从源码编译
git clone https://github.com/riscv/riscv-gnu-toolchain.git
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --target=riscv64-unknown-elf
make -j$(nproc)
```

### 2. Spike模拟器
```bash
git clone https://github.com/riscv-software-src/riscv-isa-sim.git
cd riscv-isa-sim
mkdir build && cd build
../configure --prefix=/usr/local
make -j$(nproc)
sudo make install
```

### 3. Proxy Kernel (pk)
```bash
git clone https://github.com/riscv-software-src/riscv-pk.git
cd riscv-pk
mkdir build && cd build
../configure --prefix=/usr/local --host=riscv64-unknown-elf
make -j$(nproc)
sudo make install
```

### 4. Buddy MLIR
确保Buddy MLIR项目已构建：
```bash
cd /path/to/buddy-mlir
mkdir build && cd build
cmake .. -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="RISCV"
make -j$(nproc)
```

## 使用方法

### 方法1: 使用Makefile目标

#### 运行快速CI测试（推荐用于日常测试）
```bash
make ci-test-quick
```

#### 运行完整CI测试套件
```bash
make ci-test
```

#### 运行特定类型的测试
```bash
# 基础功能测试
make ci-test-basic

# 矩阵乘法测试
make ci-test-matmul

# 卷积测试
make ci-test-conv

# 激活函数测试
make ci-test-activation
```

#### 清理测试文件
```bash
make ci-clean
```

#### 显示帮助信息
```bash
make ci-help
```

### 方法2: 使用CI测试脚本

#### 运行快速测试
```bash
./run-ci-tests.sh quick
```

#### 运行完整测试
```bash
./run-ci-tests.sh full
```

#### 运行特定类型测试
```bash
./run-ci-tests.sh basic      # 基础功能
./run-ci-tests.sh matmul     # 矩阵乘法
./run-ci-tests.sh conv       # 卷积
./run-ci-tests.sh activation # 激活函数
```

#### 带详细输出的测试
```bash
./run-ci-tests.sh -v full
```

#### 测试后自动清理
```bash
./run-ci-tests.sh -c full
```

#### 显示帮助信息
```bash
./run-ci-tests.sh --help
```

## CI/CD集成

### GitHub Actions

项目已包含`.github/workflows/gemmini-ci.yml`配置文件，会在以下情况自动触发：
- 推送到main或develop分支
- 创建针对main或develop分支的Pull Request
- 修改`examples/GemminiDialect/`目录下的文件

### 本地CI测试

在提交代码前，建议在本地运行CI测试：
```bash
# 快速验证
./run-ci-tests.sh quick

# 完整验证
./run-ci-tests.sh full
```

## 测试输出说明

### 成功输出示例
```
[INFO] 开始Gemmini CI测试...
[INFO] 测试类型: quick
[INFO] 检查依赖工具...
[SUCCESS] 所有依赖工具都已安装
[INFO] 检查Buddy MLIR工具...
[SUCCESS] 所有Buddy MLIR工具都已就绪
[INFO] 运行快速CI测试...
运行快速CI测试...
测试mvin-mvout...
✓ 快速CI测试通过
[SUCCESS] CI测试完成！
```

### 失败输出示例
```
✗ mvin-mvout测试失败
make: *** [ci-test-quick] Error 1
```

## 故障排除

### 常见问题

1. **找不到buddy工具**
   - 确保Buddy MLIR项目已构建
   - 检查`../../build/bin/`目录是否存在

2. **RISC-V工具链问题**
   - 确保RISC-V GCC已正确安装
   - 检查PATH环境变量

3. **Spike模拟器问题**
   - 确保spike命令可用
   - 检查gemmini扩展是否正确加载

4. **pk问题**
   - 确保pk已正确安装
   - 检查pk路径是否正确

### 调试技巧

1. **启用详细输出**
   ```bash
   ./run-ci-tests.sh -v full
   ```

2. **手动运行单个测试**
   ```bash
   make ci-test-basic
   ```

3. **检查环境变量**
   ```bash
   echo $BUDDY_OPT
   echo $PK
   ```

## 扩展测试

如需添加新的测试用例，可以：

1. 在makefile中添加新的测试目标
2. 在CI脚本中添加对应的测试逻辑
3. 更新GitHub Actions配置

## 联系支持

如果遇到问题，请：
1. 检查本文档的故障排除部分
2. 查看项目的issue页面
3. 联系项目维护者





