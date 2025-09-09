# RUN: python3 %s

#!/usr/bin/env python3
import pytest
import subprocess
import re
import csv
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

# 矩阵大小配置
MATRIX_SIZES = [
    (32, 32, 32),    # 对应 MATMUL=1
    (64, 64, 64),    # 对应 MATMUL=2
    (128, 128, 128), # 对应 MATMUL=3
    (256, 256, 256), # 对应 MATMUL=4
    (512, 512, 512), # 对应 MATMUL=5
    (1024, 1024, 1024) # 对应 MATMUL=6
]

# 卷积配置
CONV_CONFIGS = [
    {"size": 1, "kernel": 3, "in_dim": 256, "out_dim": 254},   # CONV=1: 3x3 kernel
    {"size": 2, "kernel": 5, "in_dim": 256, "out_dim": 252},   # CONV=2: 5x5 kernel
    {"size": 3, "kernel": 7, "in_dim": 256, "out_dim": 250},   # CONV=3: 7x7 kernel
    {"size": 4, "kernel": 9, "in_dim": 256, "out_dim": 248},   # CONV=4: 9x9 kernel
    {"size": 5, "kernel": 11, "in_dim": 256, "out_dim": 246},  # CONV=5: 11x11 kernel
    {"size": 6, "kernel": 13, "in_dim": 256, "out_dim": 244}   # CONV=6: 13x13 kernel
]

class BenchmarkResult:
    def __init__(self, size_label: str, implementation: str, operation: str = "matmul"):
        self.size_label = size_label
        self.implementation = implementation
        self.operation = operation
        self.cycles = 0
        self.execution_time = 0.0
        self.result_values = None
        self.success = False
    
    def __str__(self):
        return f"{self.operation} - {self.implementation} (size:{self.size_label}): {self.cycles} cycles, {self.execution_time:.4f}s"

class GemminiBenchmark:
    def __init__(self, root_dir="./"):
        self.root_dir = Path(root_dir)
        self.matmul_results = []
        self.conv_results = []
        # 创建结果目录
        os.makedirs(self.root_dir / "benchmark_results", exist_ok=True)
    
    def run_matmul_benchmark(self, matmul_size: int, runs: int = 3) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """运行特定大小的矩阵乘法基准测试，比较Gemmini和Linalg实现"""
        size_i, size_k, size_j = MATRIX_SIZES[matmul_size-1]
        size_label = f"{size_i}x{size_j}"
        
        print(f"\n正在运行 {size_label} 矩阵乘法基准测试...")
        
        # 创建结果对象
        gemmini_result = BenchmarkResult(size_label, "Gemmini", "matmul")
        linalg_result = BenchmarkResult(size_label, "Linalg", "matmul")
        
        # 运行Gemmini实现
        for _ in range(runs):
            start_time = time.time()
            gemmini_output = self._run_command(f"make gemmini-matmul-{size_i}x{size_j}-gemmini-run", matmul_size, "2", op_type="MATMUL")
            end_time = time.time()
            
            gemmini_result.execution_time += (end_time - start_time)
            cycles = self._extract_cycles(gemmini_output)
            if cycles > 0:
                gemmini_result.cycles += cycles
                gemmini_result.success = True
        
        # 运行Linalg实现
        for _ in range(runs):
            start_time = time.time()
            linalg_output = self._run_command(f"make linalg-matmul-{size_i}x{size_j}-cpu-run", matmul_size, "1", op_type="MATMUL")
            end_time = time.time()
            
            linalg_result.execution_time += (end_time - start_time)
            cycles = self._extract_cycles(linalg_output)
            if cycles > 0:
                linalg_result.cycles += cycles
                linalg_result.success = True
        
        # 计算平均值
        if runs > 0:
            gemmini_result.execution_time /= runs
            gemmini_result.cycles /= runs
            linalg_result.execution_time /= runs
            linalg_result.cycles /= runs
        
        # 保存结果到结果列表
        self.matmul_results.append(gemmini_result)
        self.matmul_results.append(linalg_result)
        
        return gemmini_result, linalg_result
    
    def run_conv_benchmark(self, conv_size: int, runs: int = 3) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """运行特定大小的卷积基准测试，比较Gemmini和Linalg实现"""
        config = CONV_CONFIGS[conv_size-1]
        kernel_size = config["kernel"]
        in_dim = config["in_dim"]
        out_dim = config["out_dim"]
        size_label = f"{in_dim}x{in_dim}_k{kernel_size}"
        
        print(f"\n正在运行 {size_label} 卷积基准测试...")
        
        # 创建结果对象
        gemmini_result = BenchmarkResult(size_label, "Gemmini", "conv")
        linalg_result = BenchmarkResult(size_label, "Linalg", "conv")
        
        # 运行Gemmini实现
        for _ in range(runs):
            start_time = time.time()
            gemmini_output = self._run_command(f"make gemmini-conv-{kernel_size}x{kernel_size}-gemmini-run", conv_size, "2", op_type="CONV")
            end_time = time.time()
            
            gemmini_result.execution_time += (end_time - start_time)
            cycles = self._extract_cycles(gemmini_output)
            if cycles > 0:
                gemmini_result.cycles += cycles
                gemmini_result.success = True
        
        # 运行Linalg实现
        for _ in range(runs):
            start_time = time.time()
            linalg_output = self._run_command(f"make linalg-conv-{kernel_size}x{kernel_size}-cpu-run", conv_size, "1", op_type="CONV")
            end_time = time.time()
            
            linalg_result.execution_time += (end_time - start_time)
            cycles = self._extract_cycles(linalg_output)
            if cycles > 0:
                linalg_result.cycles += cycles
                linalg_result.success = True
        
        # 计算平均值
        if runs > 0:
            gemmini_result.execution_time /= runs
            gemmini_result.cycles /= runs
            linalg_result.execution_time /= runs
            linalg_result.cycles /= runs
        
        # 保存结果到结果列表
        self.conv_results.append(gemmini_result)
        self.conv_results.append(linalg_result)
        
        return gemmini_result, linalg_result
    
    def _run_command(self, make_target: str, size: int, dialect_type: str, op_type: str = "MATMUL") -> str:
        """运行指定的make命令并返回输出"""
        try:
            cmd = f"{make_target} {op_type}={size} DIALECT={dialect_type}"
            print(f"执行命令: {cmd}")
            output = subprocess.check_output(
                cmd, 
                shell=True, 
                stderr=subprocess.STDOUT,
                cwd=self.root_dir,
                universal_newlines=True
            )
            return output
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败: {e.output}")
            return ""
    
    def _extract_cycles(self, output: str) -> int:
        """从命令输出中提取周期数"""
        match = re.search(r"Cycles taken (\d+)", output)
        if match:
            return int(match.group(1))
        return 0
    
    def save_matmul_results_to_csv(self, filename: str = "matmul_benchmark_results.csv"):
        """将矩阵乘法基准测试结果保存到CSV文件"""
        filepath = self.root_dir / "benchmark_results" / filename
        self._save_results_to_csv(filepath, self.matmul_results, "矩阵乘法")
    
    def save_conv_results_to_csv(self, filename: str = "conv_benchmark_results.csv"):
        """将卷积基准测试结果保存到CSV文件"""
        filepath = self.root_dir / "benchmark_results" / filename
        self._save_results_to_csv(filepath, self.conv_results, "卷积")
    
    def _save_results_to_csv(self, filepath, results, operation_name):
        """将基准测试结果保存到CSV文件"""
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["尺寸", "实现方式", "周期数", "执行时间(s)", "加速比(相对于Linalg)"])
            
            # 按大小分组
            size_groups = {}
            for result in results:
                if result.size_label not in size_groups:
                    size_groups[result.size_label] = []
                size_groups[result.size_label].append(result)
            
            # 写入结果
            for size_label, group in size_groups.items():
                gemmini_result = next((r for r in group if r.implementation == "Gemmini"), None)
                linalg_result = next((r for r in group if r.implementation == "Linalg"), None)
                
                if gemmini_result and linalg_result and linalg_result.cycles > 0:
                    speedup = linalg_result.cycles / gemmini_result.cycles
                    writer.writerow([size_label, "Gemmini", gemmini_result.cycles, 
                                    f"{gemmini_result.execution_time:.4f}", f"{speedup:.2f}x"])
                    writer.writerow([size_label, "Linalg", linalg_result.cycles, 
                                    f"{linalg_result.execution_time:.4f}", "1.00x"])
                else:
                    if gemmini_result:
                        writer.writerow([size_label, "Gemmini", gemmini_result.cycles, 
                                        f"{gemmini_result.execution_time:.4f}", "N/A"])
                    if linalg_result:
                        writer.writerow([size_label, "Linalg", linalg_result.cycles, 
                                        f"{linalg_result.execution_time:.4f}", "N/A"])
        
        print(f"{operation_name}结果已保存到 {filepath}")
    
    def generate_matmul_report(self):
        """生成矩阵乘法基准测试报告"""
        self._generate_report(self.matmul_results, "矩阵乘法")
    
    def generate_conv_report(self):
        """生成卷积基准测试报告"""
        self._generate_report(self.conv_results, "卷积")
    
    def _generate_report(self, results, operation_name):
        """生成基准测试报告"""
        if not results:
            print(f"没有可用的{operation_name}基准测试结果")
            return
        
        print(f"\n==================== {operation_name}基准测试报告 ====================")
        print("尺寸\t\t\t实现方式\t周期数\t\t执行时间(s)\t加速比")
        print("---------------------------------------------------------------------------")
        
        # 按大小分组
        size_groups = {}
        for result in results:
            if result.size_label not in size_groups:
                size_groups[result.size_label] = []
            size_groups[result.size_label].append(result)
        
        # 输出结果
        for size_label, group in sorted(size_groups.items()):
            gemmini_result = next((r for r in group if r.implementation == "Gemmini"), None)
            linalg_result = next((r for r in group if r.implementation == "Linalg"), None)
            
            if gemmini_result and linalg_result and linalg_result.cycles > 0:
                speedup = linalg_result.cycles / gemmini_result.cycles
                print(f"{size_label}\t\tGemmini\t\t{gemmini_result.cycles:,}\t{gemmini_result.execution_time:.4f}s\t{speedup:.2f}x")
                print(f"{size_label}\t\tLinalg\t\t{linalg_result.cycles:,}\t{linalg_result.execution_time:.4f}s\t1.00x")
            else:
                if gemmini_result:
                    print(f"{size_label}\t\tGemmini\t\t{gemmini_result.cycles:,}\t{gemmini_result.execution_time:.4f}s\tN/A")
                if linalg_result:
                    print(f"{size_label}\t\tLinalg\t\t{linalg_result.cycles:,}\t{linalg_result.execution_time:.4f}s\tN/A")
            print("---------------------------------------------------------------------------")

# pytest fixtures和测试函数
@pytest.fixture
def benchmark_runner():
    """返回一个GemminiBenchmark实例用于运行基准测试"""
    return GemminiBenchmark()

@pytest.mark.parametrize("matmul_size", [1, 2, 3, 4, 5, 6])
def test_matmul_performance(benchmark_runner, matmul_size):
    """测试不同大小的矩阵乘法性能"""
    gemmini_result, linalg_result = benchmark_runner.run_matmul_benchmark(matmul_size)
    
    # 确保两个实现都成功运行
    assert gemmini_result.success, f"Gemmini实现运行失败 (matmul_size={matmul_size})"
    assert linalg_result.success, f"Linalg实现运行失败 (matmul_size={matmul_size})"
    
    # 验证Gemmini实现是否比Linalg实现更快
    if gemmini_result.cycles > 0 and linalg_result.cycles > 0:
        speedup = linalg_result.cycles / gemmini_result.cycles
        print(f"矩阵乘法 Gemmini加速比: {speedup:.2f}x")
        # 不一定要求Gemmini更快，但我们应该看到某些性能差异
        assert speedup != 1.0, "Gemmini和Linalg实现的性能相同，这可能表明测试有问题"

@pytest.mark.parametrize("conv_size", [1, 2, 3, 4, 5, 6])
def test_conv_performance(benchmark_runner, conv_size):
    """测试不同卷积核大小的卷积性能"""
    gemmini_result, linalg_result = benchmark_runner.run_conv_benchmark(conv_size)
    
    # 确保两个实现都成功运行
    assert gemmini_result.success, f"Gemmini实现运行失败 (conv_size={conv_size})"
    assert linalg_result.success, f"Linalg实现运行失败 (conv_size={conv_size})"
    
    # 验证Gemmini实现是否比Linalg实现更快
    if gemmini_result.cycles > 0 and linalg_result.cycles > 0:
        speedup = linalg_result.cycles / gemmini_result.cycles
        print(f"卷积 Gemmini加速比: {speedup:.2f}x")
        # 不一定要求Gemmini更快，但我们应该看到某些性能差异
        assert speedup != 1.0, "Gemmini和Linalg实现的性能相同，这可能表明测试有问题"

def test_generate_matmul_report(benchmark_runner):
    """运行所有矩阵乘法测试并生成报告"""
    # 运行所有大小的矩阵乘法测试
    for size in range(1, 7):
        benchmark_runner.run_matmul_benchmark(size)
    
    # 生成并保存报告
    benchmark_runner.generate_matmul_report()
    benchmark_runner.save_matmul_results_to_csv()

def test_generate_conv_report(benchmark_runner):
    """运行所有卷积测试并生成报告"""
    # 运行所有大小的卷积测试
    for size in range(1, 7):
        benchmark_runner.run_conv_benchmark(size)
    # benchmark_runner.run_conv_benchmark(3)

    # 生成并保存报告
    benchmark_runner.generate_conv_report()
    benchmark_runner.save_conv_results_to_csv()

if __name__ == "__main__":
    # 直接运行脚本
    benchmark = GemminiBenchmark()
    
    print("======== 运行矩阵乘法基准测试 ========")
    for size in range(1, 7):
        benchmark.run_matmul_benchmark(size)
    
    print("======== 运行卷积基准测试 ========")
    for size in range(1, 7):
        benchmark.run_conv_benchmark(size)
    
    # 生成并保存矩阵乘法报告
    benchmark.generate_matmul_report()
    benchmark.save_matmul_results_to_csv()
    
    # 生成并保存卷积报告
    benchmark.generate_conv_report()
    benchmark.save_conv_results_to_csv()