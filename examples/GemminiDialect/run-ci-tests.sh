#!/bin/bash

# Gemmini CI测试脚本
# 在本地环境中运行gemmini测试

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查必要的工具是否存在
check_dependencies() {
    print_info "检查依赖工具..."
    
    local missing_tools=()
    
    # 检查make
    if ! command -v make &> /dev/null; then
        missing_tools+=("make")
    fi
    
    # 检查riscv64-unknown-linux-gnu-gcc
    if ! command -v riscv64-unknown-linux-gnu-gcc &> /dev/null; then
        missing_tools+=("riscv64-unknown-linux-gnu-gcc")
    fi
    
    # 检查riscv64-unknown-linux-gnu-g++
    if ! command -v riscv64-unknown-linux-gnu-g++ &> /dev/null; then
        missing_tools+=("riscv64-unknown-linux-gnu-g++")
    fi
    
    # 检查spike
    if ! command -v spike &> /dev/null; then
        missing_tools+=("spike")
    fi
    
    # 检查pk
    if [ ! -f "/opt/riscv/riscv64-unknown-elf/bin/pk" ]; then
        missing_tools+=("pk (Proxy Kernel)")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "缺少以下工具："
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        print_error "请安装缺失的工具后重试"
        exit 1
    fi
    
    print_success "所有依赖工具都已安装"
}

# 检查buddy工具是否存在
check_buddy_tools() {
    print_info "检查Buddy MLIR工具..."
    
    local missing_tools=()
    
    # 检查buddy-opt
    if [ ! -f "../../build/bin/buddy-opt" ]; then
        missing_tools+=("buddy-opt")
    fi
    
    # 检查buddy-translate
    if [ ! -f "../../build/bin/buddy-translate" ]; then
        missing_tools+=("buddy-translate")
    fi
    
    # 检查buddy-llc
    if [ ! -f "../../build/bin/buddy-llc" ]; then
        missing_tools+=("buddy-llc")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "缺少以下Buddy MLIR工具："
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        print_error "请先构建Buddy MLIR项目"
        exit 1
    fi
    
    print_success "所有Buddy MLIR工具都已就绪"
}

# 运行测试
run_tests() {
    local test_type="$1"
    
    case "$test_type" in
        "quick")
            print_info "运行快速CI测试..."
            make ci-test-quick
            ;;
        "basic")
            print_info "运行基础功能测试..."
            make ci-test-basic
            ;;
        "matmul")
            print_info "运行矩阵乘法测试..."
            make ci-test-matmul
            ;;
        "conv")
            print_info "运行卷积测试..."
            make ci-test-conv
            ;;
        "activation")
            print_info "运行激活函数测试..."
            make ci-test-activation
            ;;
        "full")
            print_info "运行完整CI测试套件..."
            make ci-test
            ;;
        *)
            print_error "未知的测试类型: $test_type"
            print_usage
            exit 1
            ;;
    esac
}

# 清理测试文件
cleanup() {
    print_info "清理测试文件..."
    make ci-clean
    print_success "清理完成"
}

# 显示帮助信息
print_usage() {
    echo "用法: $0 [选项] [测试类型]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -c, --clean    清理测试文件"
    echo "  -v, --verbose  详细输出"
    echo ""
    echo "测试类型:"
    echo "  quick      快速CI测试（只测试基本功能）"
    echo "  basic      基础功能测试"
    echo "  matmul     矩阵乘法测试"
    echo "  conv       卷积测试"
    echo "  activation 激活函数测试"
    echo "  full       完整CI测试套件"
    echo ""
    echo "示例:"
    echo "  $0 quick          # 运行快速测试"
    echo "  $0 full           # 运行完整测试"
    echo "  $0 --clean        # 清理测试文件"
}

# 主函数
main() {
    local test_type="quick"  # 默认测试类型
    local verbose=false
    local should_cleanup=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -c|--clean)
                should_cleanup=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -*)
                print_error "未知选项: $1"
                print_usage
                exit 1
                ;;
            *)
                test_type="$1"
                shift
                ;;
        esac
    done
    
    # 设置详细输出
    if [ "$verbose" = true ]; then
        set -x
    fi
    
    print_info "开始Gemmini CI测试..."
    print_info "测试类型: $test_type"
    
    # 检查依赖
    check_dependencies
    check_buddy_tools
    
    # 运行测试
    run_tests "$test_type"
    
    # 清理（如果需要）
    if [ "$should_cleanup" = true ]; then
        cleanup
    fi
    
    print_success "CI测试完成！"
}

# 捕获中断信号
trap 'print_error "测试被中断"; exit 1' INT TERM

# 运行主函数
main "$@"





