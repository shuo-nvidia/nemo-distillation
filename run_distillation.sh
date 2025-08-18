#!/bin/bash

# NeMo-RL 蒸馏训练运行脚本
# 使用方法: ./run_distillation.sh [配置文件名] [其他参数]

set -e  # 遇到错误时退出

# 默认配置
DEFAULT_CONFIG="examples/configs/distillation_math_cl.yaml"
DEFAULT_LOG_DIR="logs/distillation"

# 颜色输出
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

# 显示帮助信息
show_help() {
    echo "NeMo-RL 蒸馏训练运行脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [配置文件名] [其他参数]"
    echo ""
    echo "参数说明:"
    echo "  配置文件名     YAML配置文件路径 (默认: $DEFAULT_CONFIG)"
    echo "  其他参数      传递给Python脚本的其他参数"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置"
    echo "  $0 examples/configs/distillation.yaml # 使用指定配置"
    echo "  $0 --help                             # 显示帮助"
    echo ""
    echo "环境变量:"
    echo "  WANDB_PROJECT     wandb项目名称 (默认: nemo-distillation)"
    echo "  WANDB_ENTITY      wandb用户名 (可选)"
    echo "  CUDA_VISIBLE_DEVICES GPU设备ID (例如: 0,1)"
}



# 检查配置文件
check_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        print_error "配置文件不存在: $config_file"
        exit 1
    fi
    
    print_success "配置文件检查完成: $config_file"
}

# 创建日志目录
create_log_dirs() {
    print_info "创建日志目录..."
    
    mkdir -p "$DEFAULT_LOG_DIR"
    mkdir -p "checkpoints/distillation"
    mkdir -p "results/distillation"
    
    print_success "日志目录创建完成"
}

# 设置环境变量
setup_environment() {
    print_info "设置环境变量..."
    
    # 设置默认的wandb项目名称
    if [[ -z "$WANDB_PROJECT" ]]; then
        export WANDB_PROJECT="nemo-distillation"
    fi
    
    # 设置CUDA设备（如果指定了）
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        print_info "使用GPU设备: $CUDA_VISIBLE_DEVICES"
    else
        print_info "使用所有可用GPU设备"
    fi
    
    print_success "环境变量设置完成"
}

# 运行训练
run_training() {
    local config_file="$1"
    shift  # 移除第一个参数，剩下的都是其他参数
    
    print_info "开始蒸馏训练..."
    print_info "配置文件: $config_file"
    print_info "其他参数: $*"
    
    # 构建完整的命令
    local cmd="uv run python examples/run_distillation.py --config $config_file $*"
    
    print_info "执行命令: $cmd"
    echo ""
    
    # 执行训练
    if eval "$cmd"; then
        print_success "蒸馏训练完成！"
    else
        print_error "蒸馏训练失败！"
        exit 1
    fi
}

# 主函数
main() {
    local config_file="$DEFAULT_CONFIG"
    local other_args=()
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                # 其他参数传递给Python脚本
                other_args+=("$1")
                shift
                ;;
            *)
                # 第一个非选项参数作为配置文件
                if [[ "$config_file" == "$DEFAULT_CONFIG" ]]; then
                    config_file="$1"
                else
                    other_args+=("$1")
                fi
                shift
                ;;
        esac
    done
    
    # 显示启动信息
    echo "=========================================="
    echo "    NeMo-RL 蒸馏训练启动器"
    echo "=========================================="
    echo ""
    
    # 执行检查和设置
    check_dependencies
    check_config "$config_file"
    create_log_dirs
    setup_environment
    
    echo ""
    print_info "准备启动蒸馏训练..."
    echo ""
    
    # 运行训练
    run_training "$config_file" "${other_args[@]}"
}

# 如果脚本被直接执行，调用主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
