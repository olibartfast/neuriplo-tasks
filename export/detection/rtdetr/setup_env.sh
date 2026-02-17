#!/bin/bash

# RT-DETR Virtual Environment Setup Script
#
# This script creates optimized virtual environments for RT-DETR export pipelines.
# Supports both PyTorch-based and PaddlePaddle-based workflows.
#
# Usage:
#   cd vision-core
#   export/detection/rtdetr/setup_env.sh --framework pytorch --env-name rtdetr-pytorch  --output-dir ./environments
#   export/detection/rtdetr/setup_env.sh --framework paddlepaddle --env-name rtdetr-paddle --output-dir ./environments
#   export/detection/rtdetr/setup_env.sh --help for more information.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Default values
ENV_NAME=""
PYTHON_VERSION="3.11"
CUDA_VERSION="11.8"
OUTPUT_DIR="./environments"
FORCE_CREATE=false
INSTALL_EXTRAS=true
DRY_RUN=false

# Help function
show_help() {
    cat << EOF
RT-DETR Virtual Environment Setup Script

DESCRIPTION:
    Create optimized virtual environments for RT-DETR export pipelines.
    Supports PyTorch framework with export dependencies for RT-DETR v1, v2, v4.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -n, --env-name NAME         Environment name (auto-generated if not specified)
    -p, --python-version VER    Python version (default: 3.11)
    -c, --cuda-version VER      CUDA version for GPU support (default: 11.8)
    -o, --output-dir PATH       Directory for environments (default: ./environments)
    --force                     Force recreate environment if exists
    --no-extras                 Skip optional export tools installation
    --dry-run                   Show what would be installed without executing
    -h, --help                  Show this help message

PIPELINE COMPONENTS:

PyTorch Pipeline:
    ├── PyTorch (CUDA-enabled)
    ├── TorchVision
    ├── ONNX & ONNX Runtime
    ├── ONNXSim (model optimization)
    ├── TensorRT Python bindings
    └── Export utilities

EXAMPLES:
    # Create environment with defaults
    $0

    # Create with custom name
    $0 --env-name my-rtdetr-env

    # Create with specific Python/CUDA versions
    $0 --python-version 3.10 --cuda-version 12.1

    # Force recreate environment
    $0 --force

    # Dry run to see what would be installed
    $0 --dry-run

ENVIRONMENT STRUCTURE:
    environments/
    └── rtdetr-py311-cuda118/

USAGE AFTER SETUP:
    # Activate environment
    source environments/rtdetr-py311-cuda118/bin/activate

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--env-name)
                ENV_NAME="$2"
                shift 2
                ;;
            -p|--python-version)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            -c|--cuda-version)
                CUDA_VERSION="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --force)
                FORCE_CREATE=true
                shift
                ;;
            --no-extras)
                INSTALL_EXTRAS=false
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Validate arguments
validate_arguments() {
    # Generate environment name if not provided
    if [[ -z "$ENV_NAME" ]]; then
        local python_short=$(echo "$PYTHON_VERSION" | tr -d '.')
        local cuda_short=$(echo "$CUDA_VERSION" | tr -d '.')
        ENV_NAME="rtdetr-py${python_short}-cuda${cuda_short}"
    fi
}

# Check dependencies
check_dependencies() {
    local python_cmd="python${PYTHON_VERSION}"

    # Check if specific python version exists
    if ! command -v "$python_cmd" &> /dev/null; then
        log_warning "$python_cmd not found, checking for python3..."
        if ! command -v python3 &> /dev/null; then
            log_error "Python 3 is not installed"
            exit 1
        fi

        # Check if python3 is the right version
        local ver=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [[ "$ver" != "$PYTHON_VERSION" ]]; then
            log_warning "System python3 is version $ver, but $PYTHON_VERSION was requested."
            log_warning "This might cause issues if binary wheels are not available for $ver."
        fi
    fi

    # Check pip
    if ! "$python_cmd" -m pip --version &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        log_error "pip is not installed"
        exit 1
    fi

    # Check conda (recommended for environment management)
    if command -v conda &> /dev/null; then
        ENV_MANAGER="conda"
        log_info "Using conda for environment management"
    elif "$python_cmd" -m venv --help &> /dev/null || python3 -m venv --help &> /dev/null; then
        ENV_MANAGER="venv"
        log_info "Using venv for environment management"
    else
        log_error "No suitable environment manager found (conda or venv required)"
        exit 1
    fi
}

# Check GPU availability
check_gpu_availability() {
    local has_gpu=false

    # Check for nvidia-smi command and NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            local gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
            if [[ "$gpu_count" -gt 0 ]]; then
                has_gpu=true
                log_info "NVIDIA GPU detected: $gpu_count GPU(s) available"
            fi
        fi
    fi

    # Check for CUDA toolkit
    if [[ "$has_gpu" == true ]] && command -v nvcc &> /dev/null; then
        local cuda_ver=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" || echo "unknown")
        log_info "CUDA toolkit detected: version $cuda_ver"
    fi

    if [[ "$has_gpu" == false ]]; then
        log_warning "No NVIDIA GPU detected - will install CPU versions"
    fi

    echo "$has_gpu"
}

# Get PyTorch installation commands
get_pytorch_commands() {
    local cuda_version="$1"
    local has_gpu="$2"
    local commands=()

    # Install PyTorch based on GPU availability
    if [[ "$has_gpu" == "true" ]]; then
        # Convert CUDA version for PyTorch index
        local torch_cuda
        case "$cuda_version" in
            "11.8") torch_cuda="cu118" ;;
            "12.1") torch_cuda="cu121" ;;
            "12.4") torch_cuda="cu124" ;;
            *) torch_cuda="cu118" ;;  # default
        esac
        commands+=("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$torch_cuda")
    else
        commands+=("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    fi

    commands+=(
        "pip install transformers timm"
        "pip install opencv-python pillow"
    )

    printf '%s\n' "${commands[@]}"
}


# Get common export dependencies
get_export_commands() {
    local has_gpu="$1"
    local commands=(
        "pip install 'onnx>=1.15.0'"
        "pip install onnxsim"
        "pip install 'numpy<2.0'"
        "pip install PyYAML"
        "pip install tqdm"
        "pip install matplotlib"
        "pip install pycocotools"
    )

    # Install GPU or CPU version of onnxruntime based on availability
    if [[ "$has_gpu" == "true" ]]; then
        commands+=(
            "pip install onnxruntime-gpu"
            "pip install tensorrt"
            "pip install pycuda"
        )
    else
        commands+=("pip install onnxruntime")
    fi

    if [[ "$INSTALL_EXTRAS" == true ]]; then
        commands+=(
            "pip install fiftyone"
            "pip install tensorboard"
            "pip install wandb"
            "pip install gradio"
        )
    fi

    printf '%s\n' "${commands[@]}"
}


# Create environment
create_environment() {
    local env_name="$1"
    local env_path="$OUTPUT_DIR/$env_name"

    mkdir -p "$OUTPUT_DIR"

    # Check if environment exists
    if [[ -d "$env_path" ]]; then
        if [[ "$FORCE_CREATE" == true ]]; then
            log_warning "Environment exists, removing: $env_path"
            rm -rf "$env_path"
        else
            log_error "Environment already exists: $env_path"
            log_info "Use --force to recreate"
            exit 1
        fi
    fi

    log_step "Creating $framework environment: $env_name"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would create environment at $env_path"
        return
    fi

    case "$ENV_MANAGER" in
        "conda")
            conda create -y -p "$env_path" python="$PYTHON_VERSION"
            ;;
        "venv")
            # Try to use specific python version
            local python_cmd="python${PYTHON_VERSION}"
            if ! command -v "$python_cmd" &> /dev/null; then
                log_warning "$python_cmd not found, falling back to python3"
                python_cmd="python3"
            fi

            log_info "Using $python_cmd to create venv"

            # Try standard creation
            if ! "$python_cmd" -m venv "$env_path"; then
                log_warning "Standard venv creation failed (likely missing ensurepip). Retrying without pip..."

                # Try without pip
                if "$python_cmd" -m venv --without-pip "$env_path"; then
                    log_step "Bootstrapping pip..."
                    local pip_script="/tmp/get-pip.py"
                    if curl -sS https://bootstrap.pypa.io/get-pip.py -o "$pip_script"; then
                        if "$env_path/bin/python" "$pip_script"; then
                            log_success "Pip bootstrapped successfully"
                            rm -f "$pip_script"
                        else
                            log_error "Failed to install pip via get-pip.py"
                            exit 1
                        fi
                    else
                        log_error "Failed to download get-pip.py"
                        exit 1
                    fi
                else
                    log_error "Failed to create environment even without pip"
                    exit 1
                fi
            fi
            ;;
    esac

    log_success "Environment created: $env_path"
}

# Install packages
install_packages() {
    local env_name="$1"
    local env_path="$OUTPUT_DIR/$env_name"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would install packages in $env_path"
        show_installation_plan
        return
    fi

    log_step "Installing packages for PyTorch pipeline"

    # Activate environment
    case "$ENV_MANAGER" in
        "conda")
            source "$env_path/bin/activate"
            ;;
        "venv")
            source "$env_path/bin/activate"
            ;;
    esac

    # Upgrade pip
    log_info "Upgrading pip..."
    python -m pip install --upgrade pip

    # Check GPU availability
    local has_gpu=$(check_gpu_availability)

    # Install PyTorch pipeline dependencies
    log_info "Installing PyTorch pipeline dependencies..."
    while IFS= read -r cmd; do
        log_info "Running: $cmd"
        eval "$cmd"
    done < <(get_pytorch_commands "$CUDA_VERSION" "$has_gpu")

    # Install common export dependencies
    log_info "Installing export dependencies..."
    while IFS= read -r cmd; do
        log_info "Running: $cmd"
        eval "$cmd"
    done < <(get_export_commands "$has_gpu")

    log_success "Package installation completed"
}

# Show installation plan for dry run
show_installation_plan() {
    local has_gpu=$(check_gpu_availability)

    echo
    log_info "Installation Plan for PyTorch Pipeline:"
    echo

    echo "PyTorch packages:"
    get_pytorch_commands "$CUDA_VERSION" "$has_gpu" | grep -E '^pip install' | sed 's/^/  /'

    echo
    echo "Export dependencies:"
    get_export_commands "$has_gpu" | grep -E '^pip install' | sed 's/^/  /'
    echo
}

# Create activation script
create_activation_script() {
    local env_name="$1"
    local env_path="$OUTPUT_DIR/$env_name"
    local scripts_dir="$OUTPUT_DIR/activation_scripts"

    mkdir -p "$scripts_dir"

    local script_path="$scripts_dir/activate_${env_name}.sh"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would create activation script at $script_path"
        return
    fi

    cat > "$script_path" << EOF
#!/bin/bash
# RT-DETR Environment Activation Script
# Generated on $(date)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "\${BLUE}[INFO]\${NC} Activating RT-DETR environment: $env_name"

# Activate environment
source "$env_path/bin/activate"

# Set environment variables
export RTDETR_ENV_NAME="$env_name"
export RTDETR_ENV_PATH="$env_path"

# Show environment info
echo -e "\${GREEN}[SUCCESS]\${NC} Environment activated!"
echo "  Environment: $env_name"
echo "  Python: \$(python --version)"
echo "  Location: $env_path"
echo

# Show pipeline info
echo "PyTorch Pipeline: PyTorch → ONNX → TensorRT"
echo "Supported models: RT-DETR v1, v2, v4, D-FINE, DEIM"
echo

echo "Usage:"
echo "  Export models: ./export.sh ..."
echo "  Deactivate: deactivate"
echo
EOF

    chmod +x "$script_path"

    log_success "Activation script created: $script_path"
}

# Create environment info file
create_env_info() {
    local env_name="$1"
    local env_path="$OUTPUT_DIR/$env_name"

    if [[ "$DRY_RUN" == true ]]; then
        return
    fi

    local info_file="$env_path/.rtdetr_env_info"

    cat > "$info_file" << EOF
# RT-DETR Environment Information
# Generated on $(date)

env_name=$env_name
python_version=$PYTHON_VERSION
cuda_version=$CUDA_VERSION
created_date=$(date -Iseconds)
extras_installed=$INSTALL_EXTRAS

# Pipeline
pipeline_type=pytorch_to_onnx_to_tensorrt

# Supported models
supported_models=v1,v2,v4,dfine,deim
EOF

    log_info "Environment info created: $info_file"
}

# Show final summary
show_summary() {
    local env_name="$1"
    local env_path="$OUTPUT_DIR/$env_name"
    local script_path="$OUTPUT_DIR/activation_scripts/activate_${env_name}.sh"

    echo
    log_success "Environment setup completed!"
    echo
    log_info "Environment Details:"
    echo "  Name: $env_name"
    echo "  Location: $env_path"
    echo "  Python: $PYTHON_VERSION"
    echo "  CUDA: $CUDA_VERSION"
    echo

    log_info "Activation Options:"
    echo "  Direct: source $env_path/bin/activate"
    echo "  Script: source $script_path"
    echo

    log_info "Next Steps:"
    echo "  1. Activate environment: source $script_path"
    echo "  2. Clone repository: ./clone_repo.sh --version [VERSION]"
    echo "  3. Export model: ./export.sh ..."
    echo "  Supported versions: v1, v2, v4, dfine, deim"
    echo
}

# Main execution
main() {
    log_info "RT-DETR Virtual Environment Setup Script"

    # Show help if no arguments
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi

    # Parse and validate arguments
    parse_arguments "$@"
    validate_arguments

    # Check dependencies
    check_dependencies

    log_info "Setup Configuration:"
    echo "  Environment: $ENV_NAME"
    echo "  Python: $PYTHON_VERSION"
    echo "  CUDA: $CUDA_VERSION"
    echo "  Manager: $ENV_MANAGER"
    echo "  Extras: $INSTALL_EXTRAS"
    echo

    if [[ "$DRY_RUN" == true ]]; then
        log_warning "DRY RUN MODE - No changes will be made"
        show_installation_plan
        exit 0
    fi

    # Create and setup environment
    create_environment "$ENV_NAME"
    install_packages "$ENV_NAME"
    create_activation_script "$ENV_NAME"
    create_env_info "$ENV_NAME"

    # Show summary
    show_summary "$ENV_NAME"
}

# Execute main function
main "$@"
