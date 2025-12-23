#!/bin/bash

# RT-DETR Repository Cloning Script
# 
# This script clones the appropriate RT-DETR repository based on version
# for export pipelines.
#
# Usage:
#   cd vision-core
#   export/detection/rtdetr/clone_repo.sh --version v4 --output-dir ./3rdparty/repositories
#   export/detection/rtdetr/clone_repo.sh --help for more information.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Default values
VERSION=""
OUTPUT_DIR="./repositories"
FORCE_CLONE=false
SHALLOW_CLONE=true
BRANCH=""

# Repository configurations
declare -A REPO_URLS=(
    ["v1"]="https://github.com/lyuwenyu/RT-DETR.git"
    ["v2"]="https://github.com/lyuwenyu/RT-DETR.git"
    ["v4"]="https://github.com/RT-DETRs/RT-DETRv4.git"
    ["dfine"]="https://github.com/Peterande/D-FINE.git"
    ["deim"]="https://github.com/Intellindust-AI-Lab/DEIM.git"
)

declare -A REPO_NAMES=(
    ["v1"]="RT-DETR"
    ["v2"]="RT-DETR"
    ["v4"]="RT-DETRv4"
    ["dfine"]="D-FINE"
    ["deim"]="DEIM"
)


# Help function
show_help() {
    cat << EOF
RT-DETR Repository Cloning Script

DESCRIPTION:
    Clone RT-DETR repositories for export pipelines.
    Supports PyTorch-based versions (v1, v2, v4, dfine, deim).

USAGE:
    $0 [OPTIONS]

REQUIRED OPTIONS:
    -v, --version VERSION   RT-DETR version to clone: v1, v2, v4, dfine, deim

OPTIONAL OPTIONS:
    -o, --output-dir PATH   Output directory for repositories (default: ./repositories)
    -f, --force            Force clone even if directory exists
    --full-clone           Clone full repository (default: shallow clone)
    -b, --branch BRANCH    Specific branch to clone (default: main/master)
    -h, --help             Show this help message

SUPPORTED VERSIONS:
    v1    - RT-DETR v1 (PyTorch)     - github.com/lyuwenyu/RT-DETR
    v2    - RT-DETR v2 (PyTorch)     - github.com/lyuwenyu/RT-DETR
    v4    - RT-DETR v4 (PyTorch)     - github.com/RT-DETRs/RT-DETRv4
    dfine - D-FINE (PyTorch)         - github.com/Peterande/D-FINE
    deim  - DEIM (PyTorch)           - github.com/Intellindust-AI-Lab/DEIM

EXAMPLES:
    # Clone RT-DETRv4 to default location
    $0 --version v4

    # Clone RT-DETR to custom directory
    $0 --version v2 --output-dir /workspace/models

    # Force clone with full history
    $0 --version v2 --force --full-clone

    # Clone specific branch
    $0 --version v4 --branch develop

OUTPUT STRUCTURE:
    The script creates organized directories:
    
    repositories/
    ├── RT-DETR/       # v1, v2
    ├── RT-DETRv4/     # v4
    ├── D-FINE/        # dfine
    └── DEIM/          # deim

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_CLONE=true
                shift
                ;;
            --full-clone)
                SHALLOW_CLONE=false
                shift
                ;;
            -b|--branch)
                BRANCH="$2"
                shift 2
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
    if [[ -z "$VERSION" ]]; then
        log_error "Version is required. Use -v/--version to specify."
        echo "Supported versions: v1, v2, v4, dfine, deim"
        exit 1
    fi

    if [[ -z "${REPO_URLS[$VERSION]}" ]]; then
        log_error "Unsupported version: $VERSION"
        echo "Supported versions: v1, v2, v4, dfine, deim"
        exit 1
    fi
}

# Check if git is available
check_git() {
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed. Please install git first."
        exit 1
    fi
}

# Clone repository
clone_repository() {
    local version="$1"
    local repo_url="${REPO_URLS[$version]}"
    local repo_name="${REPO_NAMES[$version]}"
    
    # Create target directory directly in output directory
    local target_dir="$OUTPUT_DIR/$repo_name"
    
    log_info "Repository: $repo_url"
    log_info "Target directory: $target_dir"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Check if directory already exists
    if [[ -d "$target_dir" ]]; then
        if [[ "$FORCE_CLONE" == true ]]; then
            log_warning "Directory exists, removing: $target_dir"
            rm -rf "$target_dir"
        else
            log_error "Directory already exists: $target_dir"
            log_info "Use --force to overwrite or choose a different output directory"
            exit 1
        fi
    fi
    
    # Build git clone command
    local clone_cmd=("git" "clone")
    
    if [[ "$SHALLOW_CLONE" == true ]]; then
        clone_cmd+=(--depth 1)
        log_info "Performing shallow clone (--depth 1)"
    fi
    
    if [[ -n "$BRANCH" ]]; then
        clone_cmd+=(--branch "$BRANCH")
        log_info "Cloning branch: $BRANCH"
    fi
    
    clone_cmd+=("$repo_url" "$target_dir")
    
    log_info "Cloning repository..."
    log_info "Command: ${clone_cmd[*]}"
    
    if "${clone_cmd[@]}"; then
        log_success "Repository cloned successfully!"
        log_info "Location: $target_dir"
        
        # Show repository information
        echo
        log_info "Repository Information:"
        echo "  Version: $version"
        echo "  Framework: PyTorch"
        echo "  Name: $repo_name"
        echo "  URL: $repo_url"
        echo "  Path: $target_dir"
        
        # Show directory contents
        if [[ -d "$target_dir" ]]; then
            echo
            log_info "Repository contents:"
            ls -la "$target_dir" | head -10
            if [[ $(ls -1 "$target_dir" | wc -l) -gt 8 ]]; then
                echo "  ... (showing first 8 items)"
            fi
        fi
        
        # Create info file
        create_info_file "$target_dir" "$version" "pytorch" "$repo_url"
        
    else
        log_error "Failed to clone repository"
        exit 1
    fi
}

# Create info file with repository metadata
create_info_file() {
    local target_dir="$1"
    local version="$2"
    local framework="$3"
    local repo_url="$4"
    
    local info_file="$target_dir/.rtdetr_info"
    
    cat > "$info_file" << EOF
# RT-DETR Repository Information
# Generated by clone_repo.sh on $(date)

version=$version
framework=$framework
repository_url=$repo_url
clone_date=$(date -Iseconds)
clone_type=$(if [[ "$SHALLOW_CLONE" == true ]]; then echo "shallow"; else echo "full"; fi)
branch=${BRANCH:-"default"}
EOF

    log_info "Created repository info file: $info_file"
}

# List existing repositories
list_repositories() {
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        log_warning "Output directory does not exist: $OUTPUT_DIR"
        return
    fi
    
    log_info "Existing repositories in $OUTPUT_DIR:"
    echo
    
    for repo_dir in "$OUTPUT_DIR"/*; do
        if [[ -d "$repo_dir" ]]; then
            local repo_name=$(basename "$repo_dir")
            local info_file="$repo_dir/.rtdetr_info"
            if [[ -f "$info_file" ]]; then
                local version=$(grep "^version=" "$info_file" | cut -d'=' -f2)
                echo "  ├── $repo_name (version: $version)"
            else
                echo "  ├── $repo_name"
            fi
        fi
    done
    echo
}

# Main execution
main() {
    log_info "RT-DETR Repository Cloning Script"
    
    # Show help if no arguments
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    # Parse and validate arguments
    parse_arguments "$@"
    validate_arguments
    
    # Check dependencies
    check_git
    
    # Show current repositories
    list_repositories
    
    # Clone repository
    clone_repository "$VERSION"
    
    # Show final structure
    echo
    list_repositories
    
    log_success "Repository setup complete!"
    echo
    log_info "Next steps:"
    echo "  1. cd vision-core && export/detection/rtdetr/setup_env.sh"
    echo "  2. Run export: export/detection/rtdetr/export.sh --repo-dir $OUTPUT_DIR/${REPO_NAMES[$VERSION]}"
}

# Execute main function
main "$@"