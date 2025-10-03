#!/bin/bash

# CRANTpy Development Setup Script
# This script sets up pre-commit hooks and development tools for CRANTpy

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
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

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Function to check if poetry is available
check_poetry() {
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is not installed or not in PATH"
        print_error "Please install poetry: https://python-poetry.org/docs/#installation"
        exit 1
    fi
    print_success "Poetry found"
}

# Function to check if we're in the right directory
check_directory() {
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/crantpy" ]]; then
        print_error "This script must be run from the project root directory"
        print_error "Expected to find: pyproject.toml, src/crantpy/"
        exit 1
    fi
    print_success "Project structure verified"
}

# Function to check if pre-commit should be enabled
check_pre_commit_readiness() {
    echo ""
    print_warning "🚨 PRE-COMMIT HOOKS FORMATTING WARNING 🚨"
    echo ""
    echo "Pre-commit hooks will reformat ALL files in the repository."
    echo "This can cause merge conflicts with existing branches."
    echo ""
    print_info "Current recommendation:"
    echo "  - Wait until all active branches are merged to main"
    echo "  - Then enable pre-commit hooks project-wide"
    echo "  - This ensures consistent formatting for all future work"
    echo ""

    if [[ "${1:-}" == "--force" ]]; then
        print_warning "Force flag detected - installing pre-commit hooks anyway"
        return 0
    fi

    read -p "Do you still want to install pre-commit hooks? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Pre-commit hooks NOT installed (recommended for now)"
        print_info "You can install them later with: ./setup_dev.sh --install --force"
        print_info "Or wait for the project-wide migration"
        return 1
    fi

    print_warning "⚠️  You chose to install pre-commit hooks!"
    print_warning "   This may cause merge conflicts with existing branches."
    read -p "Are you absolutely sure? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Pre-commit hooks NOT installed"
        return 1
    fi

    return 0
}

# Function to install pre-commit
install_precommit() {
    print_step "Installing pre-commit..."

    if ! poetry run pre-commit --version &> /dev/null; then
        print_info "Installing pre-commit with Poetry..."
        poetry add --group dev pre-commit
    else
        print_info "Pre-commit already installed"
    fi

    print_success "Pre-commit installed"
}

# Function to install pre-commit hooks
install_hooks() {
    print_step "Installing pre-commit hooks..."

    if ! poetry run pre-commit install; then
        print_error "Failed to install pre-commit hooks"
        exit 1
    fi

    print_success "Pre-commit hooks installed"
}

# Function to run initial hook check
run_initial_check() {
    print_step "Running initial pre-commit check on all files..."

    print_warning "This may take a while on the first run as it downloads hook repositories..."

    if poetry run pre-commit run --all-files; then
        print_success "All pre-commit hooks passed!"
    else
        print_warning "Some pre-commit hooks failed or made changes"
        print_info "Files have been automatically formatted/fixed where possible"
        print_info "Please review the changes and commit them"
    fi
}

# Function to show usage
show_usage() {
    echo "CRANTpy Development Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --install       Install pre-commit and hooks (with safety prompts)"
    echo "  --install --force   Force install pre-commit hooks (skip safety prompts)"
    echo "  --run           Run pre-commit on all files"
    echo "  --check         Check if pre-commit is set up correctly"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --install          # Set up pre-commit hooks (with safety prompts)"
    echo "  $0 --install --force  # Force install (skip warnings)"
    echo "  $0 --run             # Run pre-commit on all files"
    echo "  $0 --check           # Check setup status"
    echo ""
    echo "⚠️  WARNING: Pre-commit hooks reformat ALL files!"
    echo "   This may cause merge conflicts with existing branches."
    echo "   Consider waiting for project-wide migration."
}

# Function to check setup status
check_setup() {
    print_step "Checking pre-commit setup status..."

    # Check if pre-commit is installed
    if poetry run pre-commit --version &> /dev/null; then
        local version=$(poetry run pre-commit --version)
        print_success "Pre-commit installed: $version"
    else
        print_error "Pre-commit not installed"
        return 1
    fi

    # Check if hooks are installed
    if [[ -f ".git/hooks/pre-commit" ]]; then
        print_success "Pre-commit hooks are installed"
    else
        print_error "Pre-commit hooks not installed"
        print_info "Run: $0 --install"
        return 1
    fi

    # Check configuration file
    if [[ -f ".pre-commit-config.yaml" ]]; then
        print_success "Pre-commit configuration found"
    else
        print_error "Pre-commit configuration missing"
        return 1
    fi

    print_success "Pre-commit setup is complete!"
}

# Function to setup everything
setup_all() {
    print_info "Setting up CRANTpy development environment..."
    echo ""

    check_poetry
    check_directory

    # Check if user wants to install pre-commit hooks
    if ! check_pre_commit_readiness "$@"; then
        echo ""
        print_success "Development environment check complete!"
        print_info "Pre-commit hooks were NOT installed (by choice)"
        echo ""
        print_info "To install later:"
        echo "  • Force install: ./setup_dev.sh --install --force"
        echo "  • Wait for project-wide migration (recommended)"
        return 0
    fi

    install_precommit
    install_hooks

    # Ask about running on all files
    echo ""
    print_warning "Do you want to run pre-commit on all files now?"
    print_warning "This will reformat ALL files and may cause merge conflicts!"
    read -p "Run pre-commit on all files? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_initial_check
    else
        print_info "Skipping initial pre-commit run"
        print_info "Pre-commit will run automatically on your next commit"
    fi

    echo ""
    print_success "Development environment setup complete!"
    echo ""
    print_info "What happens now:"
    echo "  • Code will be automatically formatted and linted before each commit"
    echo "  • You can manually run checks with: poetry run pre-commit run --all-files"
    echo "  • To skip hooks for a specific commit: git commit --no-verify"
    echo ""
    print_warning "Note: The first time you commit, it may take longer as hooks download dependencies"
}

# Function to run pre-commit manually
run_precommit() {
    print_step "Running pre-commit on all files..."

    if ! poetry run pre-commit run --all-files; then
        print_warning "Some hooks failed or made changes"
        print_info "Please review the changes"
        exit 1
    fi

    print_success "All pre-commit checks passed!"
}

# Parse command line arguments
case "${1:-}" in
    --install)
        if [[ "${2:-}" == "--force" ]]; then
            setup_all --force
        else
            setup_all
        fi
        ;;
    --run)
        run_precommit
        ;;
    --check)
        check_setup
        ;;
    --help|-h)
        show_usage
        ;;
    "")
        print_error "No option specified"
        show_usage
        exit 1
        ;;
    *)
        print_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac
