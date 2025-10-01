#!/bin/bash

# CRANTpy Documentation Build Script
# This script builds the documentation using Jupyter Book and deploys to GitHub Pages

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/crantpy" ]] || [[ ! -d "docs" ]]; then
        print_error "This script must be run from the project root directory"
        print_error "Expected to find: pyproject.toml, src/crantpy/, docs/"
        exit 1
    fi
    print_success "Project structure verified"
}

# Function to apply temporary fixes for dependency compatibility
apply_compatibility_fixes() {
    print_status "Applying compatibility fixes for documentation build..."
    
    # Set environment variables to help with import issues
    export SPHINX_BUILD_MODE=1
    export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
    
    # Temporarily downgrade scikit-learn if needed
    local current_sklearn=$(poetry run python -c "import sklearn; print(sklearn.__version__)" 2>/dev/null || echo "none")
    if [[ "$current_sklearn" == "1.6.1" ]]; then
        print_warning "Detected scikit-learn 1.6.1 which has compatibility issues"
        print_status "Installing compatible scikit-learn version for documentation build..."
        poetry add "scikit-learn>=1.5.0,<1.6.0" --quiet || print_warning "Could not downgrade scikit-learn"
    fi
    
    print_success "Compatibility fixes applied"
}

# Function to restore dependencies after build
restore_dependencies() {
    print_status "Restoring original dependencies..."
    poetry install --quiet || print_warning "Could not restore original dependencies"
    print_success "Dependencies restored"
}

# Function to clean previous builds
clean_build() {
    print_status "Cleaning previous build artifacts..."
    
    if [[ -d "docs/_build" ]]; then
        rm -rf docs/_build
        print_success "Removed docs/_build directory"
    fi
    
    if [[ -d "docs/api" ]]; then
        rm -rf docs/api
        print_success "Removed docs/api directory"
    fi
}

# Function to generate API documentation
generate_api_docs() {
    print_status "Generating API documentation with sphinx-apidoc..."
    
    # Create a temporary fix for pandas version compatibility
    export PYTHONPATH=".:$PYTHONPATH"
    
    # Create a temporary conf.py to help with imports
    cat > docs/temp_conf.py << 'EOF'
import sys
import os
sys.path.insert(0, os.path.abspath('../src'))

# Mock problematic imports
import unittest.mock as mock
sys.modules['pcg_skel'] = mock.MagicMock()
sys.modules['meshparty'] = mock.MagicMock()
sys.modules['sklearn'] = mock.MagicMock()
sys.modules['scikit-learn'] = mock.MagicMock()
sys.modules['trimesh'] = mock.MagicMock()
sys.modules['skeletor'] = mock.MagicMock()
EOF
    
    # Run sphinx-apidoc with the temporary config
    PYTHONPATH="docs:$PYTHONPATH" poetry run sphinx-apidoc -o docs/api/ src/crantpy --force --separate
    local apidoc_result=$?
    
    # Clean up temporary file
    rm -f docs/temp_conf.py
    
    if [[ $apidoc_result -ne 0 ]]; then
        print_error "Failed to generate API documentation"
        return 1
    fi
    
    print_success "API documentation generated successfully"
    return 0
}

# Function to build the Jupyter Book
build_book() {
    print_status "Building Jupyter Book documentation..."
    
    if ! poetry run jupyter-book build docs/ --verbose; then
        print_error "Failed to build Jupyter Book"
        return 1
    fi
    
    print_success "Jupyter Book built successfully"
    return 0
}

# Function to deploy to GitHub Pages
deploy_docs() {
    local deploy_flag="$1"
    
    if [[ "$deploy_flag" == "--deploy" ]] || [[ "$deploy_flag" == "-d" ]]; then
        print_status "Deploying documentation to GitHub Pages..."
        
        if ! poetry run ghp-import -n -p -f docs/_build/html; then
            print_error "Failed to deploy to GitHub Pages"
            exit 1
        fi
        
        print_success "Documentation deployed to GitHub Pages successfully!"
    else
        print_warning "Documentation built but not deployed."
        print_warning "To deploy, run: $0 --deploy"
        print_status "Local documentation available at: docs/_build/html/index.html"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --deploy, -d    Deploy to GitHub Pages after building"
    echo "  --clean, -c     Clean build artifacts before building"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              Build documentation only"
    echo "  $0 --clean     Clean and build documentation"
    echo "  $0 --deploy    Build and deploy documentation"
    echo "  $0 -c -d       Clean, build, and deploy documentation"
}

# Parse command line arguments
DEPLOY=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --deploy|-d)
            DEPLOY=true
            shift
            ;;
        --clean|-c)
            CLEAN=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting CRANTpy documentation build process..."
    
    # Pre-flight checks
    check_poetry
    check_directory
    
    # Apply compatibility fixes
    apply_compatibility_fixes
    
    # Optional cleaning
    if [[ "$CLEAN" == true ]]; then
        clean_build
    fi
    
    # Build process with error handling
    if generate_api_docs && build_book; then
        # Optional deployment
        if [[ "$DEPLOY" == true ]]; then
            deploy_docs "--deploy"
        else
            deploy_docs
        fi
        
        print_success "Documentation build process completed!"
    else
        print_error "Documentation build failed!"
        restore_dependencies
        exit 1
    fi
    
    # Restore original dependencies
    restore_dependencies
}

# Run main function
main
