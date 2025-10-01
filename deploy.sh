#!/bin/bash

# CRANTpy PyPI Deployment Script
# This script builds and deploys CRANTpy to PyPI using Poetry

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
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/crantpy" ]]; then
        print_error "This script must be run from the project root directory"
        print_error "Expected to find: pyproject.toml, src/crantpy/"
        exit 1
    fi
    print_success "Project structure verified"
}

# Function to check git status
check_git_status() {
    if [[ -n $(git status --porcelain) ]]; then
        print_warning "There are uncommitted changes in your repository"
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Deployment cancelled"
            exit 1
        fi
    fi
    print_success "Git status checked"
}

# Function to get current version
get_current_version() {
    local version=$(poetry version --short)
    echo "$version"
}

# Function to bump version
bump_version() {
    local bump_type="$1"
    
    print_status "Current version: $(get_current_version)"
    
    case $bump_type in
        major|minor|patch)
            poetry version "$bump_type"
            ;;
        *)
            print_error "Invalid version bump type. Use: major, minor, or patch"
            exit 1
            ;;
    esac
    
    local new_version=$(get_current_version)
    print_success "Version bumped to: $new_version"
    return 0
}

# Function to update lazy imports
update_imports() {
    print_status "Updating lazy imports with mkinit..."
    
    if ! poetry run mkinit --lazy_loader src/crantpy --recursive --inplace; then
        print_error "Failed to update lazy imports"
        exit 1
    fi
    
    print_success "Lazy imports updated successfully"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    if ! poetry run pytest tests/ -v; then
        print_error "Tests failed"
        exit 1
    fi
    
    print_success "All tests passed"
}

# Function to check code quality
check_code_quality() {
    print_status "Checking code quality..."
    
    # Format code
    print_status "Formatting code with Black..."
    poetry run black src/ tests/
    
    # Check linting (fix what can be fixed automatically)
    print_status "Checking and fixing code with Ruff..."
    poetry run ruff check src/ tests/ --fix || true
    
    # Check remaining issues (non-blocking for now due to mkinit generated code)
    print_status "Final code quality check..."
    if poetry run ruff check src/ tests/ --quiet; then
        print_success "All code quality checks passed"
    else
        print_warning "Some code quality issues remain (mostly from auto-generated code)"
        print_warning "Continuing with deployment..."
    fi
}

# Function to build documentation
build_docs() {
    print_status "Building documentation..."
    
    if ! ./build_docs.sh --clean; then
        print_error "Documentation build failed"
        exit 1
    fi
    
    print_success "Documentation built successfully"
}

# Function to build package
build_package() {
    print_status "Building package with Poetry..."
    
    # Clean previous builds
    rm -rf dist/
    
    if ! poetry build; then
        print_error "Package build failed"
        exit 1
    fi
    
    print_success "Package built successfully"
    
    # Show build artifacts
    print_status "Build artifacts:"
    ls -la dist/
}

# Function to check package
check_package() {
    print_status "Checking package with twine..."
    
    if ! poetry run twine check dist/*; then
        print_error "Package check failed"
        exit 1
    fi
    
    print_success "Package check passed"
}

# Function to deploy to TestPyPI
deploy_test() {
    print_status "Deploying to TestPyPI..."
    
    if ! poetry config repositories.testpypi https://test.pypi.org/legacy/; then
        print_error "Failed to configure TestPyPI repository"
        exit 1
    fi
    
    if ! poetry publish -r testpypi; then
        print_error "Deployment to TestPyPI failed"
        exit 1
    fi
    
    local version=$(get_current_version)
    print_success "Successfully deployed to TestPyPI"
    print_status "Test installation: pip install -i https://test.pypi.org/simple/ crantpy==$version"
}

# Function to deploy to PyPI
deploy_pypi() {
    print_status "Deploying to PyPI..."
    
    print_warning "This will deploy to the LIVE PyPI repository!"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Deployment cancelled"
        exit 1
    fi
    
    if ! poetry publish; then
        print_error "Deployment to PyPI failed"
        exit 1
    fi
    
    local version=$(get_current_version)
    print_success "Successfully deployed to PyPI"
    print_status "Installation: pip install crantpy==$version"
}

# Function to create git tag
create_git_tag() {
    local version=$(get_current_version)
    local tag="v$version"
    
    print_status "Creating git tag: $tag"
    
    git add .
    git commit -m "chore: bump version to $version" || true
    git tag -a "$tag" -m "Release version $version"
    
    print_success "Git tag created: $tag"
    
    read -p "Push tag to remote? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push origin main
        git push origin "$tag"
        print_success "Tag pushed to remote"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] <COMMAND>"
    echo ""
    echo "Commands:"
    echo "  test-deploy     Deploy to TestPyPI (recommended first)"
    echo "  deploy          Deploy to PyPI (production)"
    echo "  build-only      Build package without deploying"
    echo ""
    echo "Options:"
    echo "  --bump LEVEL    Bump version (major|minor|patch)"
    echo "  --skip-tests    Skip running tests"
    echo "  --skip-docs     Skip building documentation"
    echo "  --skip-checks   Skip code quality checks"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --bump patch test-deploy    # Bump patch version and deploy to TestPyPI"
    echo "  $0 --bump minor deploy         # Bump minor version and deploy to PyPI"
    echo "  $0 build-only                  # Just build the package"
}

# Parse command line arguments
BUMP_VERSION=""
SKIP_TESTS=false
SKIP_DOCS=false
SKIP_CHECKS=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --bump)
            BUMP_VERSION="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-docs)
            SKIP_DOCS=true
            shift
            ;;
        --skip-checks)
            SKIP_CHECKS=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        test-deploy|deploy|build-only)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate command
if [[ -z "$COMMAND" ]]; then
    print_error "No command specified"
    show_usage
    exit 1
fi

# Main execution
main() {
    print_status "Starting CRANTpy deployment process..."
    
    # Pre-flight checks
    check_poetry
    check_directory
    check_git_status
    
    # Bump version if requested
    if [[ -n "$BUMP_VERSION" ]]; then
        bump_version "$BUMP_VERSION"
    fi
    
    # Update lazy imports (always required)
    update_imports
    
    # Optional quality checks
    if [[ "$SKIP_CHECKS" == false ]]; then
        check_code_quality
    fi
    
    # Optional tests
    if [[ "$SKIP_TESTS" == false ]]; then
        run_tests
    fi
    
    # Optional documentation build
    if [[ "$SKIP_DOCS" == false ]]; then
        build_docs
    fi
    
    # Build package
    build_package
    check_package
    
    # Deploy based on command
    case $COMMAND in
        test-deploy)
            deploy_test
            ;;
        deploy)
            deploy_pypi
            create_git_tag
            ;;
        build-only)
            print_success "Package built successfully. Files in dist/"
            ;;
    esac
    
    print_success "Deployment process completed!"
}

# Run main function
main