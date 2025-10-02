#!/bin/bash

# CRANTpy API Token Setup Script
# This script helps developers configure PyPI and TestPyPI tokens for package deployment

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

# Function to detect shell
detect_shell() {
    # Check $SHELL environment variable first (most reliable for user's login shell)
    if [[ "$SHELL" == *"zsh" ]]; then
        echo "zsh"
    elif [[ "$SHELL" == *"bash" ]]; then
        echo "bash"
    # Then check environment variables for current shell
    elif [[ -n "$ZSH_VERSION" ]]; then
        echo "zsh"
    elif [[ -n "$BASH_VERSION" ]]; then
        echo "bash"
    # Check what shell is running this script
    elif [[ "$0" == *"zsh" ]] || [[ -n "$ZSH_NAME" ]]; then
        echo "zsh"
    elif [[ "$0" == *"bash" ]]; then
        echo "bash"
    # Fallback: detect by OS
    elif [[ "$(uname)" == "Darwin" ]]; then
        # macOS defaults to zsh since Catalina
        echo "zsh"
    else
        # Most Linux distros use bash by default
        echo "bash"
    fi
}

# Function to get shell config file
get_shell_config() {
    local shell_type="$1"
    local os_type="$(uname)"
    
    case $shell_type in
        zsh)
            # Check for existing zsh config files in order of preference
            if [[ -f "$HOME/.zshrc" ]]; then
                echo "$HOME/.zshrc"
            elif [[ -f "$HOME/.zshenv" ]]; then
                echo "$HOME/.zshenv"
            else
                echo "$HOME/.zshrc"  # Will be created
            fi
            ;;
        bash)
            # Different behavior for macOS vs Linux
            if [[ "$os_type" == "Darwin" ]]; then
                # macOS: Terminal.app sources .bash_profile by default
                if [[ -f "$HOME/.bash_profile" ]]; then
                    echo "$HOME/.bash_profile"
                elif [[ -f "$HOME/.bashrc" ]]; then
                    echo "$HOME/.bashrc"
                else
                    echo "$HOME/.bash_profile"  # Will be created
                fi
            else
                # Linux: Usually .bashrc
                if [[ -f "$HOME/.bashrc" ]]; then
                    echo "$HOME/.bashrc"
                elif [[ -f "$HOME/.bash_profile" ]]; then
                    echo "$HOME/.bash_profile"
                else
                    echo "$HOME/.bashrc"  # Will be created
                fi
            fi
            ;;
        *)
            # Fallback to generic profile
            if [[ -f "$HOME/.profile" ]]; then
                echo "$HOME/.profile"
            else
                echo "$HOME/.profile"  # Will be created
            fi
            ;;
    esac
}

# Function to show system and shell information
show_system_info() {
    local shell_type="$1"
    local config_file="$2"
    local os_type="$(uname)"
    local os_name
    
    case "$os_type" in
        Darwin)
            os_name="macOS"
            ;;
        Linux)
            os_name="Linux"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            os_name="Windows"
            ;;
        *)
            os_name="$os_type"
            ;;
    esac
    
    print_info "System: $os_name ($os_type)"
    print_info "Detected shell: $shell_type"
    print_info "Configuration file: $config_file"
    
    # Additional info for macOS users
    if [[ "$os_type" == "Darwin" ]] && [[ "$shell_type" == "bash" ]]; then
        print_warning "Note: macOS Terminal.app sources ~/.bash_profile by default"
        print_warning "If you use iTerm2 or other terminals, they might source ~/.bashrc"
    fi
}

# Function to show instructions for getting tokens
show_token_instructions() {
    echo ""
    print_step "How to get your API tokens:"
    echo ""
    echo "1. ${CYAN}TestPyPI Token:${NC}"
    echo "   - Go to: https://test.pypi.org/manage/account/token/"
    echo "   - Click 'Add API token'"
    echo "   - Token name: 'crantpy-dev' (or any name you prefer)"
    echo "   - Scope: 'Entire account' (or limit to specific project)"
    echo "   - Copy the generated token (starts with 'pypi-')"
    echo ""
    echo "2. ${CYAN}PyPI Token:${NC}"
    echo "   - Go to: https://pypi.org/manage/account/token/"
    echo "   - Click 'Add API token'"
    echo "   - Token name: 'crantpy-dev' (or any name you prefer)"
    echo "   - Scope: 'Entire account' (or limit to specific project)"
    echo "   - Copy the generated token (starts with 'pypi-')"
    echo ""
    print_warning "Keep these tokens secure and never share them!"
    echo ""
}

# Function to validate token format
validate_token() {
    local token="$1"
    if [[ $token =~ ^pypi-[A-Za-z0-9_-]+$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to setup environment variables
setup_environment_variables() {
    local shell_type="$1"
    local config_file="$2"
    local testpypi_token="$3"
    local pypi_token="$4"
    
    print_step "Setting up environment variables in $config_file"
    
    # Create config file if it doesn't exist
    if [[ ! -f "$config_file" ]]; then
        touch "$config_file"
        print_info "Created new configuration file: $config_file"
    else
        # Backup existing config
        cp "$config_file" "$config_file.backup.$(date +%Y%m%d_%H%M%S)"
        print_info "Backed up existing config to $config_file.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Remove any existing CRANTpy token entries
    if [[ -f "$config_file" ]] && grep -q "CRANTpy PyPI Tokens" "$config_file"; then
        if [[ "$(uname)" == "Darwin" ]]; then
            # macOS sed syntax
            sed -i '' '/# CRANTpy PyPI Tokens/,/# End CRANTpy PyPI Tokens/d' "$config_file"
        else
            # Linux sed syntax
            sed -i '/# CRANTpy PyPI Tokens/,/# End CRANTpy PyPI Tokens/d' "$config_file"
        fi
        print_info "Removed existing CRANTpy token configuration"
    fi
    
    # Add new token entries
    cat >> "$config_file" << EOF

# CRANTpy PyPI Tokens
export POETRY_PYPI_TOKEN_TESTPYPI="$testpypi_token"
export POETRY_PYPI_TOKEN_PYPI="$pypi_token"
# End CRANTpy PyPI Tokens
EOF
    
    print_success "Environment variables added to $config_file"
    
    # Special handling for macOS bash users
    if [[ "$(uname)" == "Darwin" ]] && [[ "$shell_type" == "bash" ]] && [[ "$config_file" == *".bash_profile" ]]; then
        # Check if .bashrc exists and is sourced
        if [[ -f "$HOME/.bashrc" ]] && ! grep -q "source.*\.bashrc" "$config_file"; then
            print_info "Consider adding 'source ~/.bashrc' to $config_file if you use .bashrc"
        fi
    fi
}

# Function to setup Poetry config
setup_poetry_config() {
    local testpypi_token="$1"
    local pypi_token="$2"
    
    print_step "Configuring Poetry with API tokens"
    
    # Check if poetry is available
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is not installed or not in PATH"
        print_error "Please install poetry: https://python-poetry.org/docs/#installation"
        return 1
    fi
    
    # Configure repositories
    poetry config repositories.testpypi https://test.pypi.org/legacy/
    print_info "Configured TestPyPI repository"
    
    # Configure tokens
    poetry config pypi-token.testpypi "$testpypi_token"
    poetry config pypi-token.pypi "$pypi_token"
    
    print_success "Poetry configured with API tokens"
}

# Function to test configuration
test_configuration() {
    print_step "Testing configuration"
    
    # Test environment variables
    if [[ -n "$POETRY_PYPI_TOKEN_TESTPYPI" ]] && [[ -n "$POETRY_PYPI_TOKEN_PYPI" ]]; then
        print_success "Environment variables are set"
    else
        print_warning "Environment variables not set in current session"
        print_info "Please restart your terminal or run: source $(get_shell_config $(detect_shell))"
    fi
    
    # Test Poetry configuration
    if command -v poetry &> /dev/null; then
        if poetry config pypi-token.testpypi &> /dev/null && poetry config pypi-token.pypi &> /dev/null; then
            print_success "Poetry tokens configured"
        else
            print_warning "Poetry tokens not found"
        fi
        
        # Test repository configuration
        if poetry config repositories.testpypi &> /dev/null; then
            print_success "TestPyPI repository configured"
        else
            print_warning "TestPyPI repository not configured"
        fi
    else
        print_warning "Poetry not found, skipping Poetry configuration test"
    fi
}

# Function to show next steps
show_next_steps() {
    local shell_type=$(detect_shell)
    local config_file=$(get_shell_config "$shell_type")
    
    echo ""
    print_success "API token setup completed!"
    echo ""
    print_step "Next steps:"
    echo ""
    echo "1. ${CYAN}Restart your terminal${NC} or run:"
    echo "   source $config_file"
    echo ""
    echo "2. ${CYAN}Verify the setup${NC} by running:"
    echo "   ./api_token.sh --test"
    echo ""
    echo "3. ${CYAN}Test deployment${NC} by running:"
    echo "   ./deploy.sh --bump patch test-deploy"
    echo ""
    echo "4. ${CYAN}For production deployment${NC}:"
    echo "   ./deploy.sh --bump patch deploy"
    echo ""
    
    # OS-specific additional notes
    if [[ "$(uname)" == "Darwin" ]]; then
        print_info "macOS Note: If you use multiple terminal apps (Terminal.app, iTerm2, etc.),"
        print_info "make sure they all use the same configuration file."
    fi
    
    print_warning "Remember: Keep your tokens secure and never commit them to git!"
}

# Function to show current status
show_status() {
    print_step "Current token configuration status:"
    echo ""
    
    # Check environment variables
    if [[ -n "$POETRY_PYPI_TOKEN_TESTPYPI" ]]; then
        print_success "TestPyPI environment variable: Set"
    else
        print_error "TestPyPI environment variable: Not set"
    fi
    
    if [[ -n "$POETRY_PYPI_TOKEN_PYPI" ]]; then
        print_success "PyPI environment variable: Set"
    else
        print_error "PyPI environment variable: Not set"
    fi
    
    # Check Poetry configuration
    if command -v poetry &> /dev/null; then
        if poetry config pypi-token.testpypi &> /dev/null; then
            print_success "Poetry TestPyPI token: Configured"
        else
            print_error "Poetry TestPyPI token: Not configured"
        fi
        
        if poetry config pypi-token.pypi &> /dev/null; then
            print_success "Poetry PyPI token: Configured"
        else
            print_error "Poetry PyPI token: Not configured"
        fi
        
        if poetry config repositories.testpypi &> /dev/null; then
            print_success "TestPyPI repository: Configured"
        else
            print_error "TestPyPI repository: Not configured"
        fi
    else
        print_error "Poetry: Not installed or not in PATH"
    fi
    
    # Check shell configuration
    local shell_type=$(detect_shell)
    local config_file=$(get_shell_config "$shell_type")
    
    if [[ -f "$config_file" ]] && grep -q "POETRY_PYPI_TOKEN_TESTPYPI" "$config_file"; then
        print_success "Shell configuration ($config_file): CRANTpy tokens found"
    else
        print_error "Shell configuration ($config_file): CRANTpy tokens not found"
    fi
}

# Function to clean up configuration
cleanup_configuration() {
    print_step "Cleaning up CRANTpy token configuration"
    
    # Remove from Poetry
    if command -v poetry &> /dev/null; then
        poetry config --unset pypi-token.testpypi 2>/dev/null || true
        poetry config --unset pypi-token.pypi 2>/dev/null || true
        poetry config --unset repositories.testpypi 2>/dev/null || true
        print_info "Removed tokens from Poetry configuration"
    fi
    
    # Remove from shell configuration
    local shell_type=$(detect_shell)
    local config_file=$(get_shell_config "$shell_type")
    
    if [[ -f "$config_file" ]] && grep -q "CRANTpy PyPI Tokens" "$config_file"; then
        if [[ "$(uname)" == "Darwin" ]]; then
            # macOS sed syntax
            sed -i '' '/# CRANTpy PyPI Tokens/,/# End CRANTpy PyPI Tokens/d' "$config_file"
        else
            # Linux sed syntax
            sed -i '/# CRANTpy PyPI Tokens/,/# End CRANTpy PyPI Tokens/d' "$config_file"
        fi
        print_info "Removed tokens from $config_file"
    fi
    
    print_success "Configuration cleanup completed"
    print_warning "Please restart your terminal for changes to take effect"
}

# Function to show usage
show_usage() {
    echo "CRANTpy API Token Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --setup       Interactive setup of PyPI tokens"
    echo "  --test        Test current token configuration"
    echo "  --status      Show current configuration status"
    echo "  --cleanup     Remove all CRANTpy token configuration"
    echo "  --help, -h    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --setup     # Interactive token setup"
    echo "  $0 --test      # Test if tokens are working"
    echo "  $0 --status    # Check configuration status"
    echo "  $0 --cleanup   # Remove all token configuration"
}

# Main setup function
main_setup() {
    echo ""
    echo "🔐 CRANTpy API Token Setup"
    echo "=========================="
    
    # Detect shell and get config file
    local shell_type=$(detect_shell)
    local config_file=$(get_shell_config "$shell_type")
    
    # Show system information
    show_system_info "$shell_type" "$config_file"
    echo ""
    
    # Show instructions
    show_token_instructions
    
    # Get TestPyPI token
    echo ""
    print_step "Enter your TestPyPI API token:"
    echo -n "TestPyPI token: "
    read -s testpypi_token
    echo ""
    
    if [[ -z "$testpypi_token" ]]; then
        print_error "TestPyPI token cannot be empty"
        exit 1
    fi
    
    if ! validate_token "$testpypi_token"; then
        print_error "Invalid TestPyPI token format (should start with 'pypi-')"
        exit 1
    fi
    
    # Get PyPI token
    print_step "Enter your PyPI API token:"
    echo -n "PyPI token: "
    read -s pypi_token
    echo ""
    
    if [[ -z "$pypi_token" ]]; then
        print_error "PyPI token cannot be empty"
        exit 1
    fi
    
    if ! validate_token "$pypi_token"; then
        print_error "Invalid PyPI token format (should start with 'pypi-')"
        exit 1
    fi
    
    # Confirm setup
    echo ""
    print_warning "This will configure tokens for:"
    echo "  - Shell configuration: $config_file"
    echo "  - Poetry configuration"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Setup cancelled"
        exit 1
    fi
    
    # Setup environment variables
    setup_environment_variables "$shell_type" "$config_file" "$testpypi_token" "$pypi_token"
    
    # Setup Poetry
    setup_poetry_config "$testpypi_token" "$pypi_token"
    
    # Test configuration
    test_configuration
    
    # Show next steps
    show_next_steps
}

# Parse command line arguments
case "${1:-}" in
    --setup)
        main_setup
        ;;
    --test)
        test_configuration
        ;;
    --status)
        show_status
        ;;
    --cleanup)
        cleanup_configuration
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