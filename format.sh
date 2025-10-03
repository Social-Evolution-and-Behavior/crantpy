#!/bin/bash

# Quick Format Script for CRANTpy
# Runs formatting and linting manually without pre-commit

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info "Running Black formatter..."
poetry run black src/ tests/

print_info "Running Ruff linter with auto-fix..."
poetry run ruff check src/ tests/ --fix

print_info "Running Ruff formatter..."
poetry run ruff format src/ tests/

print_success "Code formatting and linting complete!"
print_info "Your code is now properly formatted and linted."
