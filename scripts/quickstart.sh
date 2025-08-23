#!/bin/bash

# Infinite_Money Quick Start Script
# This script sets up and runs the autonomous trading system

set -e

echo "🚀 Infinite_Money: Autonomous Quantum Trading System"
echo "=================================================="

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

# Check if Python 3.10+ is installed
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.10+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found"
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p logs data cache configs
    print_success "Directories created"
}

# Run smoke test
run_smoke_test() {
    print_status "Running smoke test..."
    python scripts/smoke_test.py
    if [ $? -eq 0 ]; then
        print_success "Smoke test passed"
    else
        print_error "Smoke test failed"
        exit 1
    fi
}

# Run the system
run_system() {
    print_status "Starting Infinite_Money system..."
    print_warning "This will run in autonomous mode with $1 initial capital"
    echo ""
    echo "Press Ctrl+C to stop the system"
    echo ""
    
    python -m infinite_money.main \
        --mode=autonomous \
        --capital=$1 \
        --duration=1 \
        --risk-limit=0.02 \
        --target-sharpe=1.5
}

# Main execution
main() {
    # Default capital
    CAPITAL=${1:-1000000}
    
    echo "Initial Capital: \$$CAPITAL"
    echo ""
    
    # Check prerequisites
    check_python
    
    # Setup environment
    setup_venv
    install_dependencies
    create_directories
    
    # Run smoke test
    run_smoke_test
    
    # Run the system
    run_system $CAPITAL
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [CAPITAL]"
        echo ""
        echo "Arguments:"
        echo "  CAPITAL    Initial capital in dollars (default: 1000000)"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run with $1M capital"
        echo "  $0 500000            # Run with $500K capital"
        echo "  $0 --help            # Show this help"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac