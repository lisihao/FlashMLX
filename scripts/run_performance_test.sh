#!/bin/bash
#
# Performance Overhead Test Runner
#
# This script runs performance overhead tests for the hybrid cache system.
# It supports both mock tests (no model required) and real model tests.
#
# Usage:
#   ./scripts/run_performance_test.sh [mock|real|both]
#
# Options:
#   mock  - Run mock tests only (default, no model required)
#   real  - Run real Qwen3.5 model tests (requires model download)
#   both  - Run both mock and real tests
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get test mode (default: mock)
TEST_MODE="${1:-mock}"

echo "=================================================="
echo "Performance Overhead Test Runner"
echo "=================================================="
echo "  Test Mode: $TEST_MODE"
echo "=================================================="
echo ""

# Function to check model availability
check_model_available() {
    # Check if mlx_lm is installed
    if ! python3 -c "import mlx_lm" 2>/dev/null; then
        echo -e "${RED}❌ mlx_lm not installed${NC}"
        echo ""
        echo "To install:"
        echo "  pip install mlx-lm"
        echo ""
        exit 1
    fi

    # Check if model is available
    MODEL_PATH="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-Instruct-4bit"

    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "${YELLOW}⚠️  Model not found locally${NC}"
        echo ""
        echo "To download Qwen3.5 model:"
        echo "  from mlx_lm import load"
        echo "  load('mlx-community/Qwen3.5-35B-Instruct-4bit')"
        echo ""
        echo "Or run:"
        echo "  huggingface-cli download mlx-community/Qwen3.5-35B-Instruct-4bit"
        echo ""
        read -p "Download model now? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Downloading model..."
            python3 -c "from mlx_lm import load; load('mlx-community/Qwen3.5-35B-Instruct-4bit')"
        else
            echo "Skipping real model tests"
            exit 0
        fi
    fi
}

# Function to run performance mock tests
run_performance_mock_tests() {
    echo -e "${GREEN}Running Performance Mock Tests${NC}"
    echo "  (No model required - tests the framework)"
    echo ""

    python3 -m pytest tests/integration/test_performance_overhead_mock.py -v

    echo ""
    echo -e "${GREEN}✅ Performance mock tests completed${NC}"
}

# Function to run performance real model tests
run_performance_real_tests() {
    echo -e "${GREEN}Running Performance Real Model Tests${NC}"
    echo "  (Requires Qwen3.5 model)"
    echo ""

    check_model_available

    # Run real performance tests
    python3 -m pytest tests/integration/test_performance_overhead.py -v

    echo ""
    echo -e "${GREEN}✅ Performance real model tests completed${NC}"
}

# Main execution
case "$TEST_MODE" in
    mock)
        run_performance_mock_tests
        ;;
    real)
        run_performance_real_tests
        ;;
    both)
        run_performance_mock_tests
        echo ""
        echo "=================================================="
        echo ""
        run_performance_real_tests
        ;;
    *)
        echo -e "${RED}❌ Invalid test mode: $TEST_MODE${NC}"
        echo ""
        echo "Usage: $0 [mock|real|both]"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo -e "${GREEN}✅ All tests completed successfully!${NC}"
echo "=================================================="
