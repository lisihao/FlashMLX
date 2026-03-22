#!/bin/bash
#
# Qwen3.5 Hybrid Cache Test Runner
#
# This script runs quality and memory validation tests for the hybrid cache system.
# It supports both mock tests (no model required) and real model tests.
#
# Usage:
#   ./scripts/run_quality_validation.sh [mock|real|both] [quality|memory|all]
#
# Options:
#   Test Mode:
#     mock  - Run mock tests only (default, no model required)
#     real  - Run real Qwen3.5 model tests (requires model download)
#     both  - Run both mock and real tests
#
#   Test Type:
#     quality - Run quality validation tests only
#     memory  - Run memory savings tests only
#     all     - Run all tests (default)
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get test mode (default: mock)
TEST_MODE="${1:-mock}"
TEST_TYPE="${2:-all}"

echo "=================================================="
echo "Qwen3.5 Hybrid Cache Test Runner"
echo "=================================================="
echo "  Test Mode: $TEST_MODE"
echo "  Test Type: $TEST_TYPE"
echo "=================================================="
echo ""

# Function to run quality mock tests
run_quality_mock_tests() {
    echo -e "${GREEN}Running Quality Mock Tests${NC}"
    echo "  (No model required - tests the framework)"
    echo ""

    python3 -m pytest tests/integration/test_qwen35_quality_mock.py -v

    echo ""
    echo -e "${GREEN}✅ Quality mock tests completed${NC}"
}

# Function to run memory mock tests
run_memory_mock_tests() {
    echo -e "${GREEN}Running Memory Mock Tests${NC}"
    echo "  (No model required - tests the framework)"
    echo ""

    python3 -m pytest tests/integration/test_memory_savings_mock.py -v

    echo ""
    echo -e "${GREEN}✅ Memory mock tests completed${NC}"
}

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

# Function to run quality real model tests
run_quality_real_tests() {
    echo -e "${GREEN}Running Quality Real Model Tests${NC}"
    echo "  (Requires Qwen3.5 model)"
    echo ""

    check_model_available

    # Run real quality tests
    python3 -m pytest tests/integration/test_qwen35_quality.py -v

    echo ""
    echo -e "${GREEN}✅ Quality real model tests completed${NC}"
}

# Function to run memory real model tests
run_memory_real_tests() {
    echo -e "${GREEN}Running Memory Real Model Tests${NC}"
    echo "  (Requires Qwen3.5 model)"
    echo ""

    check_model_available

    # Run real memory tests
    python3 -m pytest tests/integration/test_memory_savings.py -v

    echo ""
    echo -e "${GREEN}✅ Memory real model tests completed${NC}"
}

# Main execution
case "$TEST_MODE" in
    mock)
        case "$TEST_TYPE" in
            quality)
                run_quality_mock_tests
                ;;
            memory)
                run_memory_mock_tests
                ;;
            all)
                run_quality_mock_tests
                echo ""
                echo "=================================================="
                echo ""
                run_memory_mock_tests
                ;;
            *)
                echo -e "${RED}❌ Invalid test type: $TEST_TYPE${NC}"
                echo ""
                echo "Usage: $0 [mock|real|both] [quality|memory|all]"
                echo ""
                exit 1
                ;;
        esac
        ;;
    real)
        case "$TEST_TYPE" in
            quality)
                run_quality_real_tests
                ;;
            memory)
                run_memory_real_tests
                ;;
            all)
                run_quality_real_tests
                echo ""
                echo "=================================================="
                echo ""
                run_memory_real_tests
                ;;
            *)
                echo -e "${RED}❌ Invalid test type: $TEST_TYPE${NC}"
                echo ""
                echo "Usage: $0 [mock|real|both] [quality|memory|all]"
                echo ""
                exit 1
                ;;
        esac
        ;;
    both)
        case "$TEST_TYPE" in
            quality)
                run_quality_mock_tests
                echo ""
                echo "=================================================="
                echo ""
                run_quality_real_tests
                ;;
            memory)
                run_memory_mock_tests
                echo ""
                echo "=================================================="
                echo ""
                run_memory_real_tests
                ;;
            all)
                run_quality_mock_tests
                echo ""
                echo "=================================================="
                echo ""
                run_memory_mock_tests
                echo ""
                echo "=================================================="
                echo ""
                run_quality_real_tests
                echo ""
                echo "=================================================="
                echo ""
                run_memory_real_tests
                ;;
            *)
                echo -e "${RED}❌ Invalid test type: $TEST_TYPE${NC}"
                echo ""
                echo "Usage: $0 [mock|real|both] [quality|memory|all]"
                echo ""
                exit 1
                ;;
        esac
        ;;
    *)
        echo -e "${RED}❌ Invalid test mode: $TEST_MODE${NC}"
        echo ""
        echo "Usage: $0 [mock|real|both] [quality|memory|all]"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo -e "${GREEN}✅ All tests completed successfully!${NC}"
echo "=================================================="
