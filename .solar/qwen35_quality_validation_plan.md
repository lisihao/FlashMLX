# Qwen3.5 Quality Validation Plan (Task #78)

## Objective

Validate that hybrid cache injection maintains output quality on real Qwen3.5 model across 4 critical scenarios.

## Test Scenarios

### Scenario 1: Chinese Generation
- **Prompt**: 请介绍一下人工智能的发展历史。
- **Expected**: Coherent Chinese text about AI history
- **Validation**:
  - No gibberish (repeating tokens)
  - Length within 20% of baseline
  - Proper Chinese characters (no corrupted output)

### Scenario 2: Think Mode
- **Prompt**: Solve this problem step by step: What is 15 * 23? Use <think> tags.
- **Expected**: Step-by-step reasoning with <think> tags
- **Validation**:
  - Contains <think> and </think> tags
  - Shows reasoning steps
  - Arrives at correct answer (345)

### Scenario 3: Formatted Output (JSON)
- **Prompt**: Generate a JSON user profile (name, age, email, hobbies)
- **Expected**: Valid JSON object
- **Validation**:
  - Valid JSON syntax
  - All required fields present
  - Proper escaping

### Scenario 4: Mixed Language
- **Prompt**: Explain ML in Chinese, then English examples
- **Expected**: Coherent language switching
- **Validation**:
  - Contains both Chinese and English
  - Natural language transitions
  - No language mixing within sentences

## Acceptance Criteria

✅ **PASS**: All scenarios produce coherent output without gibberish
✅ **PASS**: Output length within ±20% of baseline
✅ **PASS**: Scenario-specific features present (think tags, JSON, mixed lang)

❌ **FAIL**: Any scenario produces gibberish or corrupted output
❌ **FAIL**: Output length deviates > 20% from baseline
❌ **FAIL**: Critical features missing

## Test Framework

### Mock Tests (✅ Completed)
- Framework validation with mock model
- 6 tests, all passing
- File: `tests/integration/test_qwen35_quality_mock.py`

### Real Model Tests (⏳ Pending Real Model)
- Full validation on Qwen3.5-35B-Instruct-4bit
- 4 scenario tests + 1 report generation
- File: `tests/integration/test_qwen35_quality.py`

## Running Tests

```bash
# Mock tests (no model required)
./scripts/run_quality_validation.sh mock

# Real model tests (requires Qwen3.5)
./scripts/run_quality_validation.sh real

# Both
./scripts/run_quality_validation.sh both
```

## Status

- [x] Test framework created
- [x] Mock tests passing (6/6)
- [x] Test runner script created
- [ ] Real model tests executed (pending model availability)
- [ ] Quality report generated

## Next Steps

1. **If Qwen3.5 model available**:
   - Run `./scripts/run_quality_validation.sh real`
   - Review quality report
   - Mark Task #78 as completed

2. **If model not available**:
   - Mark Task #78 as framework-ready
   - Move to Task #79 (Memory savings test)
   - Defer real model validation to integration phase

## Notes

- Mock tests validate the framework is working correctly
- Real model tests require ~70GB disk space for Qwen3.5-35B-4bit
- Tests can be run separately or together
- Quality report auto-generated after successful test run
