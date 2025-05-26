# Testing Guidelines

This document outlines the testing approach for this project.

## Test Framework

The project uses **pytest** as the primary testing framework. All tests are located in the `tests/` directory.

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with detailed output and show print statements
pytest -v -s
```

### Test Coverage
```bash
# Run tests with coverage report
pytest --cov=src --cov-report=html

# Generate coverage report in terminal
pytest --cov=src --cov-report=term-missing
```

### Specific Test Categories
```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests  
pytest tests/integration/

# Run specific test file
pytest tests/test_synthetic_data.py

# Run specific test function
pytest tests/test_cscan_generator.py::test_peak_analysis

# Run tests matching pattern
pytest -k "test_flaw_detection"
```

## Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and configuration
├── unit/                          # Unit tests
│   ├── test_synthetic_data.py     # Test synthetic data generation
│   ├── test_cscan_generator.py    # Test C-scan generation
│   ├── test_measurement_point.py  # Test data structures
│   └── test_signal_analysis.py    # Test signal processing
├── integration/                   # Integration tests
│   ├── test_end_to_end.py        # Full pipeline tests
│   ├── test_data_flow.py         # Data flow validation
│   └── test_file_io.py           # File I/O operations
├── fixtures/                     # Test data files
│   ├── sample_measurements.pkl
│   ├── reference_cscan.npy
│   └── test_config.json
└── utils/                        # Test utilities
    ├── test_helpers.py           # Common test functions
    └── assertions.py             # Custom assertions
```

## Test Configuration

### pytest.ini Configuration
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --disable-warnings
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow-running tests
    gpu: Tests requiring GPU
```

### Running Specific Test Categories
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run GPU tests only if available
pytest -m gpu
```

## Test Data Management

### Using Fixtures
```python
# In conftest.py - shared test fixtures
import pytest
from synthetic_data import SyntheticDataGenerator

@pytest.fixture
def sample_measurement_data():
    """Generate sample measurement data for testing."""
    generator = SyntheticDataGenerator(noise_level=0.01)
    return generator.generate_synthetic_data(
        nt_samples=500,
        grid_spacing=2e-3,
        add_flaws=True
    )

@pytest.fixture
def cscan_generator(sample_measurement_data):
    """Create C-scan generator with test data."""
    from cscan_generator import CScanGenerator
    return CScanGenerator(sample_measurement_data)
```

### Test Data Validation
```bash
# Validate test data integrity
pytest tests/test_fixtures.py::test_fixture_validity

# Regenerate test fixtures if needed
pytest tests/test_fixtures.py::test_regenerate_fixtures
```

## Continuous Integration

### GitHub Actions Example
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest --cov=src --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Performance Testing

### Benchmarking
```bash
# Run performance tests
pytest tests/performance/ -v

# Run with timing information
pytest --durations=10

# Profile specific tests
pytest --profile tests/test_cscan_generator.py::test_large_dataset
```

## Test Quality Checks

### Pre-commit Testing
```bash
# Run quick validation before commit
pytest tests/unit/ --maxfail=1

# Run formatting and linting checks
black --check .
flake8 src/ tests/
mypy src/
```

### Regression Testing
```bash
# Run regression test suite
pytest tests/regression/ -v

# Compare against reference results
pytest tests/regression/test_cscan_reference.py
```

## Troubleshooting Tests

### Common Issues
```bash
# Clear pytest cache
pytest --cache-clear

# Run tests in random order to catch dependencies
pytest --random-order

# Run failed tests from last run
pytest --lf

# Debug test failures with PDB
pytest --pdb
```

### Test Dependencies
```bash
# Check for test isolation issues
pytest --forked

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto
```

## Writing Good Tests

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test functions: `test_<functionality_being_tested>`
- Test classes: `Test<ClassName>`

### Example Test Structure
```python
def test_cscan_peak_analysis_with_known_flaw():
    """Test C-scan peak analysis correctly identifies synthetic flaw."""
    # Arrange
    generator = SyntheticDataGenerator(noise_level=0.01)
    data = generator.generate_synthetic_data(add_flaws=True)
    cscan = CScanGenerator(data)
    
    # Act
    X, Y, Z, coords, values = cscan.generate_cscan_data(analysis_type='peak')
    
    # Assert
    assert Z.shape == (100, 100)  # Default grid resolution
    assert np.max(values) > 0.5   # Should detect flaw response
    assert not np.any(np.isnan(Z))  # No NaN values in result
```

This testing framework ensures code quality and reliability throughout the development process.