# Coding Style Guidelines

This document outlines the coding standards and style guidelines for this project

## Core Requirements

### Documentation
- **Every function and class MUST have a Python docstring**
- Use Google-style docstrings for consistency
- Include parameter types, return types, and examples where appropriate

### Code Complexity
- **Maximum of four indentation levels** in any part of the code
- If code exceeds four indents, it MUST be refactored into smaller functions
- This prevents deeply nested code and improves readability

## Docstring Standards

### Function Docstrings
```python
def generate_cscan_data(self, measurement_set_idx: int = 0, 
                       analysis_type: str = 'peak') -> Tuple[np.ndarray, ...]:
    """Generate C-scan data from ultrasonic measurements.
    
    Processes measurement data and converts it to a regular grid suitable
    for C-scan visualization using various analysis methods.
    
    Args:
        measurement_set_idx: Index of measurement set to process (default: 0)
        analysis_type: Type of signal analysis ('peak', 'rms', 'energy', 'tof')
        
    Returns:
        Tuple containing:
            - X, Y: Meshgrid coordinates (np.ndarray)
            - Z: Interpolated C-scan values (np.ndarray)
            - coords: Original measurement coordinates (np.ndarray)
            - values: Original measurement values (np.ndarray)
            
    Raises:
        ValueError: If measurement_set_idx is invalid or analysis_type unknown
        
    Example:
        >>> cscan = CScanGenerator(pogo_data)
        >>> X, Y, Z, coords, vals = cscan.generate_cscan_data(analysis_type='peak')
    """
```

### Class Docstrings
```python
class CScanGenerator:
    """Generate C-scan visualizations from ultrasonic measurement data.
    
    This class processes ultrasonic NDT measurement data and converts it into
    C-scan format for flaw detection and material characterization. Supports
    multiple analysis methods and time gating.
    
    Attributes:
        pogo_data: Input measurement data structure
        reader: Data reader interface
        time_axis: Time axis for measurements
        
    Example:
        >>> generator = CScanGenerator(measurement_data)
        >>> fig, ax = generator.plot_cscan(analysis_type='peak')
        >>> plt.show()
    """
```

## Code Structure Guidelines

### Indentation Rules
```python
# ✅ GOOD - 4 levels maximum
def process_measurements(self, data):
    """Process measurement data with proper nesting."""
    for measurement_set in data.measurement_sets:
        for measurement in measurement_set.measurements:
            if self._is_valid_measurement(measurement):
                if self._needs_processing(measurement):
                    processed_data = self._process_single_measurement(measurement)
                    
# ❌ BAD - 5+ levels, needs refactoring
def process_measurements_bad(self, data):
    for measurement_set in data.measurement_sets:
        for measurement in measurement_set.measurements:
            if self._is_valid_measurement(measurement):
                if self._needs_processing(measurement):
                    for time_sample in measurement.data:
                        if time_sample > threshold:  # 5th level - REFACTOR!
                            # Process sample
```

### Refactoring Deep Nesting
When code exceeds 4 indentation levels, extract logic into helper methods:

```python
# ✅ GOOD - Refactored approach
def process_measurements(self, data):
    """Process measurement data using helper methods."""
    for measurement_set in data.measurement_sets:
        for measurement in measurement_set.measurements:
            if self._should_process_measurement(measurement):
                self._process_single_measurement(measurement)

def _should_process_measurement(self, measurement):
    """Check if measurement should be processed."""
    return (self._is_valid_measurement(measurement) and 
            self._needs_processing(measurement))
            
def _process_single_measurement(self, measurement):
    """Process individual measurement data."""
    for time_sample in measurement.data:
        if time_sample > self.threshold:
            self._process_time_sample(time_sample)
```

## General Python Style

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `CScanGenerator`, `MeasurementPoint`)
- **Functions/Methods**: `snake_case` (e.g., `generate_cscan`, `plot_results`)
- **Variables**: `snake_case` (e.g., `measurement_data`, `time_axis`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_FREQUENCY`, `DEFAULT_VELOCITY`)
- **Private methods**: `_leading_underscore` (e.g., `_validate_input`)

### Type Hints
- Use type hints for all function parameters and return values
- Import types from `typing` module when needed
- Use `Optional[T]` for nullable parameters

```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def analyze_signals(signals: np.ndarray, 
                   gate_start: Optional[float] = None,
                   gate_end: Optional[float] = None) -> Dict[str, float]:
    """Analyze ultrasonic signals with optional time gating."""
```

### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Local imports
from synthetic_data import PogoData, MeasurementPoint
from debug import debug_logger
```

### Error Handling
- Use specific exception types
- Provide meaningful error messages
- Document exceptions in docstrings

```python
def get_measurement_set(self, idx: int) -> MeasurementSet:
    """Get measurement set by index.
    
    Args:
        idx: Index of measurement set
        
    Returns:
        Requested measurement set
        
    Raises:
        IndexError: If idx is out of range
        TypeError: If idx is not an integer
    """
    if not isinstance(idx, int):
        raise TypeError(f"Index must be integer, got {type(idx)}")
    
    if idx >= len(self.measurement_sets):
        raise IndexError(f"Index {idx} out of range, have {len(self.measurement_sets)} sets")
    
    return self.measurement_sets[idx]
```

## Code Quality Checklist

Before committing code, ensure:

- [ ] All functions have docstrings
- [ ] All classes have docstrings  
- [ ] No code block exceeds 4 indentation levels
- [ ] Type hints are present for parameters and returns
- [ ] Variable names are descriptive
- [ ] Complex logic is broken into helper functions
- [ ] Error cases are handled appropriately
- [ ] Debug logging is included for major operations
