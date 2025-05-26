# Debugging System

This document describes the debugging system for this project.

## Overview

The debugging system provides comprehensive logging and data capture capabilities that activate only when Python is run in debug mode. It automatically creates timestamped folders and captures debug information, images, and intermediate data structures.

## Debug Mode Activation

### Running in Debug Mode
```bash
# Enable debug mode using Python's -O flag (optimize off)
python -O main.py

# Or set the debug environment variable
export PYTHON_DEBUG=1
python main.py

# Or use the development flag
python -d main.py
```

### Automatic Detection
The debug system automatically detects if Python is running in debug mode and activates accordingly. If not in debug mode, all debug functions become no-ops with minimal performance impact.

## Debug Directory Structure

When debug mode is active, the system creates the following structure:

```
project_root/
└── debug/
    └── 2025-05-26_14-30-45/  # Timestamp folder for this run
        ├── debug.log          # Main debug log file
        ├── config.json        # Run configuration
        ├── images/            # Debug images folder
        │   ├── 001_synthetic_data_overview.png
        │   ├── 002_signal_analysis.png
        │   ├── 003_cscan_peak_analysis.png
        │   └── ...
        └── data/              # Intermediate data structures
            ├── measurement_data.pkl
            ├── cscan_data.npy
            └── analysis_results.json
```

## Debug Script Usage

### Importing the Debug System
```python
from debug import debug_logger

# The debug_logger is a global singleton that handles all debug operations
```

### Basic Logging
```python
# Log messages with different levels
debug_logger.info("Starting C-scan generation for measurement set 0")
debug_logger.warning("Low signal amplitude detected at position (10, 15)")  
debug_logger.error("Failed to process measurement point 42")
debug_logger.debug("Intermediate calculation: peak_amplitude = 0.456")

# Log with context
debug_logger.info("Processing flaw detection", extra={
    'flaw_position': [10e-3, 15e-3, 5e-3],
    'flaw_size': 3e-3,
    'analysis_type': 'peak'
})
```

### Saving Debug Images
```python
import matplotlib.pyplot as plt

# Create your plot
fig, ax = plt.subplots()
ax.plot(time_axis, signal_data)
ax.set_title("Ultrasonic Signal Analysis")

# Save debug image (automatically numbered)
debug_logger.save_image(fig, "signal_analysis", description="Raw ultrasonic signal from position (x, y)")

# The image will be saved as: images/XXX_signal_analysis.png
# Where XXX is auto-incremented (001, 002, 003, etc.)
```

### Saving Data Structures
```python
# Save intermediate data for debugging
debug_logger.save_data(measurement_data, "measurement_data", "Raw measurement data from synthetic generator")
debug_logger.save_data(cscan_results, "cscan_peak_analysis", "C-scan results using peak analysis")
debug_logger.save_data(flaw_positions, "detected_flaws", "Positions of detected flaws")

# Save numpy arrays efficiently
debug_logger.save_array(signal_matrix, "signal_matrix", "2D array of all ultrasonic signals")

# Save configuration/parameters
debug_logger.save_config({
    'frequency': 5e6,
    'material_velocity': 5900,
    'grid_spacing': 1e-3,
    'analysis_type': 'peak'
}, "analysis_parameters")
```

## Integration Throughout Codebase

### In Data Generation (synthetic_data.py)
```python
class SyntheticDataGenerator:
    def generate_synthetic_data(self, **kwargs):
        """Generate synthetic measurement data with debug logging."""
        debug_logger.info("Starting synthetic data generation", extra=kwargs)
        
        # Log major steps
        debug_logger.debug("Creating measurement grid")
        positions, grid_shape = self.create_measurement_grid(...)
        debug_logger.save_data(positions, "measurement_positions", "Grid positions for measurements")
        
        debug_logger.debug(f"Generated grid with {len(positions)} positions")
        
        # Log flaw creation
        for i, flaw in enumerate(flaws):
            debug_logger.info(f"Adding synthetic flaw {i+1}", extra=flaw)
        
        # Save final data structure
        debug_logger.save_data(pogo_data, "synthetic_pogo_data", "Complete synthetic measurement dataset")
        
        # Create overview plot
        self._create_debug_overview_plot(pogo_data)
        
        return pogo_data
    
    def _create_debug_overview_plot(self, pogo_data):
        """Create debug overview visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot measurement grid
        coords = [m.location for m in pogo_data.measurement_sets[0].measurements]
        coords = np.array(coords)
        axes[0,0].scatter(coords[:, 0]*1000, coords[:, 1]*1000, s=1)
        axes[0,0].set_title("Measurement Grid")
        
        # Plot sample signals
        signals = [m.data for m in pogo_data.measurement_sets[0].measurements[:4]]
        time_axis = pogo_data.time_axis
        for i, signal in enumerate(signals):
            axes[0,1].plot(time_axis*1e6, signal, label=f'Signal {i+1}')
        axes[0,1].set_title("Sample Signals")
        axes[0,1].legend()
        
        # Add more debug plots...
        
        debug_logger.save_image(fig, "synthetic_data_overview", 
                               "Overview of synthetic measurement data generation")
```

### In C-Scan Generation (cscan_generator.py)
```python
class CScanGenerator:
    def generate_cscan_data(self, **kwargs):
        """Generate C-scan with comprehensive debug logging."""
        debug_logger.info("Starting C-scan generation", extra=kwargs)
        
        # Log input data characteristics
        coords = self.reader.get_coordinates()
        signals = self.reader.get_signals()
        debug_logger.info(f"Processing {len(coords)} measurement points")
        debug_logger.save_array(coords, "measurement_coordinates", "3D coordinates of all measurements")
        debug_logger.save_array(signals, "raw_signals", "Raw ultrasonic signals matrix")
        
        # Debug signal analysis
        debug_logger.debug("Analyzing signals with method: " + kwargs.get('analysis_type', 'peak'))
        values = []
        for i, signal in enumerate(signals):
            value = self.analyze_signal(signal, **kwargs)
            values.append(value)
            
            # Log problematic signals
            if np.isnan(value) or value < 0:
                debug_logger.warning(f"Unusual analysis result for signal {i}: {value}")
        
        values = np.array(values)
        debug_logger.save_array(values, "analysis_values", f"Analysis results using {kwargs.get('analysis_type')}")
        
        # Debug interpolation
        debug_logger.debug("Performing spatial interpolation")
        X, Y, Z, coords_2d, values = self._perform_interpolation(coords, values, **kwargs)
        
        # Save intermediate results
        debug_logger.save_array(Z, "cscan_interpolated", "Interpolated C-scan data")
        
        # Create debug visualization
        self._create_debug_cscan_plot(X, Y, Z, coords_2d, values, **kwargs)
        
        return X, Y, Z, coords_2d, values
    
    def _create_debug_cscan_plot(self, X, Y, Z, coords, values, **kwargs):
        """Create detailed debug C-scan visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original measurement points
        axes[0,0].scatter(coords[:, 0]*1000, coords[:, 1]*1000, c=values, s=20)
        axes[0,0].set_title("Original Measurements")
        
        # Interpolated C-scan
        im = axes[0,1].imshow(Z, extent=[X.min()*1000, X.max()*1000, Y.min()*1000, Y.max()*1000])
        axes[0,1].set_title("Interpolated C-scan")
        
        # Analysis statistics
        debug_logger.info(f"C-scan statistics: min={np.nanmin(values):.3f}, max={np.nanmax(values):.3f}, mean={np.nanmean(values):.3f}")
        
        debug_logger.save_image(fig, f"cscan_{kwargs.get('analysis_type', 'unknown')}", 
                               f"C-scan debug visualization using {kwargs.get('analysis_type')} analysis")
```

## Debug Logger API

### Core Methods
```python
class DebugLogger:
    def info(self, message: str, extra: dict = None) -> None:
        """Log info message with optional context."""
        
    def warning(self, message: str, extra: dict = None) -> None:
        """Log warning message."""
        
    def error(self, message: str, extra: dict = None) -> None:
        """Log error message."""
        
    def debug(self, message: str, extra: dict = None) -> None:
        """Log debug message (only in debug mode)."""
        
    def save_image(self, figure, name: str, description: str = "") -> str:
        """Save matplotlib figure with auto-incrementing filename."""
        
    def save_data(self, data, name: str, description: str = "") -> str:
        """Save Python object using pickle."""
        
    def save_array(self, array: np.ndarray, name: str, description: str = "") -> str:
        """Save numpy array efficiently."""
        
    def save_config(self, config: dict, name: str, description: str = "") -> str:
        """Save configuration as JSON."""
        
    def get_debug_folder(self) -> Path:
        """Get current debug session folder path."""
        
    def is_active(self) -> bool:
        """Check if debug mode is active."""
```

## Best Practices

### When to Use Debug Logging
- **Major function entry/exit**: Log when entering/exiting important functions
- **Data transformations**: Save intermediate data structures  
- **Algorithm steps**: Log key algorithmic decisions and results
- **Error conditions**: Log warnings and errors with context
- **Performance metrics**: Log timing information for optimization

### Performance Considerations
- Debug functions are no-ops when not in debug mode (minimal overhead)
- Large data structures are saved efficiently using appropriate formats
- Image saving is asynchronous where possible
- Log messages are buffered and written in batches

### Debug Session Management
```python
# Manually start/stop debug sessions if needed
debug_logger.start_session("custom_session_name")
# ... do work ...
debug_logger.end_session()

# Context manager support
with debug_logger.session("flaw_detection_analysis"):
    # All debug output goes to this specific session
    detect_flaws(measurement_data)
```

## Troubleshooting

### Common Issues
1. **Debug folder not created**: Check write permissions in project root
2. **Images not saving**: Verify matplotlib backend supports file output
3. **Large debug files**: Use `debug_logger.cleanup_old_sessions(days=7)` to remove old data
4. **Performance impact**: Ensure debug mode is disabled in production

### Log File Analysis
The debug log uses structured format for easy parsing:
```
2025-05-26 14:30:45 [INFO] Starting C-scan generation | analysis_type=peak grid_resolution=150
2025-05-26 14:30:46 [DEBUG] Processing 2601 measurement points
2025-05-26 14:30:47 [WARNING] Low signal amplitude at position (25, -20) | amplitude=0.023
```

This debugging system provides comprehensive insight into the NDT analysis process and helps identify issues during development and testing.