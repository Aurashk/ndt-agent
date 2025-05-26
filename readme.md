# Ultrasonic NDT C-Scan Generation

A Python package for generating C-scan visualizations from ultrasonic non-destructive testing (NDT) measurement data. This project provides tools for processing phased array ultrasonic data and converting it into C-scan format for flaw detection and material characterization.

## Features

- **Synthetic Data Generation**: Create realistic ultrasonic measurement data with configurable flaws, material properties, and measurement grids
- **Multiple Analysis Methods**: Peak amplitude, RMS, energy, time-of-flight, and peak-to-peak analysis
- **Time Gating**: Analyze specific depth ranges using configurable time gates
- **Spatial Interpolation**: Convert irregular measurement points to regular grids for visualization
- **Comprehensive Visualization**: Generate publication-quality C-scan plots with customizable colormaps
- **Debug System**: Extensive logging and debug visualization capabilities
- **Extensible Architecture**: Clean modular design for easy extension and customization

## Installation

### Setup
```bash
# Clone the repository
git clone https://github.com/Aurashk/ndt-agent
cd ndt-agent

# Install dependencies
pip install -r requirements.txt

# Install in development mode (recommended)
pip install -e .
```

## Quick Start

### Basic Usage
```python
from synthetic_data import SyntheticDataGenerator
from cscan_generator import CScanGenerator

# Generate synthetic ultrasonic data
generator = SyntheticDataGenerator(
    freq_center=5e6,     # 5 MHz transducer
    c_material=5900,     # Steel P-wave velocity
    noise_level=0.02     # 2% noise
)

# Create measurement dataset with flaws
data = generator.generate_synthetic_data(
    grid_spacing=1e-3,   # 1mm spacing
    add_flaws=True,      # Include synthetic flaws
    add_backwall=True    # Include backwall echo
)

# Generate C-scan
cscan = CScanGenerator(data)
fig, ax = cscan.plot_cscan(analysis_type='peak')
plt.show()
```

### Complete Example
```python
# Run the complete demonstration
python main.py
```

This will generate:
- Basic C-scan visualization
- Time-gated analysis (different depth ranges)
- Comparison of analysis methods
- Individual signal inspection

## Core Components

### Data Structures
- **`MeasurementPoint`**: Individual ultrasonic measurement with 3D location and time-series data
- **`MeasurementSet`**: Collection of measurements (e.g., one scan)

### Synthetic Data Generation
- Realistic ultrasonic pulse generation (Gaussian-modulated sinusoids)
- Configurable material properties and transducer parameters
- Synthetic flaw responses with proper time-of-flight calculations
- Backwall echo simulation
- Measurement grid generation with customizable spacing

### C-Scan Analysis
- **Peak Analysis**: Maximum amplitude detection for flaw identification
- **RMS Analysis**: Root mean square for material characterization
- **Energy Analysis**: Total signal energy calculation
- **Time-of-Flight**: Depth/ranging analysis
- **Time Gating**: Analyze specific depth ranges

## Usage Examples

### Creating Custom Synthetic Data
```python
from synthetic_data import SyntheticDataGenerator

# Custom transducer and material properties
generator = SyntheticDataGenerator(
    freq_center=10e6,    # 10 MHz high-frequency transducer
    bandwidth=5e6,       # 5 MHz bandwidth
    c_material=6420,     # Aluminum P-wave velocity
    noise_level=0.05     # 5% noise level
)

# Generate dataset with custom parameters
data = generator.generate_synthetic_data(
    measurement_name="Aluminum_Plate_Scan",
    nt_samples=2000,     # High temporal resolution
    dt_sample=5e-9,      # 5 ns sampling (200 MHz)
    grid_spacing=0.5e-3, # 0.5mm fine grid
    add_flaws=True,
    add_backwall=True
)
```

### Advanced C-Scan Analysis
```python
from cscan_generator import CScanGenerator

cscan = CScanGenerator(data)

# Time-gated analysis for specific depth
fig, ax = cscan.plot_cscan(
    analysis_type='peak',
    gate_start=1e-6,     # Start at 1 microsecond
    gate_end=4e-6,       # End at 4 microseconds
    grid_resolution=200, # High-resolution interpolation
    colormap='viridis'   # Custom colormap
)

# Compare multiple analysis methods
fig = cscan.compare_analysis_types(
    analysis_types=['peak', 'rms', 'energy'],
    gate_start=0.5e-6,
    gate_end=6e-6
)

# Get quantitative statistics
stats = cscan.get_cscan_stats(analysis_type='peak')
print(f"Flaw detection range: {stats['min_value']:.3f} to {stats['max_value']:.3f}")
```

## Debug Mode

Enable comprehensive debugging for development:

```bash
# Run with debug logging
python -O main.py

# Or set environment variable
export PYTHON_DEBUG=1
python main.py
```

Debug mode creates timestamped folders in `debug/` with:
- Detailed execution logs
- Intermediate data structures
- Debug visualization images
- Performance metrics

See [debugging.md](docs/debugging.md) for detailed information.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
```

See [testing.md](docs/testing.md) for comprehensive testing guidelines.

## Configuration

### Material Properties
Common ultrasonic velocities for C-scan generation:

| Material | P-wave Velocity (m/s) | Typical Frequency (MHz) |
|----------|----------------------|-------------------------|
| Steel    | 5900                 | 2-10                   |
| Aluminum | 6420                 | 5-15                   |
| Titanium | 6100                 | 2-10                   |
| Composite| 2700-3200            | 1-5                    |

### Analysis Guidelines
- **Peak Analysis**: Best for flaw detection and sizing
- **RMS Analysis**: Good for material property assessment
- **Energy Analysis**: Sensitive to overall signal strength
- **Time-of-Flight**: Essential for depth measurement and ranging

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis-method`)
3. Follow the coding standards in [codingstyle.md](docs/codingstyle.md)
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -e .[dev]

# Run quality checks
black --check .
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the test suite for usage examples
