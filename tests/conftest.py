"""
Shared test fixtures and configuration for cscanmaker tests.

This module provides common test utilities and fixtures used across
multiple test modules in the test suite.
"""

import pytest
import numpy as np
from cscanmaker.generate_measure_data import (
    SyntheticDataGenerator, 
    PogoData, 
    MeasurementSet, 
    MeasurementPoint,
    PogoMetadata
)


@pytest.fixture
def simple_measurement_point():
    """Create a simple measurement point for testing.
    
    Returns:
        MeasurementPoint: Basic measurement point with synthetic signal
    """
    # Create a simple synthetic signal with a peak
    time_samples = 100
    signal = np.zeros(time_samples)
    signal[30:35] = np.array([0.1, 0.5, 1.0, 0.5, 0.1])  # Peak at samples 30-34
    signal += np.random.normal(0, 0.01, time_samples)  # Add small noise
    
    return MeasurementPoint(
        node_num=1,
        node_dof=1,
        location=np.array([0.0, 0.0, 0.0]),
        data=signal
    )


@pytest.fixture
def simple_measurement_set(simple_measurement_point):
    """Create a measurement set with a few measurement points.
    
    Args:
        simple_measurement_point: Fixture providing basic measurement point
        
    Returns:
        MeasurementSet: Set containing multiple measurement points
    """
    measurements = []
    
    # Create a 3x3 grid of measurements
    for i, x in enumerate([-1e-3, 0.0, 1e-3]):  # -1mm, 0mm, 1mm
        for j, y in enumerate([-1e-3, 0.0, 1e-3]):
            # Create signal with varying amplitude based on position
            time_samples = 100
            signal = np.zeros(time_samples)
            amplitude = 0.5 + 0.3 * np.sin(i) * np.cos(j)  # Varying amplitude
            signal[30:35] = amplitude * np.array([0.1, 0.5, 1.0, 0.5, 0.1])
            signal += np.random.normal(0, 0.01, time_samples)
            
            measurement = MeasurementPoint(
                node_num=i*3 + j + 1,
                node_dof=1,
                location=np.array([x, y, 0.0]),
                data=signal
            )
            measurements.append(measurement)
    
    return MeasurementSet(
        name="TestSet_3x3_Grid",
        measurements=measurements
    )


@pytest.fixture
def simple_pogo_data(simple_measurement_set):
    """Create basic PogoData structure for testing.
    
    Args:
        simple_measurement_set: Fixture providing measurement set
        
    Returns:
        PogoData: Complete pogo data structure with test data
    """
    metadata = [
        PogoMetadata(name="test_freq", type="float", value=5e6),
        PogoMetadata(name="test_velocity", type="float", value=5900),
        PogoMetadata(name="test_description", type="string", value="Unit test data")
    ]
    
    return PogoData(
        metadata=metadata,
        measurement_sets=[simple_measurement_set],
        precision=4,
        n_dims=3,
        nt_meas=100,
        dt_meas=1e-8,  # 10 ns
        start_meas=0.0
    )


@pytest.fixture
def synthetic_data_generator():
    """Create a synthetic data generator with test parameters.
    
    Returns:
        SyntheticDataGenerator: Generator configured for testing
    """
    return SyntheticDataGenerator(
        freq_center=5e6,
        bandwidth=3e6,
        c_material=5900,
        noise_level=0.01  # Low noise for predictable tests
    )
