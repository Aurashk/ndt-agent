"""
Unit tests for measurement data structures and synthetic data generation.

Tests the core data structures (MeasurementPoint, MeasurementSet, PogoData)
and synthetic data generation functionality in generate_measure_data.py.
"""

import pytest
import numpy as np
from cscanmaker.generate_measure_data import (
    MeasurementPoint,
    MeasurementSet,
    PogoData,
    PogoMetadata,
    SyntheticDataGenerator,
    PogoDataReader
)


class TestMeasurementPoint:
    """Test MeasurementPoint data structure."""
    
    def test_measurement_point_creation(self):
        """Test basic MeasurementPoint creation and attributes."""
        location = np.array([1.0, 2.0, 3.0])
        data = np.array([0.1, 0.5, 1.0, 0.3, 0.0])
        
        point = MeasurementPoint(
            node_num=42,
            node_dof=1,
            location=location,
            data=data
        )
        
        assert point.node_num == 42
        assert point.node_dof == 1
        assert np.array_equal(point.location, location)
        assert np.array_equal(point.data, data)
    
    def test_measurement_point_location_dimensions(self):
        """Test that location is properly handled as 3D coordinates."""
        point = MeasurementPoint(
            node_num=1,
            node_dof=1,
            location=np.array([10e-3, 20e-3, 5e-3]),  # 10mm, 20mm, 5mm
            data=np.zeros(100)        )
        
        assert len(point.location) == 3
        assert point.location[0] == 10e-3  # X coordinate
        assert point.location[1] == 20e-3  # Y coordinate  
        assert point.location[2] == 5e-3   # Z coordinate
    
    def test_measurement_point_signal_properties(self):
        """Test signal data properties."""
        # Create a signal with known properties
        signal = np.sin(np.linspace(0, 4*np.pi, 1000))
        
        point = MeasurementPoint(
            node_num=1,
            node_dof=1,
            location=np.array([0.0, 0.0, 0.0]),
            data=signal
        )
        
        assert len(point.data) == 1000
        assert np.max(point.data) == pytest.approx(1.0, abs=2e-6)
        assert np.min(point.data) == pytest.approx(-1.0, abs=2e-6)


class TestMeasurementSet:
    """Test MeasurementSet data structure."""
    
    def test_measurement_set_creation(self, simple_measurement_set):
        """Test basic MeasurementSet creation."""
        assert simple_measurement_set.name == "TestSet_3x3_Grid"
        assert len(simple_measurement_set.measurements) == 9  # 3x3 grid
    
    def test_measurement_set_grid_positions(self, simple_measurement_set):
        """Test that measurement positions form expected grid."""
        positions = np.array([m.location for m in simple_measurement_set.measurements])
        
        # Check we have the expected X positions
        x_positions = positions[:, 0]
        expected_x = [-1e-3, 0.0, 1e-3]
        for x in expected_x:
            assert x in x_positions
        
        # Check we have the expected Y positions  
        y_positions = positions[:, 1]
        expected_y = [-1e-3, 0.0, 1e-3]
        for y in expected_y:
            assert y in y_positions
    
    def test_measurement_set_signal_consistency(self, simple_measurement_set):
        """Test that all measurements have consistent signal properties."""
        for measurement in simple_measurement_set.measurements:
            assert len(measurement.data) == 100  # All signals same length
            assert measurement.node_dof == 1     # All same DOF
            assert len(measurement.location) == 3  # All 3D positions


class TestPogoData:
    """Test PogoData structure and properties."""
    
    def test_pogo_data_creation(self, simple_pogo_data):
        """Test basic PogoData creation and structure."""
        assert len(simple_pogo_data.metadata) == 3
        assert len(simple_pogo_data.measurement_sets) == 1
        assert simple_pogo_data.nt_meas == 100
        assert simple_pogo_data.dt_meas == 1e-8
        assert simple_pogo_data.start_meas == 0.0
    
    def test_pogo_data_time_axis(self, simple_pogo_data):
        """Test time axis generation."""
        time_axis = simple_pogo_data.time_axis
        
        assert len(time_axis) == simple_pogo_data.nt_meas
        assert time_axis[0] == simple_pogo_data.start_meas
        assert time_axis[1] - time_axis[0] == pytest.approx(simple_pogo_data.dt_meas)
        
        # Check total duration
        expected_duration = (simple_pogo_data.nt_meas - 1) * simple_pogo_data.dt_meas
        actual_duration = time_axis[-1] - time_axis[0]
        assert actual_duration == pytest.approx(expected_duration)
    
    def test_pogo_data_metadata_access(self, simple_pogo_data):
        """Test metadata structure and access."""
        metadata_names = [m.name for m in simple_pogo_data.metadata]
        
        assert "test_freq" in metadata_names
        assert "test_velocity" in metadata_names
        assert "test_description" in metadata_names
        
        # Find specific metadata
        freq_meta = next(m for m in simple_pogo_data.metadata if m.name == "test_freq")
        assert freq_meta.value == 5e6
        assert freq_meta.type == "float"


class TestSyntheticDataGenerator:
    """Test synthetic data generation functionality."""
    
    def test_generator_initialization(self, synthetic_data_generator):
        """Test SyntheticDataGenerator initialization."""
        gen = synthetic_data_generator
        
        assert gen.freq_center == 5e6
        assert gen.bandwidth == 3e6
        assert gen.c_material == 5900
        assert gen.noise_level == 0.01
    
    def test_synthetic_data_generation_basic(self, synthetic_data_generator):
        """Test basic synthetic data generation."""
        pogo_data = synthetic_data_generator.generate_synthetic_data(
            measurement_name="Test_Basic_Generation",
            nt_samples=200,
            dt_sample=1e-8,
            grid_spacing=2e-3,  # 2mm spacing
            add_flaws=False,    # No flaws for basic test
            add_backwall=False  # No backwall for basic test
        )
        
        assert isinstance(pogo_data, PogoData)
        assert pogo_data.nt_meas == 200
        assert pogo_data.dt_meas == 1e-8
        assert len(pogo_data.measurement_sets) >= 1
        
        # Check we got some measurements
        measurements = pogo_data.measurement_sets[0].measurements
        assert len(measurements) > 0
        
        # Check all measurements have correct signal length
        for measurement in measurements:
            assert len(measurement.data) == 200
    
    def test_synthetic_data_with_flaws(self, synthetic_data_generator):
        """Test synthetic data generation with flaws."""
        pogo_data = synthetic_data_generator.generate_synthetic_data(
            measurement_name="Test_With_Flaws",
            nt_samples=300,
            dt_sample=8e-9,
            grid_spacing=1e-3,
            add_flaws=True,
            add_backwall=False
        )
        
        # With flaws, we should see higher amplitude signals at some locations
        measurements = pogo_data.measurement_sets[0].measurements
        signal_maxes = [np.max(m.data) for m in measurements]
        
        # Should have variation in signal amplitudes due to flaws
        assert np.std(signal_maxes) > 0.01  # Some variation expected
        assert np.max(signal_maxes) > 0.1   # Some significant amplitudes
    
    def test_grid_spacing_affects_measurement_count(self, synthetic_data_generator):
        """Test that grid spacing affects the number of measurements."""
        # Generate with coarse grid
        pogo_coarse = synthetic_data_generator.generate_synthetic_data(
            nt_samples=100,
            grid_spacing=5e-3,  # 5mm spacing
            add_flaws=False
        )
        
        # Generate with fine grid  
        pogo_fine = synthetic_data_generator.generate_synthetic_data(
            nt_samples=100,
            grid_spacing=2e-3,  # 2mm spacing
            add_flaws=False
        )
        
        coarse_count = len(pogo_coarse.measurement_sets[0].measurements)
        fine_count = len(pogo_fine.measurement_sets[0].measurements)
        
        # Fine grid should have more measurements
        assert fine_count > coarse_count


class TestPogoDataReader:
    """Test PogoDataReader functionality."""
    
    def test_reader_initialization(self, simple_pogo_data):
        """Test PogoDataReader initialization."""
        reader = PogoDataReader(simple_pogo_data)
        
        assert reader.data == simple_pogo_data
    
    def test_reader_get_coordinates(self, simple_pogo_data):
        """Test coordinate extraction."""
        reader = PogoDataReader(simple_pogo_data)
        coords = reader.get_coordinates()
        
        assert coords.shape[1] == 3  # 3D coordinates
        assert coords.shape[0] == 9  # 3x3 grid = 9 points
        
        # Check coordinate ranges
        assert np.min(coords[:, 0]) == -1e-3  # Min X
        assert np.max(coords[:, 0]) == 1e-3   # Max X
        assert np.min(coords[:, 1]) == -1e-3  # Min Y
        assert np.max(coords[:, 1]) == 1e-3   # Max Y
    
    def test_reader_get_signals(self, simple_pogo_data):
        """Test signal data extraction."""
        reader = PogoDataReader(simple_pogo_data)
        signals = reader.get_signals()
        
        assert signals.shape[0] == 9    # 9 measurements
        assert signals.shape[1] == 100  # 100 time samples each
    def test_reader_get_time_axis(self, simple_pogo_data):
        """Test time axis extraction."""
        reader = PogoDataReader(simple_pogo_data)
        time_axis = reader.data.time_axis
        
        expected_time_axis = simple_pogo_data.time_axis
        assert np.array_equal(time_axis, expected_time_axis)
    
    def test_reader_summary_output(self, simple_pogo_data, capsys):
        """Test that summary produces reasonable output."""
        reader = PogoDataReader(simple_pogo_data)
        reader.summary()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that summary contains key information
        assert "measurement sets" in output.lower()
        assert "measurements" in output.lower()
        assert "time samples" in output.lower()
        assert "3x3" in output  # Grid information


# Integration-style tests for the module
class TestDataGenerationIntegration:
    """Integration tests for complete data generation workflow."""
    
    def test_complete_workflow_no_flaws(self):
        """Test complete workflow from generation to reading."""
        # Generate data
        generator = SyntheticDataGenerator(
            freq_center=5e6,
            bandwidth=2e6,
            c_material=6000,
            noise_level=0.005
        )
        
        pogo_data = generator.generate_synthetic_data(
            measurement_name="Integration_Test_No_Flaws",
            nt_samples=150,
            dt_sample=1e-8,
            grid_spacing=3e-3,
            add_flaws=False,
            add_backwall=True
        )
          # Read and validate data
        reader = PogoDataReader(pogo_data)
        coords = reader.get_coordinates()
        signals = reader.get_signals()
        time_axis = reader.data.time_axis
        
        # Basic consistency checks
        assert coords.shape[0] == signals.shape[0]  # Same number of positions and signals
        assert signals.shape[1] == len(time_axis)   # Signal length matches time axis
        assert len(time_axis) == pogo_data.nt_meas  # Time axis length correct
        
        # Signal quality checks
        assert not np.any(np.isnan(signals))  # No NaN values
        assert np.all(np.isfinite(signals))   # All finite values
        
        # Backwall should create some signal amplitude
        max_amplitudes = np.max(np.abs(signals), axis=1)
        assert np.mean(max_amplitudes) > 0.01  # Some signal present
    
    def test_complete_workflow_with_flaws(self):
        """Test complete workflow with synthetic flaws."""
        generator = SyntheticDataGenerator(
            freq_center=5e6,
            bandwidth=3e6,
            c_material=5900,
            noise_level=0.02
        )
        
        pogo_data = generator.generate_synthetic_data(
            measurement_name="Integration_Test_With_Flaws",
            nt_samples=250,
            dt_sample=8e-9,
            grid_spacing=1.5e-3,
            add_flaws=True,
            add_backwall=True
        )
        
        reader = PogoDataReader(pogo_data)
        coords = reader.get_coordinates()
        signals = reader.get_signals()
        
        # With flaws, we should see spatial variation in signals
        max_amplitudes = np.max(np.abs(signals), axis=1)
        
        # Check for spatial variation (flaws create local amplitude changes)
        amplitude_std = np.std(max_amplitudes)
        amplitude_mean = np.mean(max_amplitudes)
        coefficient_of_variation = amplitude_std / amplitude_mean
        
        # Should have reasonable variation due to flaws
        assert coefficient_of_variation > 0.1  # At least 10% variation
        assert amplitude_mean > 0.02           # Reasonable signal levels
