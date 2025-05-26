"""
Data structures and synthetic data generation for ultrasonic NDT measurements
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Any, Optional


@dataclass
class PogoMetadata:
    """Container for metadata from pogo file"""
    name: str
    type: str
    value: Any


@dataclass
class MeasurementPoint:
    """Single measurement point with location and time-series data"""
    node_num: int
    node_dof: int
    location: np.ndarray  # [x, y, z] coordinates
    data: np.ndarray      # Time-series ultrasonic signal


@dataclass
class MeasurementSet:
    """Collection of measurements in a set"""
    name: str
    measurements: List[MeasurementPoint]


@dataclass
class PogoData:
    """Complete pogo-hist data structure"""
    metadata: List[PogoMetadata]
    measurement_sets: List[MeasurementSet]
    
    # Timing information
    precision: int = 4  # bytes (4=float32, 8=float64)
    n_dims: int = 3
    nt_meas: int = 1000  # number of time samples
    dt_meas: float = 1e-8  # time step (10 ns)
    start_meas: float = 0.0  # start time
    
    @property
    def time_axis(self) -> np.ndarray:
        """Generate time axis for measurements"""
        return np.arange(self.nt_meas) * self.dt_meas + self.start_meas


class SyntheticDataGenerator:
    """Generate realistic synthetic ultrasonic measurement data"""
    
    def __init__(self, 
                 freq_center: float = 5e6,  # 5 MHz center frequency
                 bandwidth: float = 2e6,    # 2 MHz bandwidth
                 c_material: float = 5900,  # P-wave velocity in steel (m/s)
                 noise_level: float = 0.1):
        self.freq_center = freq_center
        self.bandwidth = bandwidth
        self.c_material = c_material
        self.noise_level = noise_level
    
    def create_measurement_grid(self, 
                              x_range: tuple = (-50e-3, 50e-3),  # -50mm to +50mm
                              y_range: tuple = (-50e-3, 50e-3),  # -50mm to +50mm
                              z_pos: float = 0.0,                # Surface position
                              grid_spacing: float = 2e-3):       # 2mm spacing
        """Create a regular grid of measurement positions"""
        x_positions = np.arange(x_range[0], x_range[1] + grid_spacing, grid_spacing)
        y_positions = np.arange(y_range[0], y_range[1] + grid_spacing, grid_spacing)
        
        positions = []
        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                node_num = i * len(y_positions) + j
                positions.append({
                    'node_num': node_num,
                    'location': np.array([x, y, z_pos]),
                    'grid_i': i,
                    'grid_j': j
                })
        
        return positions, (len(x_positions), len(y_positions))
    
    def generate_ultrasonic_pulse(self, 
                                 time_axis: np.ndarray,
                                 amplitude: float = 1.0,
                                 arrival_time: float = 1e-6) -> np.ndarray:
        """Generate a realistic ultrasonic pulse"""
        # Create Gaussian-modulated sinusoid (typical ultrasonic pulse)
        t_shifted = time_axis - arrival_time
        
        # Gaussian envelope
        sigma = 2.0 / (2 * np.pi * self.bandwidth)  # Pulse width from bandwidth
        envelope = amplitude * np.exp(-0.5 * (t_shifted / sigma) ** 2)
        
        # Sinusoidal carrier
        carrier = np.cos(2 * np.pi * self.freq_center * t_shifted)
        
        # Combined pulse (only after arrival time)
        pulse = envelope * carrier
        pulse[time_axis < arrival_time] = 0  # Causal pulse
        
        return pulse
    
    def add_flaw_response(self, 
                         time_axis: np.ndarray,
                         position: np.ndarray,
                         flaw_center: np.ndarray,
                         flaw_size: float,
                         flaw_depth: float) -> np.ndarray:
        """Add response from a circular flaw"""
        # Distance from measurement point to flaw center
        lateral_dist = np.linalg.norm(position[:2] - flaw_center[:2])
        
        # Flaw response amplitude based on distance and size
        if lateral_dist <= flaw_size:
            # Strong response if directly over flaw
            amplitude = 0.8 * (1 - lateral_dist / flaw_size)
        elif lateral_dist <= 2 * flaw_size:
            # Weaker response from flaw edges
            amplitude = 0.3 * np.exp(-(lateral_dist - flaw_size) / flaw_size)
        else:
            # No response if too far
            return np.zeros_like(time_axis)
        
        # Time of flight to flaw and back
        total_distance = 2 * flaw_depth  # Pulse-echo
        arrival_time = total_distance / self.c_material
        
        return self.generate_ultrasonic_pulse(time_axis, amplitude, arrival_time)
    
    def add_backwall_response(self, 
                             time_axis: np.ndarray,
                             thickness: float = 20e-3) -> np.ndarray:
        """Add backwall echo response"""
        total_distance = 2 * thickness  # Pulse-echo from backwall
        arrival_time = total_distance / self.c_material
        amplitude = 0.6  # Typical backwall amplitude
        
        return self.generate_ultrasonic_pulse(time_axis, amplitude, arrival_time)
    
    def generate_synthetic_data(self,
                               measurement_name: str = "Synthetic_Scan",
                               nt_samples: int = 2000,
                               dt_sample: float = 5e-9,  # 5 ns sampling
                               grid_spacing: float = 1e-3,  # 1mm spacing
                               add_flaws: bool = True,
                               add_backwall: bool = True) -> PogoData:
        """Generate complete synthetic measurement dataset"""
        
        # Create time axis
        time_axis = np.arange(nt_samples) * dt_sample
        
        # Create measurement grid
        positions, grid_shape = self.create_measurement_grid(grid_spacing=grid_spacing)
        
        # Define some flaws for testing
        flaws = []
        if add_flaws:
            flaws = [
                {'center': np.array([10e-3, 15e-3, 5e-3]), 'size': 3e-3, 'depth': 5e-3},   # 3mm flaw at 5mm depth
                {'center': np.array([-20e-3, -10e-3, 8e-3]), 'size': 2e-3, 'depth': 8e-3}, # 2mm flaw at 8mm depth
                {'center': np.array([25e-3, -20e-3, 12e-3]), 'size': 4e-3, 'depth': 12e-3}  # 4mm flaw at 12mm depth
            ]
        
        # Generate measurements
        measurements = []
        
        for pos_info in positions:
            position = pos_info['location']
            
            # Start with noise
            signal = self.noise_level * np.random.randn(len(time_axis))
            
            # Add flaw responses
            for flaw in flaws:
                flaw_signal = self.add_flaw_response(
                    time_axis, position, flaw['center'], flaw['size'], flaw['depth'])
                signal += flaw_signal
            
            # Add backwall response
            if add_backwall:
                backwall_signal = self.add_backwall_response(time_axis, thickness=15e-3)
                signal += backwall_signal
            
            # Create measurement point
            meas_point = MeasurementPoint(
                node_num=pos_info['node_num'],
                node_dof=1,  # Assuming single DOF (normal displacement)
                location=position,
                data=signal
            )
            measurements.append(meas_point)
        
        # Create measurement set
        meas_set = MeasurementSet(name=measurement_name, measurements=measurements)
        
        # Create metadata
        metadata = [
            PogoMetadata("scan_type", "string", "C-scan"),
            PogoMetadata("frequency", "float", self.freq_center),
            PogoMetadata("material_velocity", "float", self.c_material),
            PogoMetadata("grid_shape", "int", grid_shape),
            PogoMetadata("flaws", "string", f"{len(flaws)} synthetic flaws")
        ]
        
        # Create complete data structure
        pogo_data = PogoData(
            metadata=metadata,
            measurement_sets=[meas_set],
            nt_meas=nt_samples,
            dt_meas=dt_sample,
            start_meas=0.0
        )
        
        return pogo_data


class PogoDataReader:
    """Reader for PogoData structures (synthetic or loaded from file)"""
    
    def __init__(self, pogo_data: PogoData):
        self.data = pogo_data
    
    def get_measurement_set(self, idx: int = 0) -> MeasurementSet:
        """Get a specific measurement set"""
        if idx >= len(self.data.measurement_sets):
            raise IndexError(f"Measurement set {idx} not found")
        return self.data.measurement_sets[idx]
    
    def get_coordinates(self, measurement_set_idx: int = 0) -> np.ndarray:
        """Extract coordinates from measurement set"""
        meas_set = self.get_measurement_set(measurement_set_idx)
        return np.array([meas.location for meas in meas_set.measurements])
    
    def get_signals(self, measurement_set_idx: int = 0) -> np.ndarray:
        """Extract all signals as 2D array (n_measurements x n_time_samples)"""
        meas_set = self.get_measurement_set(measurement_set_idx)
        return np.array([meas.data for meas in meas_set.measurements])
    
    def summary(self):
        """Print summary of the data"""
        print(f"Pogo Data Summary:")
        print(f"  Time samples: {self.data.nt_meas}")
        print(f"  Time step: {self.data.dt_meas*1e9:.1f} ns")
        print(f"  Total time: {self.data.time_axis[-1]*1e6:.1f} μs")
        print(f"  Measurement sets: {len(self.data.measurement_sets)}")
        
        for i, meas_set in enumerate(self.data.measurement_sets):
            print(f"    Set {i}: '{meas_set.name}' - {len(meas_set.measurements)} measurements")
        
        print(f"  Metadata entries: {len(self.data.metadata)}")
        for meta in self.data.metadata:
            print(f"    {meta.name}: {meta.value}")