"""
cscan_generator.py

C-scan generation from ultrasonic measurement data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Tuple, Optional
from cscanmaker.generate_measure_data import PogoData, PogoDataReader


class CScanGenerator:
    """Generate C-scan from measurement data"""
    
    def __init__(self, pogo_data: PogoData):
        self.pogo_data = pogo_data
        self.reader = PogoDataReader(pogo_data)
        self.time_axis = pogo_data.time_axis
    
    def analyze_signal(self, 
                      signal: np.ndarray, 
                      analysis_type: str = 'peak',
                      gate_start: Optional[float] = None,
                      gate_end: Optional[float] = None) -> float:
        """
        Analyze a single signal and return a scalar value
        
        Parameters:
        - signal: Time-domain signal
        - analysis_type: 'peak', 'rms', 'energy', 'tof', 'peak_to_peak'
        - gate_start, gate_end: Time gates (None = full signal)
        """
        # Apply time gating if specified
        if gate_start is not None or gate_end is not None:
            start_idx = 0 if gate_start is None else np.argmin(np.abs(self.time_axis - gate_start))
            end_idx = len(self.time_axis) if gate_end is None else np.argmin(np.abs(self.time_axis - gate_end))
            gated_signal = signal[start_idx:end_idx]
            gated_time = self.time_axis[start_idx:end_idx]
        else:
            gated_signal = signal
            gated_time = self.time_axis
        
        # Compute analysis value
        if analysis_type == 'peak':
            return np.max(np.abs(gated_signal))
        
        elif analysis_type == 'peak_to_peak':
            return np.max(gated_signal) - np.min(gated_signal)
        
        elif analysis_type == 'rms':
            return np.sqrt(np.mean(gated_signal**2))
        
        elif analysis_type == 'energy':
            return np.sum(gated_signal**2)
        
        elif analysis_type == 'tof':
            # Time of flight to peak
            peak_idx = np.argmax(np.abs(gated_signal))
            return gated_time[peak_idx]
        
        elif analysis_type == 'mean':
            return np.mean(np.abs(gated_signal))
        
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    def generate_cscan_data(self, 
                           measurement_set_idx: int = 0,
                           analysis_type: str = 'peak',
                           gate_start: Optional[float] = None,
                           gate_end: Optional[float] = None,
                           grid_resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate C-scan data from measurements
        
        Returns:
        - X, Y: Meshgrid coordinates
        - Z: Interpolated C-scan values
        - coords: Original measurement coordinates
        - values: Original measurement values
        """
        if measurement_set_idx >= len(self.pogo_data.measurement_sets):
            raise ValueError(f"Measurement set {measurement_set_idx} not found")
        
        # Get measurement data
        coordinates = self.reader.get_coordinates(measurement_set_idx)
        signals = self.reader.get_signals(measurement_set_idx)
        
        # Analyze all signals
        values = []
        for signal in signals:
            value = self.analyze_signal(signal, analysis_type, gate_start, gate_end)
            values.append(value)
        
        values = np.array(values)
        
        # Use only X-Y coordinates for C-scan (ignore Z if present)
        coords_2d = coordinates[:, :2]
        
        # Create regular grid for interpolation
        x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        
        # Add small margin to avoid edge effects
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        
        xi = np.linspace(x_min - x_margin, x_max + x_margin, grid_resolution)
        yi = np.linspace(y_min - y_margin, y_max + y_margin, grid_resolution)
        X, Y = np.meshgrid(xi, yi)
        
        # Interpolate to regular grid
        Z = griddata(coords_2d, values, (X, Y), method='cubic', fill_value=np.nan)
        
        # Replace NaN values with minimum value for better visualization
        Z_clean = Z.copy()
        Z_clean[np.isnan(Z)] = np.nanmin(Z) if not np.all(np.isnan(Z)) else 0
        
        return X, Y, Z_clean, coords_2d, values
    
    def plot_cscan(self, 
                   measurement_set_idx: int = 0,
                   analysis_type: str = 'peak',
                   gate_start: Optional[float] = None,
                   gate_end: Optional[float] = None,
                   grid_resolution: int = 100,
                   show_points: bool = False,
                   colormap: str = 'jet',
                   figsize: Tuple[float, float] = (10, 8),
                   title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot C-scan with customizable parameters
        
        Returns:
        - fig, ax: Matplotlib figure and axes objects
        """
        # Generate C-scan data
        X, Y, Z, coords, values = self.generate_cscan_data(
            measurement_set_idx, analysis_type, gate_start, gate_end, grid_resolution)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create C-scan image
        im = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                      origin='lower', aspect='equal', cmap=colormap)
        
        # Overlay measurement points if requested
        if show_points:
            scatter = ax.scatter(coords[:, 0] * 1000, coords[:, 1] * 1000, 
                               c=values, s=15, cmap=colormap, 
                               edgecolors='black', linewidth=0.3, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        
        # Convert axis to mm for better readability
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        ax.set_xticklabels([f'{x*1000:.0f}' for x in x_ticks])
        ax.set_yticklabels([f'{y*1000:.0f}' for y in y_ticks])
        
        # Title
        if title is None:
            gate_info = ""
            if gate_start is not None or gate_end is not None:
                start_str = f"{gate_start*1e6:.1f}" if gate_start else "start"
                end_str = f"{gate_end*1e6:.1f}" if gate_end else "end"
                gate_info = f" (Gate: {start_str} - {end_str} μs)"
            
            meas_set_name = self.pogo_data.measurement_sets[measurement_set_idx].name
            ax.set_title(f'C-Scan: {meas_set_name}\nAnalysis: {analysis_type.title()}{gate_info}')
        else:
            ax.set_title(title)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{analysis_type.title()} Amplitude', rotation=270, labelpad=20)
        
        # Grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig, ax
    
    def compare_analysis_types(self, 
                              measurement_set_idx: int = 0,
                              analysis_types: list = ['peak', 'rms', 'energy'],
                              gate_start: Optional[float] = None,
                              gate_end: Optional[float] = None,
                              figsize: Tuple[float, float] = (15, 5)) -> plt.Figure:
        """
        Compare multiple analysis types side by side
        """
        n_types = len(analysis_types)
        fig, axes = plt.subplots(1, n_types, figsize=figsize)
        
        if n_types == 1:
            axes = [axes]
        
        for i, analysis_type in enumerate(analysis_types):
            X, Y, Z, coords, values = self.generate_cscan_data(
                measurement_set_idx, analysis_type, gate_start, gate_end)
            
            im = axes[i].imshow(Z, extent=[X.min()*1000, X.max()*1000, Y.min()*1000, Y.max()*1000], 
                              origin='lower', aspect='equal', cmap='jet')
            
            axes[i].set_xlabel('X Position (mm)')
            if i == 0:
                axes[i].set_ylabel('Y Position (mm)')
            axes[i].set_title(f'{analysis_type.title()} Analysis')
            axes[i].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label('Amplitude', rotation=270, labelpad=15)
        
        plt.tight_layout()
        return fig
    
    def get_cscan_stats(self, 
                       measurement_set_idx: int = 0,
                       analysis_type: str = 'peak') -> dict:
        """Get statistics about the C-scan data"""
        X, Y, Z, coords, values = self.generate_cscan_data(measurement_set_idx, analysis_type)
        
        stats = {
            'min_value': np.nanmin(values),
            'max_value': np.nanmax(values),
            'mean_value': np.nanmean(values),
            'std_value': np.nanstd(values),
            'n_measurements': len(values),
            'x_range_mm': (coords[:, 0].min() * 1000, coords[:, 0].max() * 1000),
            'y_range_mm': (coords[:, 1].min() * 1000, coords[:, 1].max() * 1000),
            'grid_spacing_mm': np.mean(np.diff(np.unique(coords[:, 0]))) * 1000
        }
        
        return stats