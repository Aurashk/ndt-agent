"""
main.py

Main file for testing synthetic data generation and C-scan creation
"""

import numpy as np
import matplotlib.pyplot as plt
from cscanmaker.generate_measure_data import SyntheticDataGenerator, PogoDataReader
from cscanmaker.measure_data_to_c_scan import CScanGenerator


def create_test_data():
    """Create synthetic test data with known flaws"""
    print("Creating synthetic ultrasonic measurement data...")
    
    # Create synthetic data generator with realistic parameters
    generator = SyntheticDataGenerator(
        freq_center=5e6,     # 5 MHz transducer
        bandwidth=3e6,       # 3 MHz bandwidth
        c_material=5900,     # Steel P-wave velocity (m/s)
        noise_level=0.02     # Low noise level
    )
    
    # Generate dataset with fine grid spacing for good C-scan resolution
    pogo_data = generator.generate_synthetic_data(
        measurement_name="Steel_Plate_with_Flaws",
        nt_samples=1500,     # 1500 time samples
        dt_sample=8e-9,      # 8 ns sampling (125 MHz sampling rate)
        grid_spacing=1e-3,   # 1mm grid spacing
        add_flaws=True,      # Include synthetic flaws
        add_backwall=True    # Include backwall echo
    )
    
    return pogo_data


def demonstrate_basic_cscan(pogo_data):
    """Demonstrate basic C-scan generation"""
    print("\n=== Basic C-Scan Generation ===")
    
    # Create C-scan generator
    cscan_gen = CScanGenerator(pogo_data)
    
    # Show data summary
    reader = PogoDataReader(pogo_data)
    reader.summary()
    
    # Generate and plot basic C-scan
    fig, ax = cscan_gen.plot_cscan(
        analysis_type='peak',
        grid_resolution=150,
        show_points=False,
        title="Peak Amplitude C-Scan"
    )
    
    # Get and print statistics
    stats = cscan_gen.get_cscan_stats(analysis_type='peak')
    print(f"\nC-Scan Statistics:")
    print(f"  Measurements: {stats['n_measurements']}")
    print(f"  Value range: {stats['min_value']:.3f} to {stats['max_value']:.3f}")
    print(f"  Mean ± Std: {stats['mean_value']:.3f} ± {stats['std_value']:.3f}")
    print(f"  Scan area: {stats['x_range_mm'][0]:.1f} to {stats['x_range_mm'][1]:.1f} mm (X)")
    print(f"            {stats['y_range_mm'][0]:.1f} to {stats['y_range_mm'][1]:.1f} mm (Y)")
    print(f"  Grid spacing: ~{stats['grid_spacing_mm']:.1f} mm")
    
    plt.show()
    return fig


def demonstrate_gated_analysis(pogo_data):
    """Demonstrate time-gated analysis"""
    print("\n=== Time-Gated C-Scan Analysis ===")
    
    cscan_gen = CScanGenerator(pogo_data)
    
    # Create C-scans for different time gates
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gate 1: Early time (surface/near-surface flaws)
    axes[0,0].set_title("Early Gate: 0-3 μs\n(Near-surface flaws)")
    X, Y, Z, _, _ = cscan_gen.generate_cscan_data(
        analysis_type='peak', 
        gate_start=0, 
        gate_end=3e-6,
        grid_resolution=120
    )
    im1 = axes[0,0].imshow(Z, extent=[X.min()*1000, X.max()*1000, Y.min()*1000, Y.max()*1000], 
                          origin='lower', aspect='equal', cmap='jet')
    axes[0,0].set_xlabel('X (mm)')
    axes[0,0].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Gate 2: Mid-time (medium depth flaws)
    axes[0,1].set_title("Mid Gate: 2-6 μs\n(Medium depth flaws)")
    X, Y, Z, _, _ = cscan_gen.generate_cscan_data(
        analysis_type='peak', 
        gate_start=2e-6, 
        gate_end=6e-6,
        grid_resolution=120
    )
    im2 = axes[0,1].imshow(Z, extent=[X.min()*1000, X.max()*1000, Y.min()*1000, Y.max()*1000], 
                          origin='lower', aspect='equal', cmap='jet')
    axes[0,1].set_xlabel('X (mm)')
    axes[0,1].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Gate 3: Late time (backwall)
    axes[1,0].set_title("Late Gate: 4-8 μs\n(Backwall region)")
    X, Y, Z, _, _ = cscan_gen.generate_cscan_data(
        analysis_type='peak', 
        gate_start=4e-6, 
        gate_end=8e-6,
        grid_resolution=120
    )
    im3 = axes[1,0].imshow(Z, extent=[X.min()*1000, X.max()*1000, Y.min()*1000, Y.max()*1000], 
                          origin='lower', aspect='equal', cmap='jet')
    axes[1,0].set_xlabel('X (mm)')
    axes[1,0].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Gate 4: Full signal
    axes[1,1].set_title("Full Signal\n(All reflectors)")
    X, Y, Z, _, _ = cscan_gen.generate_cscan_data(
        analysis_type='peak',
        grid_resolution=120
    )
    im4 = axes[1,1].imshow(Z, extent=[X.min()*1000, X.max()*1000, Y.min()*1000, Y.max()*1000], 
                          origin='lower', aspect='equal', cmap='jet')
    axes[1,1].set_xlabel('X (mm)')
    axes[1,1].set_ylabel('Y (mm)')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.show()
    return fig


def demonstrate_analysis_comparison(pogo_data):
    """Compare different analysis methods"""
    print("\n=== Analysis Method Comparison ===")
    
    cscan_gen = CScanGenerator(pogo_data)
    
    # Compare different analysis types
    fig = cscan_gen.compare_analysis_types(
        analysis_types=['peak', 'rms', 'energy', 'peak_to_peak'],
        figsize=(20, 5)
    )
    
    plt.show()
    return fig


def demonstrate_signal_inspection(pogo_data):
    """Show some example signals from different locations"""
    print("\n=== Signal Inspection ===")
    
    reader = PogoDataReader(pogo_data)
    coordinates = reader.get_coordinates()
    signals = reader.get_signals()
    time_axis = pogo_data.time_axis
    
    # Find interesting measurement points
    # Point 1: Center of scan (likely has flaw)
    center_idx = len(signals) // 2
    
    # Point 2: Near known flaw location (10mm, 15mm)
    target_pos = np.array([10e-3, 15e-3])
    distances = np.linalg.norm(coordinates[:, :2] - target_pos, axis=1)
    flaw_idx = np.argmin(distances)
    
    # Point 3: Background area (should be mostly backwall)
    bg_pos = np.array([-30e-3, -30e-3])
    distances = np.linalg.norm(coordinates[:, :2] - bg_pos, axis=1)
    bg_idx = np.argmin(distances)
    
    # Point 4: Another flaw location (-20mm, -10mm)
    target_pos2 = np.array([-20e-3, -10e-3])
    distances = np.linalg.norm(coordinates[:, :2] - target_pos2, axis=1)
    flaw2_idx = np.argmin(distances)
    
    # Plot signals
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    indices = [center_idx, flaw_idx, bg_idx, flaw2_idx]
    titles = ['Center Position', 'Near Flaw 1 (10,15mm)', 'Background (-30,-30mm)', 'Near Flaw 2 (-20,-10mm)']
    
    for i, (idx, title) in enumerate(zip(indices, titles)):
        axes[i].plot(time_axis * 1e6, signals[idx], 'b-', linewidth=1)
        axes[i].set_xlabel('Time (μs)')
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'{title}\nLocation: ({coordinates[idx,0]*1000:.1f}, {coordinates[idx,1]*1000:.1f}) mm')
        axes[i].grid(True, alpha=0.3)
        
        # Mark some key time gates
        axes[i].axvspan(0, 3, alpha=0.2, color='red', label='Early gate')
        axes[i].axvspan(2, 6, alpha=0.2, color='green', label='Mid gate')
        axes[i].axvspan(4, 8, alpha=0.2, color='blue', label='Late gate')
        
        if i == 0:
            axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    return fig


def main():
    """Main function demonstrating all functionality"""
    print("=== Ultrasonic NDT C-Scan Generation Demo ===")
    
    # Create synthetic test data
    pogo_data = create_test_data()
    
    # Demonstrate basic C-scan
    demonstrate_basic_cscan(pogo_data)
    
    # Demonstrate time-gated analysis
    demonstrate_gated_analysis(pogo_data)
    
    # Compare analysis methods
    demonstrate_analysis_comparison(pogo_data)
    
    # Inspect individual signals
    demonstrate_signal_inspection(pogo_data)
    
    print("\n=== Demo Complete ===")
    print("Try modifying the parameters in synthetic_data.py to:")
    print("- Change flaw sizes and positions")
    print("- Adjust noise levels")
    print("- Modify material properties")
    print("- Change measurement grid density")


if __name__ == "__main__":
    main()