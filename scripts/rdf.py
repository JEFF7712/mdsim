import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Read data
rdf_data = pd.read_csv('outputs/rdf.csv')
peaks, properties = find_peaks(rdf_data['g_r'], height=1.0, distance=5, prominence=0.3)
peak_positions = rdf_data['r'].iloc[peaks].values
peak_heights = rdf_data['g_r'].iloc[peaks].values

# Print peak information
print("Detected RDF Peaks:")
for i, (pos, height) in enumerate(zip(peak_positions, peak_heights), 1):
    print(f"Peak {i}: r = {pos:.3f} Å, g(r) = {height:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(rdf_data['r'], rdf_data['g_r'], linewidth=2, color='blue', label='RDF')
plt.plot(peak_positions, peak_heights, 'ro', markersize=8, label='Peaks')
for pos, height in zip(peak_positions, peak_heights):
    plt.annotate(f'{pos:.2f} Å', 
                xy=(pos, height), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
plt.xlabel('Distance (Å)', fontsize=12)
plt.ylabel('g(r)', fontsize=12)
plt.title('Radial Distribution Function (RDF)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/rdf_plot.png', dpi=300)
plt.show()
