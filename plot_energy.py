import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('energy.csv')

# Create figure and plot
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Kinetic'], label='Kinetic (kcal/mol)', marker='o', markersize=3)
plt.plot(df['Time'], df['Potential'], label='Potential (kcal/mol)', marker='s', markersize=3)
plt.plot(df['Time'], df['Total'], label='Total Energy (kcal/mol)', marker='^', markersize=3)
plt.plot(df['Time'], df['Temperature'], label='Temperature (K)', marker='d', markersize=3)
plt.xlabel('Time (fs)')
plt.ylabel('Energy (kcal/mol) / Temperature (K)')
plt.title('Energy and Temperature vs Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()