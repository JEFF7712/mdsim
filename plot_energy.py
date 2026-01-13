import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('energy.csv')

# Create figure and plot
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Kinetic'], label='Kinetic', marker='o', markersize=3)
plt.plot(df['Time'], df['Potential'], label='Potential', marker='s', markersize=3)
plt.plot(df['Time'], df['Total'], label='Total', marker='^', markersize=3)

plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy vs Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
