# OpenMP
## Parameters
num_molecules = 1000; // Number of molecules
L = 31.0; // Box length (Angstroms)
dt = 0.5; // Time step (fs)
t_max = 50.0; // Maximum simulation time (fs)

## Before - no openMP
1. Simulation completed in 14.0267 seconds.
2. Simulation completed in 14.1268 seconds.
3. Simulation completed in 13.9361 seconds.
min: 13.9361

## After - openMP
1. Simulation completed in 2.87121 seconds.
2. Simulation completed in 2.46077 seconds.
3. Simulation completed in 2.50393 seconds.
min: 2.46077

Improvement: 82% 

# CUDA
## Parameters
num_molecules = 1000; // Number of molecules
L = 31.0; // Box length (Angstroms)
dt = 0.5; // Time step (fs)
t_max = 5000.0; // Maximum simulation time (fs)

## Before - OpenMP
1. Simulation completed in 253.752 seconds.
2. Simulation completed in 248.661 seconds.
3. Simulation completed in 245.117 seconds.
min: 245.117

## After - CUDA
1. Simulation completed in 70.2646 seconds.
2. Simulation completed in 70.7528 seconds.
3. Simulation completed in 70.6951 seconds.
min: 70.2646

Improvement: 71%