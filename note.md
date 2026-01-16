## Parameters
// Parameters
const int num_molecules = 1000; // Number of molecules
const double L = 31.0; // Box length (Angstroms)
const double dt = 0.5; // Time step (fs)
const double t_max = 50.0; // Maximum simulation time (fs)
const Ensemble ensemble = Ensemble::NVT; // Ensemble type
const double kb = 0.001987204; // Boltzmann constant (kcal/mol/K)
const double target_temp = 300.0; // Target temperature (K)

// Conversion Factor:
const double F_TO_ACC = 4.184e-4; // [kcal/mol/Angstrom] -> [amu * (A/fs^2)]
const double KE_TO_KCAL = 2390.057; // [amu * (A/fs)^2] -> [kcal/mol]

// Force cutoff
const double CUTOFF = 10.0; // Cutoff distance (Angstroms)
const double CUTOFF_SQ = CUTOFF * CUTOFF;
const double SKIN = 2.0;
const double VERLET_CUTOFF = CUTOFF + SKIN;
const double VERLET_CUTOFF_SQ = VERLET_CUTOFF * VERLET_CUTOFF;

## Before
1. Simulation completed in 14.0267 seconds.
2. Simulation completed in 14.1268 seconds.
3. Simulation completed in 13.9361 seconds.
min: 13.9361

## After
1. Simulation completed in 2.87121 seconds.
2. Simulation completed in 2.46077 seconds.
3. Simulation completed in 2.50393 seconds.
min: 2.46077

Improvement: 82% 