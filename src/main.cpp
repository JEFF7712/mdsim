#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <omp.h>
#include "gpu_interface.h"
#include "config.h"

// Constants
const double kb = 0.001987204; // Boltzmann constant (kcal/mol/K)
const double F_TO_ACC = 4.184e-4; // [kcal/mol/Angstrom] -> [amu * (A/fs^2)]
const double KE_TO_KCAL = 2390.057; // [amu * (A/fs)^2] -> [kcal/mol]

// Random number generator
std::mt19937& get_rng() {
    // static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static unsigned seed = 42;
    static std::mt19937 engine(seed);
    return engine;
}

// Atom type structure
struct AtomType {
    std::string name;
    double mass; // amu
    double sigma; // Angstroms
    double epsilon; // kcal/mol
    double charge; 
};

// Bond structure
struct Bond {
    int atom1;
    int atom2;
    double k; // bond strength (kcal/mol/Angstrom^2)
    double rb; // bond length (Angstroms)
};

struct Angle {
    int atom1; 
    int atom2;
    int atom3;
    double k; // Stiffness (kcal/mol/rad^2)
    double theta0; // Equilibrium angle (radians)
};

const std::vector<AtomType> ATOM_TYPES = {
    {"H", 1.008, 0.4000, 0.0460, 0.417},    // Type 0: Hydrogen
    {"O", 15.999, 3.1506, 0.1521, -0.834}   // Type 1: Oxygen
};


struct System {
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> fx, fy, fz;
    std::vector<int> type;
    std::vector<int> id;
    std::vector<std::vector<int>> exclusions;

    void resize(int n) {
        x.resize(n); y.resize(n); z.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
        fx.resize(n); fy.resize(n); fz.resize(n);
        type.resize(n);
        id.resize(n);
        exclusions.resize(n);
    }
};

void init_water_box(System& sys, std::vector<Bond>& bonds, std::vector<Angle>& angles, int num_molecules, double L) {
    int per_side = std::ceil(std::cbrt(num_molecules));
    double spacing = L / per_side;
    sys.resize(num_molecules * 3);
    int p_idx = 0;
    
    for (int x = 0; x < per_side; ++x) {
        for (int y = 0; y < per_side; ++y) {
            for (int z = 0; z < per_side; ++z) {
                // Check if there is enough particles for another water molecule
                if (p_idx + 2 >= sys.x.size()) return;

                double cx = x * spacing + spacing/2;
                double cy = y * spacing + spacing/2;
                double cz = z * spacing + spacing/2;

                // O
                sys.id[p_idx] = p_idx;
                sys.type[p_idx] = 1;
                sys.x[p_idx] = cx;
                sys.y[p_idx] = cy;
                sys.z[p_idx] = cz;
                
                // H1
                sys.id[p_idx + 1] = p_idx + 1;
                sys.type[p_idx + 1] = 0;
                sys.x[p_idx + 1] = cx + 0.6;
                sys.y[p_idx + 1] = cy + 0.6;
                sys.z[p_idx + 1] = cz;

                // H2
                sys.id[p_idx + 2] = p_idx + 2;
                sys.type[p_idx + 2] = 0;
                sys.x[p_idx + 2] = cx - 0.6;
                sys.y[p_idx + 2] = cy + 0.6;
                sys.z[p_idx + 2] = cz;

                // Create Bonds
                bonds.push_back({sys.id[p_idx], sys.id[p_idx + 1], 450.0, 0.9572});
                bonds.push_back({sys.id[p_idx], sys.id[p_idx + 2], 450.0, 0.9572});

                // Create Angle
                double theta_eq = 104.52 * 3.14159 / 180.0;
                angles.push_back({sys.id[p_idx + 1], sys.id[p_idx], sys.id[p_idx + 2], 100.0, theta_eq});

                // Add Exclusions
                sys.exclusions[sys.id[p_idx]].push_back(sys.id[p_idx + 1]);
                sys.exclusions[sys.id[p_idx]].push_back(sys.id[p_idx + 2]);
                sys.exclusions[sys.id[p_idx + 1]].push_back(sys.id[p_idx]);
                sys.exclusions[sys.id[p_idx + 2]].push_back(sys.id[p_idx]);
                sys.exclusions[sys.id[p_idx + 1]].push_back(sys.id[p_idx + 2]);
                sys.exclusions[sys.id[p_idx + 2]].push_back(sys.id[p_idx + 1]);

                p_idx += 3;
            }
        }
    }
}

void compute_bond_forces(System& sys, const std::vector<Bond>& bonds, double& pe_total, double L) {
    double local_pe = 0.0;
    
    #pragma omp parallel for reduction(+:local_pe)
    for (const auto& b : bonds) {

        double dx = sys.x[b.atom1] - sys.x[b.atom2];
        double dy = sys.y[b.atom1] - sys.y[b.atom2];
        double dz = sys.z[b.atom1] - sys.z[b.atom2];

        if (dx > L/2) dx -= L; 
        if (dx < -L/2) dx += L;
        if (dy > L/2) dy -= L; 
        if (dy < -L/2) dy += L;
        if (dz > L/2) dz -= L; 
        if (dz < -L/2) dz += L;

        double r2 = dx*dx + dy*dy + dz*dz;
        double r = std::sqrt(r2);

        double delta = r - b.rb;
        double f_mag = -b.k * delta;
        local_pe += 0.5 * b.k * delta * delta;

        double fx = f_mag * (dx / r);
        double fy = f_mag * (dy / r);
        double fz = f_mag * (dz / r);

        #pragma omp atomic
        sys.fx[b.atom1] += fx;
        #pragma omp atomic 
        sys.fy[b.atom1] += fy; 
        #pragma omp atomic
        sys.fz[b.atom1] += fz;
        #pragma omp atomic
        sys.fx[b.atom2] -= fx; 
        #pragma omp atomic
        sys.fy[b.atom2] -= fy;
        #pragma omp atomic 
        sys.fz[b.atom2] -= fz;
    }
    pe_total += local_pe;
}

void compute_angle_forces(System& sys, const std::vector<Angle>& angles, double& pe_total, double L) {
    double local_pe = 0.0;
    
    #pragma omp parallel for reduction(+:local_pe)
    for (const auto& ang : angles) {

        double r21x = sys.x[ang.atom1] - sys.x[ang.atom2];
        double r21y = sys.y[ang.atom1] - sys.y[ang.atom2];
        double r21z = sys.z[ang.atom1] - sys.z[ang.atom2];

        double r23x = sys.x[ang.atom3] - sys.x[ang.atom2];
        double r23y = sys.y[ang.atom3] - sys.y[ang.atom2];
        double r23z = sys.z[ang.atom3] - sys.z[ang.atom2];

        // Periodic Boundaries
        if (r21x > L/2) r21x -= L; 
        if (r21x < -L/2) r21x += L;
        if (r21y > L/2) r21y -= L; 
        if (r21y < -L/2) r21y += L;
        if (r21z > L/2) r21z -= L; 
        if (r21z < -L/2) r21z += L;

        if (r23x > L/2) r23x -= L; 
        if (r23x < -L/2) r23x += L;
        if (r23y > L/2) r23y -= L; 
        if (r23y < -L/2) r23y += L;
        if (r23z > L/2) r23z -= L; 
        if (r23z < -L/2) r23z += L;

        // Magnitudes squared
        double r21_sq = r21x*r21x + r21y*r21y + r21z*r21z;
        double r23_sq = r23x*r23x + r23y*r23y + r23z*r23z;
        double r21 = std::sqrt(r21_sq);
        double r23 = std::sqrt(r23_sq);

        // Dot product
        double dot = r21x*r23x + r21y*r23y + r21z*r23z;
        double cos_theta = dot / (r21 * r23);

        // Make sure cos_theta is in valid range for acos
        if (cos_theta > 1.0) cos_theta = 1.0;
        if (cos_theta < -1.0) cos_theta = -1.0;

        double theta = std::acos(cos_theta);
        double d_theta = theta - ang.theta0;
        
        // Potential Energy
        local_pe += 0.5 * ang.k * d_theta * d_theta;

        // Force Magnitude
        double force_factor = -ang.k * d_theta;
        double sin_theta = std::sin(theta);
        if (std::abs(sin_theta) < 1e-6) sin_theta = 1e-6; // Avoid div by zero

        double coef1 = force_factor / (r21 * sin_theta);
        double coef2 = force_factor / (r23 * sin_theta);

        double c1 = force_factor / (r21_sq * sin_theta);
        double c2 = force_factor / (r23_sq * sin_theta);

        double f1x = c1 * (r21x * cos_theta - r23x * (r21/r23));
        double f1y = c1 * (r21y * cos_theta - r23y * (r21/r23));
        double f1z = c1 * (r21z * cos_theta - r23z * (r21/r23));

        double f3x = c2 * (r23x * cos_theta - r21x * (r23/r21));
        double f3y = c2 * (r23y * cos_theta - r21y * (r23/r21));
        double f3z = c2 * (r23z * cos_theta - r21z * (r23/r21));

        // Apply forces
        #pragma omp atomic
        sys.fx[ang.atom1] += f1x; 
        #pragma omp atomic
        sys.fy[ang.atom1] += f1y; 
        #pragma omp atomic
        sys.fz[ang.atom1] += f1z;
        #pragma omp atomic
        sys.fx[ang.atom3] += f3x; 
        #pragma omp atomic
        sys.fy[ang.atom3] += f3y; 
        #pragma omp atomic
        sys.fz[ang.atom3] += f3z;
        
        // Center Atom
        #pragma omp atomic
        sys.fx[ang.atom2] -= (f1x + f3x);
        #pragma omp atomic
        sys.fy[ang.atom2] -= (f1y + f3y);
        #pragma omp atomic
        sys.fz[ang.atom2] -= (f1z + f3z);
    }
    pe_total += local_pe;
} 

void integrate_langevin_1(System& sys, double dt, double temp, double friction, double L) {
    const double MAX_DISP = 0.1;
    double gamma = friction / 1000.0;
    double c1 = (1.0 - gamma * dt * 0.5) / (1.0 + gamma * dt * 0.5);
    double c2 = 1.0 / (1.0 + gamma * dt * 0.5);

    std::normal_distribution<double> noise(0.0, 1.0);
    auto& rng = get_rng();

    #pragma omp parallel for
    for (size_t i = 0; i < sys.x.size(); ++i) {
        double m = ATOM_TYPES[sys.type[i]].mass;
        double acc_factor = F_TO_ACC / m; 
        
        double ax = sys.fx[i] * acc_factor;
        double ay = sys.fy[i] * acc_factor;
        double az = sys.fz[i] * acc_factor;

        double variance = (2.0 * kb * temp * gamma * dt) / (m * KE_TO_KCAL);
        double sigma_v = std::sqrt(variance);

        // Update velocity
        sys.vx[i] = c1 * sys.vx[i] + c2 * dt * ax + c2 * sigma_v * noise(rng);
        sys.vy[i] = c1 * sys.vy[i] + c2 * dt * ay + c2 * sigma_v * noise(rng);
        sys.vz[i] = c1 * sys.vz[i] + c2 * dt * az + c2 * sigma_v * noise(rng);

        // Update position 
        double dx = sys.vx[i] * dt;
        double dy = sys.vy[i] * dt;
        double dz = sys.vz[i] * dt;

        // Clamp 
        double dist_sq = dx*dx + dy*dy + dz*dz;
        if (dist_sq > MAX_DISP * MAX_DISP) {
            double scale = MAX_DISP / std::sqrt(dist_sq);
            dx *= scale; dy *= scale; dz *= scale;
            sys.vx[i] *= scale; sys.vy[i] *= scale; sys.vz[i] *= scale;
        }

        sys.x[i] += dx;
        sys.y[i] += dy;
        sys.z[i] += dz;

        // Periodic boundaries
        if (sys.x[i] < 0) sys.x[i] += L; 
        if (sys.x[i] >= L) sys.x[i] -= L;
        if (sys.y[i] < 0) sys.y[i] += L; 
        if (sys.y[i] >= L) sys.y[i] -= L;
        if (sys.z[i] < 0) sys.z[i] += L; 
        if (sys.z[i] >= L) sys.z[i] -= L;
    }
}

void integrate_langevin_2(System& sys, double dt, double friction) {
    double gamma = friction / 1000.0; 
    double c2 = 1.0 / (1.0 + gamma * dt * 0.5);

    #pragma omp parallel for
    for (size_t i = 0; i < sys.x.size(); ++i) {
        double m = ATOM_TYPES[sys.type[i]].mass;
        double acc_factor = F_TO_ACC / m;
        
        double ax = sys.fx[i] * acc_factor;
        double ay = sys.fy[i] * acc_factor;
        double az = sys.fz[i] * acc_factor;

        sys.vx[i] += c2 * dt * ax;
        sys.vy[i] += c2 * dt * ay;
        sys.vz[i] += c2 * dt * az;
    }
}

void save_frame(int step, const System& sys, double L) {
    std::string filename = "dumps/t" + std::to_string(step) + ".dump";
    std::ofstream data(filename);
    
    data << "ITEM: TIMESTEP\n" << step << "\n";
    data << "ITEM: NUMBER OF ATOMS\n" << sys.x.size() << "\n";
    data << "ITEM: BOX BOUNDS pp pp pp\n";
    data << "0.0 " << L << "\n0.0 " << L << "\n0.0 " << L << "\n";
    data << "ITEM: ATOMS id type x y z vx vy vz\n";

    for (size_t i = 0; i < sys.x.size(); ++i) {
        data << i << " " << sys.type[i] << " " 
             << sys.x[i] << " " << sys.y[i] << " " << sys.z[i] << " "
             << sys.vx[i] << " " << sys.vy[i] << " " << sys.vz[i] << "\n";
    }
    
    data.close();
}

double compute_kinetic_energy(const System& sys) {
    double ke = 0.0;
    for (size_t i = 0; i < sys.x.size(); ++i) {
        double v2 = sys.vx[i]*sys.vx[i] + 
                    sys.vy[i]*sys.vy[i] + 
                    sys.vz[i]*sys.vz[i];
        double m = ATOM_TYPES[sys.type[i]].mass;
        ke += 0.5 * m * v2;
    }
    double ke_cal = ke * KE_TO_KCAL;
    return ke_cal;
}

void randomize_velocities(System& sys, double temp) {
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < sys.x.size(); ++i) {
        double m = ATOM_TYPES[sys.type[i]].mass;
        double sigma = std::sqrt(kb * temp / (m * F_TO_ACC / 4.184e-4));
        sys.vx[i] = dist(get_rng());
        sys.vy[i] = dist(get_rng());
        sys.vz[i] = dist(get_rng());
    }
    // Rescale to exact temp
    double ke = compute_kinetic_energy(sys);
    double dof = 3.0 * (sys.x.size() - 1.0);
    double current_temp = 2.0 * ke / (dof * kb);
    double scale = std::sqrt(temp / current_temp);
    for (size_t i=0; i<sys.x.size(); ++i) {
        sys.vx[i] *= scale; sys.vy[i] *= scale; sys.vz[i] *= scale;
    }
}

struct RDF {
    std::vector<double> histogram;
    double max_r;
    double bin_width;
    int num_bins;
    int num_frames;

    RDF(double cutoff_dist, double width) {
        max_r = cutoff_dist;
        bin_width = width;
        num_bins = static_cast<int>(max_r / bin_width) + 1;
        histogram.resize(num_bins, 0.0);
        num_frames = 0;
    }

    void accumulate(const System& sys, double L) {
        num_frames++;
        
        // Collect indices of all type 1 atoms
        std::vector<int> type1_indices;
        type1_indices.reserve(sys.x.size() / 3);
        for (size_t i = 0; i < sys.x.size(); ++i) {
            if (sys.type[i] == 1) type1_indices.push_back(i);
        }

        // Loop over unique pairs of type 1 atoms
        for (size_t i = 0; i < type1_indices.size(); ++i) {
            int idx_i = type1_indices[i];
            
            for (size_t j = i + 1; j < type1_indices.size(); ++j) {
                int idx_j = type1_indices[j];

                double dx = sys.x[idx_i] - sys.x[idx_j];
                double dy = sys.y[idx_i] - sys.y[idx_j];
                double dz = sys.z[idx_i] - sys.z[idx_j];

                // Minimum Image Convention
                if (dx > L/2) dx -= L; if (dx < -L/2) dx += L;
                if (dy > L/2) dy -= L; if (dy < -L/2) dy += L;
                if (dz > L/2) dz -= L; if (dz < -L/2) dz += L;

                double r2 = dx*dx + dy*dy + dz*dz;
                double r = std::sqrt(r2);

                if (r < max_r) {
                    int bin = static_cast<int>(r / bin_width);
                    if (bin < num_bins) {
                        histogram[bin] += 2.0;
                    }
                }
            }
        }
    }

    void write_file(const std::string& filename, double L, int total_type1) {
        std::ofstream file(filename);
        file << "r,g_r\n";

        // rho = N / V
        double vol_box = L * L * L;
        double rho = total_type1 / vol_box;

        for (int i = 0; i < num_bins; ++i) {
            double r = (i + 0.5) * bin_width;
            
            // V_shell = 4 * pi * r^2 * dr
            double vol_shell = 4.0 * 3.14159 * r * r * bin_width;

            double ideal_count = rho * vol_shell;

            // Normalization
            double g_r = (histogram[i] / num_frames) / (ideal_count * total_type1);

            file << r << "," << g_r << "\n";
        }
        file.close();
        std::cout << "RDF written to " << filename << "\n";
    }
};

int main() {
    Config config;
    config.load("config.txt");
    int num_molecules = config.get_int("num_molecules", 1000);
    double L = config.get_double("box_size", 31.0);
    double dt = config.get_double("timestep", 0.5);
    double t_max = config.get_double("total_time", 5000.0);
    double target_temp = config.get_double("temperature", 300.0);
    double CUTOFF = config.get_double("cutoff", 10.0);
    double SKIN = config.get_double("skin", 2.0);
    const double CUTOFF_SQ = CUTOFF * CUTOFF;
    const double VERLET_CUTOFF = CUTOFF + SKIN;
    const double VERLET_CUTOFF_SQ = VERLET_CUTOFF * VERLET_CUTOFF;
    double friction = config.get_double("friction", 1.0);

    std::filesystem::create_directory("dumps");
    std::filesystem::create_directory("outputs");
    std::ofstream energy_file("outputs/energy.csv");
    energy_file << "Time,Kinetic,Potential,Total,Temperature\n";

    System sys;
    std::vector<Bond> bonds;
    std::vector<Angle> angles;
    init_water_box(sys, bonds, angles, num_molecules, L);

    int grid_dim = std::floor(L / VERLET_CUTOFF); 
    if (grid_dim < 3) grid_dim = 3; 
    double cell_size = L / grid_dim;

    SystemGPU gpu;
    gpu.allocate(sys.x.size(), 1000, grid_dim);

    double bonded_pe = 0.0;
    double gpu_pe = 0.0;
    double pe = 0.0;

    // Warmup phase
    std::cout << "Warmup Started.\n";
    double dt_warmup = 0.01;
    double max_force = 100.0;
    for (int i = 0; i < 2000; ++i) {
        integrate_langevin_1(sys, dt_warmup, target_temp, friction, L);
        std::fill(sys.vx.begin(), sys.vx.end(), 0.0);
        std::fill(sys.vy.begin(), sys.vy.end(), 0.0);
        std::fill(sys.vz.begin(), sys.vz.end(), 0.0);
        build_and_compute_gpu(
            sys.x.size(),
            sys.x.data(), sys.y.data(), sys.z.data(), sys.type.data(),
            sys.fx.data(), sys.fy.data(), sys.fz.data(),
            gpu, L, CUTOFF_SQ, VERLET_CUTOFF_SQ, cell_size, grid_dim, &gpu_pe, true
        );
        double bonded_pe_dummy = 0;
        compute_bond_forces(sys, bonds, bonded_pe_dummy, L);
        compute_angle_forces(sys, angles, bonded_pe_dummy, L);
    }
    std::cout << "Warmup Done.\n";

    randomize_velocities(sys, target_temp);

    RDF rdf(10.0, 0.1);

    std::cout << "Starting Simulation. \n";
    auto start_time = std::chrono::high_resolution_clock::now();
    int steps = static_cast<int>(t_max / dt);

    for (int step = 0; step < steps; ++step) {
        integrate_langevin_1(sys, dt, target_temp, friction, L);
        bool rebuild = (step % 5 == 0);
        build_and_compute_gpu(
            sys.x.size(),
            sys.x.data(), sys.y.data(), sys.z.data(), sys.type.data(),
            sys.fx.data(), sys.fy.data(), sys.fz.data(),
            gpu, L, CUTOFF_SQ, VERLET_CUTOFF_SQ, cell_size, grid_dim, &gpu_pe, rebuild
        );
        bonded_pe = 0.0;
        compute_bond_forces(sys, bonds, bonded_pe, L);
        compute_angle_forces(sys, angles, bonded_pe, L);
        integrate_langevin_2(sys, dt, friction);
        pe = gpu_pe + bonded_pe;

        if (step % 10 == 0) {
            save_frame(step, sys, L);

            double ke = compute_kinetic_energy(sys);
            double total = ke + pe;

            double dof = 3.0 * (sys.x.size() - 1.0);
            double current_temp = 2.0 * ke / (dof * kb);

            energy_file << step * dt << "," << ke << "," << pe << "," << total << "," << current_temp << "\n";

            if (step % 100 == 0) {
                if (step > 1000) {
                    rdf.accumulate(sys, L);
                }
                std::cout << "Step " << step << " / " << steps << "\r" << std::flush;
            }
        }

    }
    energy_file.close();
    int num_type1 = num_molecules;
    rdf.write_file("outputs/rdf.csv", L, num_type1);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Simulation completed in " << duration.count() << " seconds.\n";
    
    gpu.cleanup();
    return 0;
}