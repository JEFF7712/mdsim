#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <string>
#include <span>

enum class BoundaryCondition { PERIODIC, REFLECTIVE };
enum class Ensemble { NVE, NVT };

// Distance: Angstroms (Å)
// Time: femtoseconds (fs)
// Mass: atomic mass units (amu)
// Energy: kcal/mol
// Temperature: Kelvin (K)
// Force: kcal/mol/Å

// Parameters
const int num_molecules = 125; // Number of molecules
const double L = 15.53; // Box length (Angstroms)
const double dt = 0.5; // Time step (femtoseconds)
const double t_max = 5000.0; // Maximum simulation time (femtoseconds = 10 ps)
const BoundaryCondition BC = BoundaryCondition::PERIODIC; // Boundary condition type
const Ensemble ensemble = Ensemble::NVT; // Ensemble type
const double kb = 0.001987204; // Boltzmann constant (kcal/mol/K)
const double target_temp = 300.0; // Target temperature (Kelvin)

// Conversion Factor: (kcal/mol/A) / amu -> A/fs^2
const double F_TO_ACC = 4.184e-4;
const double KE_TO_KCAL = 2390.057;

// Force cutoff
const double CUTOFF = 7.0; // Cutoff distance (Angstroms)
const double CUTOFF_SQ = CUTOFF * CUTOFF;

// Random number generator
std::mt19937& get_rng() {
    static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 engine(seed);
    return engine;
}

// Atom type structure
struct AtomType {
    std::string name;
    double mass;
    double sigma;
    double epsilon;
    double charge;
};

// Bond structure
struct Bond {
    int atom1;
    int atom2;
    double k; // bond strength
    double rb; // bond length
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

// Particle structure
struct Particle {
    std::array<double, 3> position;
    std::array<double, 3> velocity;
    std::array<double, 3> force;
    int type;
    int id;
    std::vector<int> exclusions;

    Particle() : position{0,0,0}, velocity{0,0,0}, force{0,0,0}, type(0), id(-1) {}
};

std::vector<Particle> init_particles(int num_particles) {
    std::vector<Particle> particles(num_particles);
    auto& rng = get_rng();
    std::uniform_int_distribution<int> type_dist(0, ATOM_TYPES.size() - 1);
    
    for (int i = 0; i < num_particles; ++i) {
        particles[i].id = i;
        particles[i].type = type_dist(rng);
    }
    return particles;
};

void init_water_box(std::vector<Particle>& particles, std::vector<Bond>& bonds, std::vector<Angle>& angles) {
    int per_side = std::ceil(std::cbrt(num_molecules));
    double spacing = L / per_side;
    
    int p_idx = 0;
    
    for (int x = 0; x < per_side; ++x) {
        for (int y = 0; y < per_side; ++y) {
            for (int z = 0; z < per_side; ++z) {
                // Check if there is enough particles for another water molecule
                if (p_idx + 2 >= particles.size()) return;

                double cx = x * spacing + spacing/2;
                double cy = y * spacing + spacing/2;
                double cz = z * spacing + spacing/2;

                // O
                Particle oxy;
                oxy.id = p_idx;
                oxy.type = 1;
                oxy.position = {cx, cy, cz};
                
                // H1
                Particle h1;
                h1.id = p_idx + 1;
                h1.type = 0;
                h1.position = {cx + 0.6, cy + 0.6, cz};

                // H2
                Particle h2;
                h2.id = p_idx + 2;
                h2.type = 0;
                h2.position = {cx - 0.6, cy + 0.6, cz};

                // Add to list
                particles[oxy.id] = oxy;
                particles[h1.id] = h1;
                particles[h2.id] = h2;

                // Create Bonds
                bonds.push_back({oxy.id, h1.id, 450.0, 0.9572});
                bonds.push_back({oxy.id, h2.id, 450.0, 0.9572});

                // Create Angle
                double theta_eq = 104.52 * 3.14159 / 180.0;
                angles.push_back({h1.id, oxy.id, h2.id, 100.0, theta_eq});

                // Add Exclusions
                particles[oxy.id].exclusions.push_back(h1.id);
                particles[oxy.id].exclusions.push_back(h2.id);
                particles[h1.id].exclusions.push_back(oxy.id);
                particles[h2.id].exclusions.push_back(oxy.id);
                particles[h1.id].exclusions.push_back(h2.id);
                particles[h2.id].exclusions.push_back(h1.id);

                p_idx += 3;
            }
        }
    }
}

struct CellList {
    double cell_size;
    int grid_dim;
    int num_cells;

    std::vector<int> head;
    std::vector<int> next;

    CellList(double box_length, double cutoff, int num_particles) {
        // How many cells fit in the box
        grid_dim = static_cast<int>(std::floor(box_length / cutoff));
        // At least one cell
        if (grid_dim == 0) grid_dim = 1; 
        
        // Actual size of cell
        cell_size = box_length / grid_dim;

        // Total number of cells 
        num_cells = grid_dim * grid_dim * grid_dim;

        // Resize arrays to match number of cells and particles
        head.resize(num_cells);
        next.resize(num_particles);
    }

    // Convert position to cell index
    int get_cell_index(double x, double y, double z) const {

        // Wrap positions to be within box using mod
        x = std::fmod(x, L);
        if (x < 0.0) x += L;
        
        y = std::fmod(y, L);
        if (y < 0.0) y += L;
        
        z = std::fmod(z, L);
        if (z < 0.0) z += L;

        // Find cell coordinates
        int cx = static_cast<int>(x / cell_size);
        int cy = static_cast<int>(y / cell_size);
        int cz = static_cast<int>(z / cell_size);
        
        // Ensure cell indices are within bounds
        cx = std::min(cx, grid_dim - 1);
        cy = std::min(cy, grid_dim - 1);
        cz = std::min(cz, grid_dim - 1);

        // Flatten 3D index to 1D
        return cx + cy * grid_dim + cz * grid_dim * grid_dim;
    }

    void build(const std::span<Particle> particles) {

        // Fill head with -1 (-1 indicates last particle in cell)
        std::fill(head.begin(), head.end(), -1);

        // Assign each particle to a cell
        for (int i = 0; i < particles.size(); ++i) {
            int cell = get_cell_index(particles[i].position[0], particles[i].position[1], particles[i].position[2]);

            next[i] = head[cell];
            head[cell] = i;
        }
    }


};

void compute_bond_forces(std::span<Particle> particles, const std::vector<Bond>& bonds, double& pe_total) {
    for (const auto& b : bonds) {

        Particle& p1 = particles[b.atom1];
        Particle& p2 = particles[b.atom2];

        double dx = p1.position[0] - p2.position[0];
        double dy = p1.position[1] - p2.position[1];
        double dz = p1.position[2] - p2.position[2];

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
        pe_total += 0.5 * b.k * delta * delta;

        double fx = f_mag * (dx / r);
        double fy = f_mag * (dy / r);
        double fz = f_mag * (dz / r);

        p1.force[0] += fx; p1.force[1] += fy; p1.force[2] += fz;
        p2.force[0] -= fx; p2.force[1] -= fy; p2.force[2] -= fz;
    }
}

void compute_angle_forces(std::span<Particle> particles, const std::vector<Angle>& angles, double& pe_total) {
    for (const auto& ang : angles) {
        Particle& p1 = particles[ang.atom1]; // H1
        Particle& p2 = particles[ang.atom2]; // O 
        Particle& p3 = particles[ang.atom3]; // H2

        double r21x = p1.position[0] - p2.position[0];
        double r21y = p1.position[1] - p2.position[1];
        double r21z = p1.position[2] - p2.position[2];

        double r23x = p3.position[0] - p2.position[0];
        double r23y = p3.position[1] - p2.position[1];
        double r23z = p3.position[2] - p2.position[2];

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
        pe_total += 0.5 * ang.k * d_theta * d_theta;

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
        p1.force[0] += f1x; p1.force[1] += f1y; p1.force[2] += f1z;
        p3.force[0] += f3x; p3.force[1] += f3y; p3.force[2] += f3z;
        
        // Center Atom
        p2.force[0] -= (f1x + f3x);
        p2.force[1] -= (f1y + f3y);
        p2.force[2] -= (f1z + f3z);
    }
}

void compute_forces(std::span<Particle> particles, CellList& cells, double& pe_total) {
    // Build cell list with current positions
    cells.build(particles);
    
    // Get grid dimension
    int dim = cells.grid_dim;

    double total_potential = 0.0;
    
    // Loop over all cells
    for (int cx = 0; cx < dim; ++cx) {
        for (int cy = 0; cy < dim; ++cy) {
            for (int cz = 0; cz < dim; ++cz) {
                
                // 1D index of current cell
                int cell_index = cx + cy * dim + cz * dim * dim;

                // Loop over particles in this cell starting at head
                int i = cells.head[cell_index];
                while (i != -1) { 
                    
                    // Check a every cell around current cell (3x3x3 block)
                    for (int nx = cx - 1; nx <= cx + 1; ++nx) {
                        for (int ny = cy - 1; ny <= cy + 1; ++ny) {
                            for (int nz = cz - 1; nz <= cz + 1; ++nz) {
                                
                                // Periodic wrap for cell indices
                                int wrapped_nx = (nx + dim) % dim;
                                int wrapped_ny = (ny + dim) % dim;
                                int wrapped_nz = (nz + dim) % dim;
                                
                                // 1D index of neighbor cell
                                int neighbor_cell_index = wrapped_nx + wrapped_ny * dim + wrapped_nz * dim * dim;

                                // Loop over particles in neighbor cell
                                int j = cells.head[neighbor_cell_index];
                                while (j != -1) {
                                    
                                    // Avoid double counting and self-interaction
                                    if (i < j) {
                                        Particle& p1 = particles[i];
                                        Particle& p2 = particles[j];

                                        bool excluded = false;
                                        for (const auto& ex : p1.exclusions) {
                                            if (ex == p2.id) {
                                                excluded = true;
                                                break;
                                            }
                                        }

                                        if (!excluded) {
                                            double dx = p1.position[0] - p2.position[0];
                                            double dy = p1.position[1] - p2.position[1];
                                            double dz = p1.position[2] - p2.position[2];

                                            // Periodic boundary
                                            if (dx > L/2) dx -= L; 
                                            if (dx < -L/2) dx += L;
                                            if (dy > L/2) dy -= L; 
                                            if (dy < -L/2) dy += L;
                                            if (dz > L/2) dz -= L; 
                                            if (dz < -L/2) dz += L;

                                            double r2 = dx*dx + dy*dy + dz*dz;

                                            // Apply LJ and Coulomb forces
                                            if (r2 < CUTOFF_SQ && r2 > 0.0) {
                                                const AtomType& type1 = ATOM_TYPES[p1.type];
                                                const AtomType& type2 = ATOM_TYPES[p2.type];

                                                double sigma = 0.5 * (type1.sigma + type2.sigma);
                                                double epsilon = std::sqrt(type1.epsilon * type2.epsilon);
                                                
                                                
                                                double r = std::sqrt(r2);
                                                double inv_r2 = 1.0 / r2;
                                                double sig_inv_r2 = sigma * sigma * inv_r2;
                                                double sig_inv_r6 = sig_inv_r2 * sig_inv_r2 * sig_inv_r2;
                                                double sig_inv_r12 = sig_inv_r6 * sig_inv_r6;
                                                double f_lj = (24.0 * epsilon * inv_r2) * (2.0 * sig_inv_r12 - sig_inv_r6);

                                                double q1 = type1.charge;
                                                double q2 = type2.charge;
                                                double coulomb_force = (332.06 * q1 * q2) / r2;
                                                double f_elec = coulomb_force * (1.0 / r);

                                                double f_total = f_lj + f_elec;

                                                double fx = f_total * dx;
                                                double fy = f_total * dy;
                                                double fz = f_total * dz;

                                                p1.force[0] += fx; 
                                                p1.force[1] += fy; 
                                                p1.force[2] += fz;

                                                p2.force[0] -= fx; 
                                                p2.force[1] -= fy; 
                                                p2.force[2] -= fz;

                                                double u_lj = 4.0 * epsilon * (sig_inv_r12 - sig_inv_r6);
                                                double u_elec = (332.06 * type1.charge * type2.charge) / r;

                                                double inv_rc2 = 1.0 / CUTOFF_SQ;
                                                double sig_rc2 = sigma * sigma * inv_rc2;
                                                double sig_rc6 = sig_rc2 * sig_rc2 * sig_rc2;
                                                double sig_rc12 = sig_rc6 * sig_rc6;
                                                double u_shift = 4.0 * epsilon * (sig_rc12 - sig_rc6); 
                                                double u_elec_shift = (332.06 * type1.charge * type2.charge) / CUTOFF;

                                                pe_total += (u_lj - u_shift);
                                                pe_total += (u_elec - u_elec_shift);
                                            }
                                        }
                                    }
                                    j = cells.next[j];
                                }
                            }
                        }
                    }
                    i = cells.next[i];
                }
            }
        }
    }
}

// Periodic boundary condition
void apply_periodic_bc(Particle& p) {
    for (int i = 0; i < 3; ++i) {
        if (p.position[i] < 0.0) {
            p.position[i] += L;
        } else if (p.position[i] >= L) {
            p.position[i] -= L;
        }
    }
}

// Reflective boundary condition
void apply_reflective_bc(Particle& p) {
    for (int i = 0; i < 3; ++i) {
        if (p.position[i] < 0.0) {
            p.position[i] = -p.position[i];
            p.velocity[i] = -p.velocity[i];
        } else if (p.position[i] > L) {
            p.position[i] = 2 * L - p.position[i];
            p.velocity[i] = -p.velocity[i];
        }
    }
}

void verlet_first_step(std::span<Particle> particles, double dt) {
    for (auto& p : particles) {
        double m = ATOM_TYPES[p.type].mass;
        double acc_factor = F_TO_ACC / m;

        for (int i = 0; i < 3; ++i) {
            p.velocity[i] += 0.5 * acc_factor * p.force[i] * dt;
            p.position[i] += p.velocity[i] * dt;
        }

        if (BC == BoundaryCondition::PERIODIC) {
            apply_periodic_bc(p);
        }
        else {
            apply_reflective_bc(p);
        }
    }
}

void verlet_second_step(std::span<Particle> particles, double dt) {
    for (auto& p : particles) {
        double m = ATOM_TYPES[p.type].mass;
        double acc_factor = F_TO_ACC / m;

        for (int i = 0; i < 3; ++i) {
            p.velocity[i] += 0.5 * acc_factor * p.force[i] * dt;
        }
    }
}

void save_frame(int step, const std::vector<Particle>& particles) {
    std::string filename = "dumps/t" + std::to_string(step) + ".dump";
    std::ofstream data(filename);
    
    data << "ITEM: TIMESTEP\n" << step << "\n";
    data << "ITEM: NUMBER OF ATOMS\n" << particles.size() << "\n";
    data << "ITEM: BOX BOUNDS pp pp pp\n";
    data << "0.0 " << L << "\n0.0 " << L << "\n0.0 " << L << "\n";
    data << "ITEM: ATOMS id type x y z vx vy vz\n";

    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        data << i << " " << p.type << " " 
             << p.position[0] << " " << p.position[1] << " " << p.position[2] << " "
             << p.velocity[0] << " " << p.velocity[1] << " " << p.velocity[2] << "\n";
    }
    
    data.close();
}

double compute_kinetic_energy(const std::vector<Particle>& particles) {
    double ke = 0.0;
    for (const auto& p : particles) {
        double v2 = p.velocity[0]*p.velocity[0] + 
                    p.velocity[1]*p.velocity[1] + 
                    p.velocity[2]*p.velocity[2];
        double m = ATOM_TYPES[p.type].mass;
        ke += 0.5 * m * v2;
    }
    double ke_cal = ke * KE_TO_KCAL;
    return ke_cal;
}

void rescale_velocities(std::vector<Particle>& particles, double target_temp) {
    double ke = compute_kinetic_energy(particles);
    double dof = 3.0 * (particles.size() - 1.0);
    double current_temp = 2.0 * ke / (dof * kb);
    double scale_factor = std::sqrt(target_temp / current_temp);

    for (auto& p : particles) {
        for (int i = 0; i < 3; ++i) {
            p.velocity[i] *= scale_factor;
        }
    }
}

int main() {
    std::filesystem::create_directory("dumps");

    std::ofstream energy_file("energy.csv");
    energy_file << "Time,Kinetic,Potential,Total,Temperature\n";

    std::vector<Particle> particles(num_molecules * 3);
    std::vector<Bond> bonds;
    std::vector<Angle> angles;
    init_water_box(particles, bonds, angles);
    std::cout << "Water box initialized. Created " << bonds.size() << " bonds.\n";

    CellList cells(L, CUTOFF, particles.size());
    std::cout << "Starting Simulation with " << particles.size() << " particles.\n";
    
    double pe = 0.0;
    std::cout << "Computing initial forces...\n";
    compute_bond_forces(particles, bonds, pe);
    compute_angle_forces(particles, angles, pe);
    compute_forces(particles, cells, pe);

    std::cout << "Initial forces computed.\n";

    std::cout << "Minimizing energy...\n";
    double dt_warmup = 0.01;
    double max_force = 100.0;
    for (int i = 0; i < 100; ++i) {
        verlet_first_step(particles, dt_warmup);
        for(auto& p : particles) {
            p.force = {0,0,0};
        }
        
        double pe_dummy = 0;
        compute_bond_forces(particles, bonds, pe_dummy);
        compute_angle_forces(particles, angles, pe_dummy);
        compute_forces(particles, cells, pe_dummy);
        
        // Cap forces to avoid instability
        for(auto& p : particles) {
            for(int k=0; k<3; ++k) {
                if (p.force[k] > max_force) p.force[k] = max_force;
                if (p.force[k] < -max_force) p.force[k] = -max_force;
            }
        }
        verlet_second_step(particles, dt_warmup);
        if (i % 50 == 0) max_force += 50.0;
    }
    std::cout << "Warmup done. Starting main run.\n";

    for(auto& p : particles) {
        p.force = {0,0,0};
    }
    rescale_velocities(particles, target_temp);

    int steps = static_cast<int>(t_max / dt);
    for (int step = 0; step < steps; ++step) {
        verlet_first_step(particles, dt);
        for (auto& p : particles) {
            p.force = {0.0, 0.0, 0.0};
        }
        pe = 0.0;
        compute_bond_forces(particles, bonds, pe);
        compute_angle_forces(particles, angles, pe);
        compute_forces(particles, cells, pe);
        verlet_second_step(particles, dt);

        if (step % 10 == 0) {

            if (ensemble == Ensemble::NVT) {
                rescale_velocities(particles, target_temp);
            }

            save_frame(step, particles);

            double ke = compute_kinetic_energy(particles);
            double total = ke + pe;

            double dof = 3.0 * (particles.size() - 1.0);
            double current_temp = 2.0 * ke / (dof * kb);

            energy_file << step * dt << "," << ke << "," << pe << "," << total << "," << current_temp << "\n";

            if (step % 100 == 0) {
                std::cout << "Step " << step << " / " << steps << "\r" << std::flush;
            }
        }

    }
    energy_file.close();
    std::cout << "\nDone.\n";
    return 0;
}