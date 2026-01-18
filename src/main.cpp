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

const std::vector<AtomType> ATOM_TYPES = {
    {"H", 3.024, 0.4000, 0.0460, 0.417},          // Type 0: Hydrogen (no LJ interactions)
    {"O", 11.967, 3.1506, 0.1521, -0.834},  // Type 1: Oxygen
    {"Ar", 39.948, 3.405, 0.238, 0.0},      // Type 2: Argon
    {"C_meth", 12.011, 3.50, 0.066, -0.24}, // Type 3: Carbon in Methane
    {"H_meth", 1.008, 2.50, 0.030, 0.06}    // Type 4: Hydrogen in Methane
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

struct System {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> fx, fy, fz;
    std::vector<int> type;
    std::vector<int> id;
    std::vector<std::vector<int>> exclusions;

    void resize(int n) {
        x.resize(n); y.resize(n); z.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
        fx.resize(n, 0.0f); fy.resize(n, 0.0f); fz.resize(n, 0.0f);
        type.resize(n);
        id.resize(n);
        exclusions.resize(n);
    }
};

void init_water_box(System& sys, std::vector<Bond>& bonds, std::vector<Angle>& angles, int num_molecules, float L) {
    sys.resize(num_molecules * 3);
    int p_idx = 0;

    const float r0 = 0.9572f;
    const float theta_rad = 104.52f * 3.14159265f / 180.0f;
    const float h_x = r0 * sinf(theta_rad / 2.0f);
    const float h_y = r0 * cosf(theta_rad / 2.0f);

    std::uniform_real_distribution<float> pos_dist(0.0f, L);
    std::uniform_real_distribution<float> angle_dist(0.0f, 6.28f);

    float min_dist_sq = 2.5f * 2.5f; 

    for (int m = 0; m < num_molecules; ++m) {
        float ox, oy, oz;
        bool valid = false;
        int attempts = 0;

        while (!valid && attempts < 1000) {
            valid = true;
            ox = pos_dist(get_rng());
            oy = pos_dist(get_rng());
            oz = pos_dist(get_rng());

            for (int k = 0; k < m; ++k) {
                int other_o_idx = k * 3;
                float dx = ox - sys.x[other_o_idx];
                float dy = oy - sys.y[other_o_idx];
                float dz = oz - sys.z[other_o_idx];
                
                if (dx > L/2) dx -= L; if (dx < -L/2) dx += L;
                if (dy > L/2) dy -= L; if (dy < -L/2) dy += L;
                if (dz > L/2) dz -= L; if (dz < -L/2) dz += L;

                if (dx*dx + dy*dy + dz*dz < min_dist_sq) {
                    valid = false;
                    break;
                }
            }
            attempts++;
        }

        float phi = angle_dist(get_rng());
        float c = cosf(phi);
        float s = sinf(phi);

        sys.id[p_idx] = p_idx; sys.type[p_idx] = 1;
        sys.x[p_idx] = ox; sys.y[p_idx] = oy; sys.z[p_idx] = oz;

        sys.id[p_idx+1] = p_idx+1; sys.type[p_idx+1] = 0;
        sys.x[p_idx+1] = ox + (h_x * c - h_y * s);
        sys.y[p_idx+1] = oy + (h_x * s + h_y * c);
        sys.z[p_idx+1] = oz;

        sys.id[p_idx+2] = p_idx+2; sys.type[p_idx+2] = 0;
        sys.x[p_idx+2] = ox + (-h_x * c - h_y * s);
        sys.y[p_idx+2] = oy + (-h_x * s + h_y * c);
        sys.z[p_idx+2] = oz;

        bonds.push_back({p_idx, p_idx+1, 450.0, 0.9572});
        bonds.push_back({p_idx, p_idx+2, 450.0, 0.9572});
        angles.push_back({p_idx+1, p_idx, p_idx+2, 300.0, theta_rad});

        sys.exclusions[p_idx].push_back(p_idx+1); sys.exclusions[p_idx].push_back(p_idx+2);
        sys.exclusions[p_idx+1].push_back(p_idx); sys.exclusions[p_idx+1].push_back(p_idx+2);
        sys.exclusions[p_idx+2].push_back(p_idx); sys.exclusions[p_idx+2].push_back(p_idx+1);

        p_idx += 3;
    }
}

void init_argon_box(System& sys, int num_atoms, float L) {
    int per_side = std::ceil(std::cbrt(num_atoms));
    float spacing = L / per_side;
    sys.resize(num_atoms);
    
    int p_idx = 0;
    for (int x = 0; x < per_side; ++x) {
        for (int y = 0; y < per_side; ++y) {
            for (int z = 0; z < per_side; ++z) {
                if (p_idx >= num_atoms) return;
                
                sys.id[p_idx] = p_idx;
                sys.type[p_idx] = 2;
                sys.x[p_idx] = x * spacing + spacing/2;
                sys.y[p_idx] = y * spacing + spacing/2;
                sys.z[p_idx] = z * spacing + spacing/2;
                
                p_idx++;
            }
        }
    }
}

void init_methane_box(System& sys, std::vector<Bond>& bonds, std::vector<Angle>& angles, int num_molecules, float L) {
    sys.resize(num_molecules * 5);
    int idx = 0;
    
    // Grid layout
    int per_side = std::ceil(std::cbrt(num_molecules));
    float spacing = L / per_side;

    for (int x = 0; x < per_side; ++x) {
        for (int y = 0; y < per_side; ++y) {
            for (int z = 0; z < per_side; ++z) {
                if (idx + 4 >= sys.x.size()) return;

                float cx = x * spacing + spacing/2;
                float cy = y * spacing + spacing/2;
                float cz = z * spacing + spacing/2;

                int id_c = idx;
                sys.type[id_c] = 3; 
                sys.x[id_c] = cx; sys.y[id_c] = cy; sys.z[id_c] = cz;

                float d = 1.09f / sqrtf(3.0f); 
                
                int id_h1 = idx+1; int id_h2 = idx+2; 
                int id_h3 = idx+3; int id_h4 = idx+4;

                sys.type[id_h1] = 4; sys.x[id_h1] = cx+d; sys.y[id_h1] = cy+d; sys.z[id_h1] = cz+d;
                sys.type[id_h2] = 4; sys.x[id_h2] = cx-d; sys.y[id_h2] = cy-d; sys.z[id_h2] = cz+d;
                sys.type[id_h3] = 4; sys.x[id_h3] = cx-d; sys.y[id_h3] = cy+d; sys.z[id_h3] = cz-d;
                sys.type[id_h4] = 4; sys.x[id_h4] = cx+d; sys.y[id_h4] = cy-d; sys.z[id_h4] = cz-d;

                double bond_k = 340.0;
                double bond_len = 1.09;
                bonds.push_back({id_c, id_h1, bond_k, bond_len});
                bonds.push_back({id_c, id_h2, bond_k, bond_len});
                bonds.push_back({id_c, id_h3, bond_k, bond_len});
                bonds.push_back({id_c, id_h4, bond_k, bond_len});

                double ang_k = 35.0;
                double ang_eq = 109.5 * 3.14159 / 180.0;
                
                angles.push_back({id_h1, id_c, id_h2, ang_k, ang_eq});
                angles.push_back({id_h1, id_c, id_h3, ang_k, ang_eq});
                angles.push_back({id_h1, id_c, id_h4, ang_k, ang_eq});
                angles.push_back({id_h2, id_c, id_h3, ang_k, ang_eq});
                angles.push_back({id_h2, id_c, id_h4, ang_k, ang_eq});
                angles.push_back({id_h3, id_c, id_h4, ang_k, ang_eq});

                for (int i = 0; i < 5; ++i) {
                    for (int j = 0; j < 5; ++j) {
                        if (i != j) {
                            sys.exclusions[idx + i].push_back(idx + j);
                        }
                    }
                }

                idx += 5;
            }
        }
    }
}

void compute_bond_forces(System& sys, const std::vector<Bond>& bonds, double& pe_total, float L) {
    double local_pe = 0.0;
    
    #pragma omp parallel for reduction(+:local_pe)
    for (const auto& b : bonds) {

        float dx = sys.x[b.atom1] - sys.x[b.atom2];
        float dy = sys.y[b.atom1] - sys.y[b.atom2];
        float dz = sys.z[b.atom1] - sys.z[b.atom2];

        if (dx > L/2) dx -= L; 
        if (dx < -L/2) dx += L;
        if (dy > L/2) dy -= L; 
        if (dy < -L/2) dy += L;
        if (dz > L/2) dz -= L; 
        if (dz < -L/2) dz += L;

        float r2 = dx*dx + dy*dy + dz*dz;
        float r = std::sqrt(r2);
        if (r < 1e-6f) r = 1e-6f;

        float delta = r - static_cast<float>(b.rb);
        float f_mag = -static_cast<float>(b.k) * delta;

        const float MAX_FORCE = 1000.0f;
        if (f_mag > MAX_FORCE) f_mag = MAX_FORCE;
        if (f_mag < -MAX_FORCE) f_mag = -MAX_FORCE;

        local_pe += 0.5f * static_cast<float>(b.k) * delta * delta;

        float fx = f_mag * (dx / r);
        float fy = f_mag * (dy / r);
        float fz = f_mag * (dz / r);

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

void compute_angle_forces(System& sys, const std::vector<Angle>& angles, double& pe_total, float L) {
    double local_pe = 0.0;
    
    #pragma omp parallel for reduction(+:local_pe)
    for (const auto& ang : angles) {
        float r21x = sys.x[ang.atom1] - sys.x[ang.atom2];
        float r21y = sys.y[ang.atom1] - sys.y[ang.atom2];
        float r21z = sys.z[ang.atom1] - sys.z[ang.atom2];

        float r23x = sys.x[ang.atom3] - sys.x[ang.atom2];
        float r23y = sys.y[ang.atom3] - sys.y[ang.atom2];
        float r23z = sys.z[ang.atom3] - sys.z[ang.atom2];

        if (r21x > L/2) r21x -= L; if (r21x < -L/2) r21x += L;
        if (r21y > L/2) r21y -= L; if (r21y < -L/2) r21y += L;
        if (r21z > L/2) r21z -= L; if (r21z < -L/2) r21z += L;

        if (r23x > L/2) r23x -= L; if (r23x < -L/2) r23x += L;
        if (r23y > L/2) r23y -= L; if (r23y < -L/2) r23y += L;
        if (r23z > L/2) r23z -= L; if (r23z < -L/2) r23z += L;

        float r21_sq = r21x*r21x + r21y*r21y + r21z*r21z;
        float r23_sq = r23x*r23x + r23y*r23y + r23z*r23z;
        float r21 = std::sqrt(r21_sq);
        float r23 = std::sqrt(r23_sq);

        if (r21 < 1e-6f || r23 < 1e-6f) continue;

        float dot = r21x*r23x + r21y*r23y + r21z*r23z;
        float cos_theta = dot / (r21 * r23);

        if (cos_theta > 1.0f) cos_theta = 1.0f;
        if (cos_theta < -1.0f) cos_theta = -1.0f;

        float theta = std::acos(cos_theta);
        float d_theta = theta - static_cast<float>(ang.theta0);
        local_pe += 0.5f * static_cast<float>(ang.k) * d_theta * d_theta;

        float sin_theta = std::sin(theta);
        if (std::abs(sin_theta) < 1e-6f) sin_theta = 1e-6f;
        
        float force_factor = -static_cast<float>(ang.k) * d_theta / sin_theta;

        const float MAX_ANGLE_FORCE = 1000000.0f; 
        if (force_factor > MAX_ANGLE_FORCE) force_factor = MAX_ANGLE_FORCE;
        if (force_factor < -MAX_ANGLE_FORCE) force_factor = -MAX_ANGLE_FORCE;

        float rr21 = 1.0f / r21_sq;
        float rr23 = 1.0f / r23_sq;
        float rr21_23 = 1.0f / (r21 * r23);

        float f1x = force_factor * (cos_theta * rr21 * r21x - rr21_23 * r23x);
        float f1y = force_factor * (cos_theta * rr21 * r21y - rr21_23 * r23y);
        float f1z = force_factor * (cos_theta * rr21 * r21z - rr21_23 * r23z);

        float f3x = force_factor * (cos_theta * rr23 * r23x - rr21_23 * r21x);
        float f3y = force_factor * (cos_theta * rr23 * r23y - rr21_23 * r21y);
        float f3z = force_factor * (cos_theta * rr23 * r23z - rr21_23 * r21z);

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
        
        #pragma omp atomic
        sys.fx[ang.atom2] -= (f1x + f3x);
        #pragma omp atomic
        sys.fy[ang.atom2] -= (f1y + f3y);
        #pragma omp atomic
        sys.fz[ang.atom2] -= (f1z + f3z);
    }
    pe_total += local_pe;
}

void integrate_langevin_1(System& sys, double dt, float temp, float friction, float L) {
    const float MAX_DISP = 0.5f;
    float dt_f = static_cast<float>(dt);
    
    float gamma = friction / 1000.0f;
    
    float c1 = expf(-gamma * dt_f);
    float c2 = sqrtf(1.0f - c1*c1);

    std::normal_distribution<float> noise(0.0f, 1.0f);
    auto& rng = get_rng();

    #pragma omp parallel for
    for (size_t i = 0; i < sys.x.size(); ++i) {
        float m = static_cast<float>(ATOM_TYPES[sys.type[i]].mass);
        
        float acc_factor = static_cast<float>(F_TO_ACC) / m; 
        float ax = sys.fx[i] * acc_factor;
        float ay = sys.fy[i] * acc_factor;
        float az = sys.fz[i] * acc_factor;

        sys.vx[i] += 0.5f * dt_f * ax;
        sys.vy[i] += 0.5f * dt_f * ay;
        sys.vz[i] += 0.5f * dt_f * az;

        float v_thermal_sq = (static_cast<float>(kb) * temp) / (m * static_cast<float>(KE_TO_KCAL));
        float v_thermal = sqrtf(v_thermal_sq);

        sys.vx[i] = c1 * sys.vx[i] + c2 * v_thermal * noise(rng);
        sys.vy[i] = c1 * sys.vy[i] + c2 * v_thermal * noise(rng);
        sys.vz[i] = c1 * sys.vz[i] + c2 * v_thermal * noise(rng);

        float dx = sys.vx[i] * dt_f;
        float dy = sys.vy[i] * dt_f;
        float dz = sys.vz[i] * dt_f;

        float dist_sq = dx*dx + dy*dy + dz*dz;
        if (dist_sq > MAX_DISP * MAX_DISP) {
            float scale = MAX_DISP / std::sqrt(dist_sq);
            dx *= scale; dy *= scale; dz *= scale;
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

void integrate_langevin_2(System& sys, double dt, float friction) {
    float dt_f = static_cast<float>(dt);

    #pragma omp parallel for
    for (size_t i = 0; i < sys.x.size(); ++i) {
        float m = static_cast<float>(ATOM_TYPES[sys.type[i]].mass);
        
        float acc_factor = static_cast<float>(F_TO_ACC) / m; 
        float ax = sys.fx[i] * acc_factor;
        float ay = sys.fy[i] * acc_factor;
        float az = sys.fz[i] * acc_factor;

        sys.vx[i] += 0.5f * dt_f * ax;
        sys.vy[i] += 0.5f * dt_f * ay;
        sys.vz[i] += 0.5f * dt_f * az;
    }
}

void remove_com_motion(System& sys) {
    double total_mass = 0.0;
    double px = 0.0, py = 0.0, pz = 0.0;
    
    for (size_t i = 0; i < sys.x.size(); ++i) {
        double m = ATOM_TYPES[sys.type[i]].mass;
        total_mass += m;
        px += m * sys.vx[i];
        py += m * sys.vy[i];
        pz += m * sys.vz[i];
    }
    
    float v_cm_x = static_cast<float>(px / total_mass);
    float v_cm_y = static_cast<float>(py / total_mass);
    float v_cm_z = static_cast<float>(pz / total_mass);
    
    for (size_t i = 0; i < sys.x.size(); ++i) {
        sys.vx[i] -= v_cm_x;
        sys.vy[i] -= v_cm_y;
        sys.vz[i] -= v_cm_z;
    }
}

void save_frame(int step, const System& sys, float L) {
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

float compute_kinetic_energy(const System& sys) {
    float ke = 0.0f;
    for (size_t i = 0; i < sys.x.size(); ++i) {
        float v2 = sys.vx[i]*sys.vx[i] + 
                    sys.vy[i]*sys.vy[i] + 
                    sys.vz[i]*sys.vz[i];
        float m = static_cast<float>(ATOM_TYPES[sys.type[i]].mass);
        ke += 0.5f * m * v2;
    }
    float ke_cal = ke * static_cast<float>(KE_TO_KCAL);
    return ke_cal;
}

void randomize_velocities(System& sys, float temp) {
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < sys.x.size(); ++i) {
        double m = ATOM_TYPES[sys.type[i]].mass;
        double sigma = std::sqrt(kb * temp / (m * KE_TO_KCAL));
        sys.vx[i] = static_cast<float>(sigma * dist(get_rng()));
        sys.vy[i] = static_cast<float>(sigma * dist(get_rng()));
        sys.vz[i] = static_cast<float>(sigma * dist(get_rng()));
    }
    float ke = compute_kinetic_energy(sys);
    float dof = 3.0f * (sys.x.size() - 1.0f);
    float current_temp = 2.0f * ke / (dof * static_cast<float>(kb));
    float scale = std::sqrt(temp / current_temp);
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

    void accumulate(const System& sys, float L, int type) {
        num_frames++;
        
        // Collect indices of all atoms of the specified type
        std::vector<int> type_indices;
        type_indices.reserve(sys.x.size() / 3);
        for (size_t i = 0; i < sys.x.size(); ++i) {
            if (sys.type[i] == type) type_indices.push_back(i);
        }

        // Loop over unique pairs of atoms of the specified type
        for (size_t i = 0; i < type_indices.size(); ++i) {
            int idx_i = type_indices[i];
            
            for (size_t j = i + 1; j < type_indices.size(); ++j) {
                int idx_j = type_indices[j];

                float dx = sys.x[idx_i] - sys.x[idx_j];
                float dy = sys.y[idx_i] - sys.y[idx_j];
                float dz = sys.z[idx_i] - sys.z[idx_j];

                // Minimum Image Convention
                if (dx > L/2) dx -= L; if (dx < -L/2) dx += L;
                if (dy > L/2) dy -= L; if (dy < -L/2) dy += L;
                if (dz > L/2) dz -= L; if (dz < -L/2) dz += L;

                float r2 = dx*dx + dy*dy + dz*dz;
                float r = std::sqrt(r2);

                if (r < max_r) {
                    int bin = static_cast<int>(r / bin_width);
                    if (bin < num_bins) {
                        histogram[bin] += 2.0;
                    }
                }
            }
        }
    }

    void write_file(const std::string& filename, float L, int total_type1) {
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
    std::string system_type = config.get_string("system_type", "water");
    int num_molecules = config.get_int("num_molecules", 1000);
    double dt = config.get_double("timestep", 0.5);
    double t_max = config.get_double("total_time", 5000.0);
    float target_temp = static_cast<float>(config.get_double("temperature", 300.0));
    float CUTOFF = static_cast<float>(config.get_double("cutoff", 10.0));
    float SKIN = static_cast<float>(config.get_double("skin", 2.0));
    float CUTOFF_SQ = CUTOFF * CUTOFF;
    float VERLET_CUTOFF = CUTOFF + SKIN;
    float VERLET_CUTOFF_SQ = VERLET_CUTOFF * VERLET_CUTOFF;
    float friction = static_cast<float>(config.get_double("friction", 1.0));
    float volume_per_molecule = static_cast<float>(config.get_double("volume_per_molecule", 40.0));
    float target_volume = num_molecules * volume_per_molecule;
    float L = std::cbrt(target_volume);
    if (L < 3.0f * VERLET_CUTOFF) {
        VERLET_CUTOFF = L / 3.001f;
        CUTOFF = VERLET_CUTOFF - SKIN;
        CUTOFF_SQ = CUTOFF * CUTOFF;
        VERLET_CUTOFF_SQ = VERLET_CUTOFF * VERLET_CUTOFF;
    }
    std::cout << "System Type: " << system_type << "\n";
    std::cout << "Number of Molecules: " << num_molecules << "\n";
    std::cout << "Box Side L: " << L << " Angstroms\n";

    std::filesystem::create_directory("dumps");
    std::filesystem::create_directory("outputs");
    std::ofstream energy_file("outputs/energy.csv");
    energy_file << "Time,Kinetic,Potential,Total,Temperature\n";

    System sys;
    std::vector<Bond> bonds;
    std::vector<Angle> angles;
    int atoms_per_molecule = 0;
    int type = 0;
    if (system_type == "water") {
        init_water_box(sys, bonds, angles, num_molecules, L);
        atoms_per_molecule = 3;
        type = 1; // Oxygen
    } else if (system_type == "argon") {
        init_argon_box(sys, num_molecules, L);
        atoms_per_molecule = 1;
        type = 2; // Argon
    } else if (system_type == "methane") {
        init_methane_box(sys, bonds, angles, num_molecules, L);
        atoms_per_molecule = 5;
        type = 3; // Carbon
    } else {
        std::cerr << "Unsupported system type: " << system_type << "\n";
        return 1;
    }

    int grid_dim = std::floor(L / VERLET_CUTOFF); 
    if (grid_dim < 3) grid_dim = 3; 
    float cell_size = L / grid_dim;

    SystemGPU gpu;
    int num_types = ATOM_TYPES.size();
    std::vector<float> h_sigma(num_types);
    std::vector<float> h_epsilon(num_types);
    std::vector<float> h_charge(num_types);
    for (int i = 0; i < num_types; ++i) {
        h_sigma[i]   = static_cast<float>(ATOM_TYPES[i].sigma);
        h_epsilon[i] = static_cast<float>(ATOM_TYPES[i].epsilon);
        h_charge[i]  = static_cast<float>(ATOM_TYPES[i].charge);
    }
    gpu.set_atom_params(num_types, h_sigma.data(), h_epsilon.data(), h_charge.data());
    int safe_max_neighbors = (grid_dim <= 3) ? sys.x.size() : 1000;
    if (safe_max_neighbors < 1000) safe_max_neighbors = 1000;
    gpu.allocate(sys.x.size(), safe_max_neighbors, grid_dim);
    gpu.set_exclusions(sys.x.size(), sys.exclusions);
    int N = sys.x.size();
    float gpu_pe_f = 0.0f;
    double gpu_pe = 0.0;
    double bonded_pe = 0.0;
    double pe = 0.0;
    gpu.set_atom_params(num_types, h_sigma.data(), h_epsilon.data(), h_charge.data());

    int warmup_steps = 1000;
    double warmup_dt = 0.5;
    float warmup_friction = 50.0f;
    std::cout << "Starting Warmup.\n";
    for (int step = 0; step < warmup_steps; ++step) {
        integrate_langevin_1(sys, warmup_dt, target_temp, warmup_friction, L);
        bool rebuild = (step % 20 == 0);
        build_and_compute_gpu(
            N,
            sys.x.data(), sys.y.data(), sys.z.data(), sys.type.data(),
            sys.fx.data(), sys.fy.data(), sys.fz.data(),
            gpu, L, CUTOFF_SQ, VERLET_CUTOFF_SQ, cell_size, grid_dim, atoms_per_molecule, &gpu_pe_f, rebuild, sys.exclusions
        );
        gpu_pe = static_cast<double>(gpu_pe_f);
        bonded_pe = 0.0;
        compute_bond_forces(sys, bonds, bonded_pe, L);
        compute_angle_forces(sys, angles, bonded_pe, L);
        integrate_langevin_2(sys, warmup_dt, warmup_friction);
    }
    std::cout << "Warmup done.\n";
    std::cout << "Starting Relaxation Phase.\n";
    float relax_dt = 0.05f;
    float relax_fric = 20.0f;
    
    for (int i = 0; i < 2000; ++i) {
        integrate_langevin_1(sys, relax_dt, target_temp, relax_fric, L);
        bool rebuild = (i % 20 == 0);
        build_and_compute_gpu(
            N, sys.x.data(), sys.y.data(), sys.z.data(), sys.type.data(),
            sys.fx.data(), sys.fy.data(), sys.fz.data(),
            gpu, L, CUTOFF_SQ, VERLET_CUTOFF_SQ, cell_size, grid_dim, atoms_per_molecule, 
            &gpu_pe_f, rebuild, sys.exclusions
        );
        double dummy = 0.0;
        compute_bond_forces(sys, bonds, dummy, L);
        compute_angle_forces(sys, angles, dummy, L);
        integrate_langevin_2(sys, relax_dt, relax_fric);
        if (i % 100 == 0) {
            remove_com_motion(sys); 
            std::cout << "Relaxing: " << i << "/2000 \r" << std::flush;
        }
    }
    std::cout << "\nRelaxation Complete.\n";
    RDF rdf(10.0, 0.1);
    randomize_velocities(sys, target_temp);
    std::cout << "Starting Simulation. \n";
    auto start_time = std::chrono::high_resolution_clock::now();
    int steps = static_cast<int>(t_max / dt);
    double current_time = 0.0;

    for (int step = 0; step < steps; ++step) {
        integrate_langevin_1(sys, dt, target_temp, friction, L);
        bool rebuild = (step % 20 == 0);
        build_and_compute_gpu(
            N,
            sys.x.data(), sys.y.data(), sys.z.data(), sys.type.data(),
            sys.fx.data(), sys.fy.data(), sys.fz.data(),
            gpu, L, CUTOFF_SQ, VERLET_CUTOFF_SQ, cell_size, grid_dim, atoms_per_molecule, &gpu_pe_f, rebuild, sys.exclusions
        );
        gpu_pe = static_cast<double>(gpu_pe_f);
        double bonded_pe = 0.0;
        compute_bond_forces(sys, bonds, bonded_pe, L);
        compute_angle_forces(sys, angles, bonded_pe, L);
        integrate_langevin_2(sys, dt, friction);
        pe = gpu_pe + bonded_pe;
        current_time += dt;
        if (step % 10 == 0) {
            double ke = static_cast<double>(compute_kinetic_energy(sys));
            double total = ke + pe;
            double dof = 3.0 * (sys.x.size() - 1.0);
            double current_temp = 2.0 * ke / (dof * kb);
            energy_file << current_time << "," << ke << "," << pe << "," << total << "," << current_temp << "\n";
            if (step % 100 == 0) {
                remove_com_motion(sys);
                save_frame(step, sys, L);
                if (step > steps*0.8) rdf.accumulate(sys, L, type);
                std::cout << "Step " << step << " / " << steps << "\r" << std::flush;
                if (step % 1000 == 0) {
                    double total_force = 0.0;
                    double total_vel = 0.0;
                    int active_neighbors = 0;
                    for(int i=0; i<N; ++i) {
                        total_force += sqrt(sys.fx[i]*sys.fx[i] + sys.fy[i]*sys.fy[i] + sys.fz[i]*sys.fz[i]);
                        total_vel += sqrt(sys.vx[i]*sys.vx[i] + sys.vy[i]*sys.vy[i] + sys.vz[i]*sys.vz[i]);
                    }
                    
                    std::cout << " [DEBUG] Avg Force: " << total_force / N << " | Avg Vel: " << total_vel / N << " | T: " << current_temp << "\n";
                }
            }
        }

    }
    energy_file.close();
    int num_type = 0;
    for (size_t i = 0; i < sys.x.size(); ++i) {
        if (sys.type[i] == type) num_type++;
    }
    rdf.write_file("outputs/rdf.csv", L, num_type);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Simulation completed in " << duration.count() << " seconds.\n";
    
    gpu.cleanup();
    return 0;
}
