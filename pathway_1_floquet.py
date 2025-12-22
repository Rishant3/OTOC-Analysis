# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import expm



def sech(x):
    return 1/(np.cosh(x))

def build_Grid(L, N):
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    return x, dx

def volcano_potential(x, a1, a2, nu):
    cosh_term = np.cosh(x)
    sech_term = sech(x)
    # Base static potential V_0(x)
    return -1*((a1 * (cosh_term)**(2 * nu)) + (a2 * (sech_term)**2))

def solve_schrodinger(x, dx, V_array):
    size = len(x)
    array = np.ones(size)
    array_off = np.ones(size - 1)
    
    V_matrice = np.diag(V_array)
    # Kinetic Energy Matrix (Finite Difference)
    T_matrice = (2*np.diag(array, k = 0) + -1*np.diag(array_off, k = 1) + -1*np.diag(array_off, k = -1)) / (2 * dx * dx)
    H_matrice = T_matrice + V_matrice
    
    eigenvalue, eigenvector = np.linalg.eigh(H_matrice)
    return eigenvalue, eigenvector

def x_pos_basis(x):
    return np.diag(x)

def P_pos_basis(x, dx):
    size = len(x)
    array = np.ones(size)
    array_off = np.ones(size - 1)
    p_matrice = -1j * (np.diag(array_off, k = 1) - np.diag(array_off, k = -1)) / (2 * dx)
    return p_matrice

def energy_matrices(x, dx, nu, a1, a2):
    V_array = volcano_potential(x, a1, a2, nu)
    x_pos_matrice = x_pos_basis(x)
    P_pos_matrice = P_pos_basis(x, dx)
    
    # Solve static H_0
    eigenvalue, eigenvector = solve_schrodinger(x, dx, V_array)

    # Transform operators to Energy Basis of H_0
    x_matrice = eigenvector.T @ x_pos_matrice @ eigenvector
    p_matrice = eigenvector.T @ P_pos_matrice @ eigenvector

    return x_matrice, p_matrice, eigenvalue



def calculate_otoc_from_unitary(U_t, x_matrice_0, p_matrice_0, n):
    """
    Calculates OTOC C(t) given the accumulated Unitary operator U_t.
    """
    # Heisenberg evolution: x(t) = U_t^dagger * x(0) * U_t
    x_t = U_t.conj().T @ x_matrice_0 @ U_t

    # Commutator c(t) = [x(t), p(0)]
    comm = x_t @ p_matrice_0 - p_matrice_0 @ x_t
    
    # OTOC Intensity: < |c(t)|^2 >
    intensity = comm.conj().T @ comm
    
    # Return element (n,n) - expectation value in n-th eigenstate
    return intensity[n, n]

def run_floquet_simulation(N, t_max, n_state, nu, a1, a2, drive_amp, drive_freq, steps=300):
    """
    Runs the simulation with a time-periodic driving force (Floquet).
    H(t) = H_0 + x * A * cos(omega * t)
    
    Args:
        drive_amp (float): Amplitude of driving field (A)
        drive_freq (float): Frequency of driving field (omega)
    """
    print(f"Running Floquet Simulation: nu={nu}, A={drive_amp}, w={drive_freq}...")
    
    # 1. Setup Static Grid and Basis
    x, dx = build_Grid(L, N)
    x_matrice, p_matrice, energies = energy_matrices(x, dx, nu, a1, a2)
    
    # Diagonal H_0 in its own basis
    H0_matrix = np.diag(energies)
    
    # 2. Time Evolution Setup
    t_range = np.linspace(0, t_max, steps)
    dt = t_range[1] - t_range[0]
    
    # Initialize Unitary Operator as Identity
    U_total = np.eye(len(energies), dtype=complex)
    
    otoc_values = []
    
    # 3. Time Loop (Trotterized Evolution)
    for t in t_range:
        # Calculate OTOC for the current state (U_total)
        otoc = calculate_otoc_from_unitary(U_total, x_matrice, p_matrice, n_state)
        otoc_values.append(np.real(otoc))
        
        # Construct instantaneous Hamiltonian H(t) in energy basis
        # V_drive(t) = x_matrix * A * cos(omega * t)
        V_drive = x_matrice * drive_amp * np.cos(drive_freq * t)
        H_t = H0_matrix + V_drive
        
      
        dU = expm(-1j * H_t * dt)
        U_total = dU @ U_total

    return t_range, otoc_values




L = 10.0  
N = 100   # Reduced N for Floquet speed (Matrix Exp is O(N^3)). Increase to 300-500 for high precision.
t_max = 50
n_target = 3 # Eigenstate index to evaluate

# Potential Parameters (Volcano / Pöschl-Teller)
a1 = -1.0 
a2 = 1.0
nu_values = [-1.0, 0.5] # nu=-1 (Integrable Pöschl-Teller), nu=0.5 (Confining)

# Floquet Parameters (The New Pathway)
drive_amplitude = 0.5  # Strength of the chaos-inducing field
drive_frequency = 1.2  # Resonance frequency

plt.figure(figsize=(12, 7), dpi=100)

colors = ['b', 'r']

for i, nu_val in enumerate(nu_values):
    # 1. Run Baseline (Integrable/Static)
    # Setting A=0 recovers the static case
    t_static, otoc_static = run_floquet_simulation(N, t_max, n_target, nu_val, a1, a2, 0, 0, steps=150)
    
    # 2. Run Floquet (Chaotic/Driven)
    t_floquet, otoc_floquet = run_floquet_simulation(N, t_max, n_target, nu_val, a1, a2, drive_amplitude, drive_frequency, steps=150)
    
    # Plotting
    plt.plot(t_static, otoc_static, linestyle='--', color=colors[i], alpha=0.6, label=f'Static (Integrable) nu={nu_val}')
    plt.plot(t_floquet, otoc_floquet, linestyle='-', color=colors[i], linewidth=2, label=f'Floquet (Chaotic) nu={nu_val}')

plt.grid(True, which="both", ls="--", alpha=0.4)  
plt.yscale('log')
plt.xlabel('Time ($t$)', fontsize=14, fontweight='bold')
plt.ylabel('OTOC Intensity $C(t)$', fontsize=14, fontweight='bold')
plt.title(f'Pathway to Chaos: Floquet Driving ($A={drive_amplitude}, \omega={drive_frequency}$)', fontsize=16, pad=15)
plt.legend(fontsize=12, frameon=True, loc='lower right', shadow=True)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# --- 1. System Definitions ---
def sech(x):
    return 1/(np.cosh(x))

def build_Grid(L, N):
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    return x, dx

def volcano_potential(x, a1, a2, nu):
    cosh_term = np.cosh(x)
    sech_term = sech(x)
    # Using the confining form: a1 < 0
    return -1*((a1 * (cosh_term)**(2 * nu)) + (a2 * (sech_term)**2))

def solve_schrodinger(x, dx, V_array):
    size = len(x)
    array = np.ones(size)
    array_off = np.ones(size - 1)
    
    # Kinetic Energy Matrix (Finite Difference)
    V_matrice = np.diag(V_array)
    T_matrice = (2*np.diag(array, k = 0) + -1*np.diag(array_off, k = 1) + -1*np.diag(array_off, k = -1)) / (2 * dx * dx)
    H_matrice = T_matrice + V_matrice
    
    eigenvalue, eigenvector = np.linalg.eigh(H_matrice)
    return eigenvalue, eigenvector

def get_operators_in_energy_basis(x, dx, nu, a1, a2):
    V_array = volcano_potential(x, a1, a2, nu)
    
    # Position operator in position basis (diagonal)
    x_pos_matrice = np.diag(x)
    
    # Momentum operator in position basis (finite difference)
    size = len(x)
    array_off = np.ones(size - 1)
    P_pos_matrice = -1j * (np.diag(array_off, k = 1) - np.diag(array_off, k = -1)) / (2 * dx)
    
    # Solve static H_0
    eigenvals, eigenvecs = solve_schrodinger(x, dx, V_array)

    # Transform operators to Energy Basis
    # O_E = U^T @ O_x @ U
    x_matrix_E = eigenvecs.T @ x_pos_matrice @ eigenvecs
    p_matrix_E = eigenvecs.T @ P_pos_matrice @ eigenvecs

    return x_matrix_E, p_matrix_E, eigenvals

def calculate_otoc(U_t, x_mat_0, p_mat_0, n):
    """Calculates OTOC < |[x(t), p(0)]|^2 > for eigenstate n"""
    # Heisenberg evolution: x(t) = U_t^dagger * x(0) * U_t
    x_t = U_t.conj().T @ x_mat_0 @ U_t
    
    # Commutator c(t) = [x(t), p(0)]
    comm = x_t @ p_mat_0 - p_mat_0 @ x_t
    
    # Intensity
    intensity = comm.conj().T @ comm
    return np.real(intensity[n, n])

def run_floquet_simulation(N, t_max, n_state, nu, a1, a2, A, omega, steps=100):
    """
    Runs time-evolution for H(t) = H0 + x * A * cos(omega * t)
    """
    x, dx = build_Grid(10.0, N) # L=10.0
    x_mat, p_mat, energies = get_operators_in_energy_basis(x, dx, nu, a1, a2)
    H0 = np.diag(energies)
    
    t_range = np.linspace(0, t_max, steps)
    dt = t_range[1] - t_range[0]
    
    U = np.eye(len(energies), dtype=complex)
    otoc_list = []
    
    for t in t_range:
        otoc_list.append(calculate_otoc(U, x_mat, p_mat, n_state))
        
        # H(t) = H0 + x * A * cos(wt)
        # Note: In energy basis, x is dense
        V_t = x_mat * A * np.cos(omega * t)
        H_t = H0 + V_t
        
        # Propagate U(t+dt) = exp(-i*H(t)*dt) * U(t)
        U = expm(-1j * H_t * dt) @ U
        
    return t_range, np.array(otoc_list)

# --- 2. Simulation Parameters ---
# NOTE: Increase N to 200-300 for higher precision if running locally.
N = 100 
t_max = 30
steps = 100
n_target = 3
a1 = -1.0
a2 = 1.0

# --- 3. Experiment 1: Amplitude Sweep (Transition to Chaos) ---
# Fixed nu=0.5 (Confining), Fixed omega=1.2
nu_fixed = 0.5
w_fixed = 1.2
amplitudes = [0.0, 2.0, 3.5, 5.0]

results_amp = {}
print("Running Amplitude Sweep...")
for A in amplitudes:
    print(f"  Simulating Amplitude A={A}...")
    t, res = run_floquet_simulation(N, t_max, n_target, nu_fixed, a1, a2, A, w_fixed, steps)
    results_amp[A] = (t, res)

# --- 4. Experiment 2: Anharmonicity Sweep (Effect of Geometry) ---
# Fixed A=0.5 (Weak/Moderate drive), Fixed omega=1.2
# Compare nu=0.5, nu=1.0, nu=2.0
A_fixed = 0.5
nu_values = [0.5, 1.0, 2.0]

results_nu = {}
print("\nRunning Anharmonicity Sweep...")
for nu in nu_values:
    print(f"  Simulating nu={nu}...")
    t, res = run_floquet_simulation(N, t_max, n_target, nu, a1, a2, A_fixed, w_fixed, steps)
    results_nu[nu] = (t, res)

# --- 5. Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

# Plot 1: Amplitudes
colors_amp = ['black', 'blue', 'orange', 'red']
for i, A in enumerate(amplitudes):
    t, res = results_amp[A]
    label = f'Static (Integrable)' if A == 0 else f'A={A}'
    style = '--' if A == 0 else '-'
    ax1.plot(t, res, label=label, color=colors_amp[i], linestyle=style, linewidth=2)

ax1.set_yscale('log')
ax1.set_title(f'Transition to Chaos (Amplitude Sweep)\nFixed $\\nu={nu_fixed}, \omega={w_fixed}$', fontsize=16, fontweight='bold')
ax1.set_xlabel('Time $t$')
ax1.set_ylabel('OTOC Intensity $C(t)$')
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.3)

# Plot 2: Anharmonicity
colors_nu = ['blue', 'green', 'purple']
for i, nu in enumerate(nu_values):
    t, res = results_nu[nu]
    ax2.plot(t, res, label=f'$\\nu={nu}$', color=colors_nu[i], linewidth=2)

ax2.set_yscale('log')
ax2.set_title(f'Effect of Anharmonicity (Geometry Sweep)\nFixed $A={A_fixed}, \omega={w_fixed}$', fontsize=16, fontweight='bold')
ax2.set_xlabel('Time $t$')
ax2.set_ylabel('OTOC Intensity $C(t)$')
ax2.legend()
ax2.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()


