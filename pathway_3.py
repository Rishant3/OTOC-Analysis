
import numpy as np
import matplotlib.pyplot as plt

# --- 1. System Setup ---
def sech(x):
    return 1/(np.cosh(x))

def build_Grid(L, N):
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    return x, dx

def volcano_potential(x, a1, a2, nu):
    # a1=0, a2 < 0 gives a positive barrier (Hill)
    # V(x) = - ( a2 * sech^2(x) ) = |a2| * sech^2(x)
    cosh_term = np.cosh(x)
    sech_term = sech(x)
    return -1*((a1 * (cosh_term)**(2 * nu)) + (a2 * (sech_term)**2))

def solve_schrodinger(x, dx, V_array):
    size = len(x)
    array = np.ones(size)
    array_off = np.ones(size - 1)
    
    V_matrice = np.diag(V_array)
    # Kinetic Energy (Finite Difference)
    T_matrice = (2*np.diag(array, k = 0) + -1*np.diag(array_off, k = 1) + -1*np.diag(array_off, k = -1)) / (2 * dx * dx)
    H_matrice = T_matrice + V_matrice
    
    # Solve Eigenproblem
    eigenvalue, eigenvector = np.linalg.eigh(H_matrice)
    return eigenvalue, eigenvector

def get_operators_in_energy_basis(x, dx, nu, a1, a2):
    V_array = volcano_potential(x, a1, a2, nu)
    
    x_pos_matrice = np.diag(x)
    size = len(x)
    array_off = np.ones(size - 1)
    P_pos_matrice = -1j * (np.diag(array_off, k = 1) - np.diag(array_off, k = -1)) / (2 * dx)
    
    eigenvals, eigenvecs = solve_schrodinger(x, dx, V_array)
    
    # Transform operators to Energy Basis
    x_matrix_E = eigenvecs.T @ x_pos_matrice @ eigenvecs
    p_matrix_E = eigenvecs.T @ P_pos_matrice @ eigenvecs
    
    return x_matrix_E, p_matrix_E, eigenvals, eigenvecs

def gaussian_state(x, x0, p0, sigma):
    """Generates a Gaussian wavepacket in position space"""
    psi = (1/(np.pi * sigma**2))**0.25 * np.exp(-0.5 * ((x - x0)/sigma)**2 + 1j * p0 * (x - x0))
    norm = np.sqrt(np.sum(np.abs(psi)**2) * (x[1]-x[0])) # Normalize on grid
    return psi / norm

def calculate_otoc_max(x_mat, p_mat, energies, psi_E, t_values):
    """
    Calculates OTOC and returns the maximum value reached during the time window.
    """
    max_otoc = 0.0
    
    # Loop over time
    for t in t_values:
        U_diag = np.exp(-1j * energies * t)
        U_dag_diag = np.exp(1j * energies * t)
        
        # x(t) = U^d x U
        phase_matrix = np.outer(U_dag_diag, U_diag) 
        x_t = x_mat * phase_matrix
        
        # Commutator
        comm = x_t @ p_mat - p_mat @ x_t
        
        # Expectation
        phi = comm @ psi_E
        intensity = np.real(np.vdot(phi, phi))
        
        if intensity > max_otoc:
            max_otoc = intensity
            
    return max_otoc

# --- 2. Simulation Parameters ---
L = 30.0 # Long grid for scattering
N = 250
x, dx = build_Grid(L, N)
t_max = 15.0
t_values = np.linspace(0, t_max, 60) 

# Potential: Barrier (Hill) of height 2.0
# V(x) = 2.0 * sech^2(x)
a1 = 0.0
a2 = -2.0 
nu = 1.0 
V_max = 2.0

# Pre-calculate operators (Potential is fixed)
x_mat, p_mat, energies, eigenvecs = get_operators_in_energy_basis(x, dx, nu, a1, a2)

# --- 3. Energy Scan ---
# Scan energy from 0.5 to 4.0 to find the resonance at E=2.0
incident_energies = np.linspace(0.5, 4.0, 20) 
max_otoc_values = []

print(f"Starting Scattering Scan (V_max = {V_max})...")

for E in incident_energies:
    # Set initial momentum p0 = sqrt(2mE)
    p0 = np.sqrt(2 * E)
    
    # Initial State: Gaussian incoming from left (x0 = -10)
    psi_x = gaussian_state(x, x0=-10.0, p0=p0, sigma=1.0)
    
    # Project to Energy Basis
    psi_E = eigenvecs.T @ psi_x
    psi_E /= np.linalg.norm(psi_E)
    
    # Run Dynamics
    val = calculate_otoc_max(x_mat, p_mat, energies, psi_E, t_values)
    max_otoc_values.append(val)
    # print(f"  Energy={E:.2f}, Max OTOC={val:.2f}")

# --- 4. Plotting ---
plt.figure(figsize=(9, 6), dpi=100)
plt.plot(incident_energies, max_otoc_values, 'o-', color='darkgreen', linewidth=2, markersize=8, label='Scattering Response')

# Mark the Barrier Height
plt.axvline(x=V_max, color='red', linestyle='--', label=f'Barrier Height ($V_0={V_max}$)')

plt.title('Pathway 3: Scattering Chaos (Resonance Spike)', fontsize=16)
plt.xlabel('Incident Energy ($E$)', fontsize=14, fontweight='bold')
plt.ylabel('Max OTOC Intensity', fontsize=14, fontweight='bold')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.legend(fontsize=12)
plt.show()


