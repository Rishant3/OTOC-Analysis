# %%
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
    # Inverted Pöschl-Teller: a1=0, a2 < 0 creates a hill
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

def calculate_otoc_static(t_values, x_mat, p_mat, energies, psi_E):
    """
    Calculates OTOC for a static Hamiltonian H.
    """
    otoc_values = []
    
    # Vectorized calculation over time
    for t in t_values:
        # Evolution operator U(t) = exp(-iEt) (diagonal)
        U_diag = np.exp(-1j * energies * t)
        U_dag_diag = np.exp(1j * energies * t)
        
        # Heisenberg Picture x(t) = U^d x U
        # Since U is diagonal, (U^d x U)_nm = exp(i(En-Em)t) * x_nm
        phase_matrix = np.outer(U_dag_diag, U_diag) 
        x_t = x_mat * phase_matrix
        
        # Commutator C(t) = [x(t), p(0)]
        comm = x_t @ p_mat - p_mat @ x_t
        
        # Expectation Value <psi | C^d C | psi>
        phi = comm @ psi_E
        intensity = np.vdot(phi, phi) # Inner product
        otoc_values.append(np.real(intensity))
        
    return np.array(otoc_values)

# --- 2. Parameters & Execution ---
L = 20.0
N = 200 # Higher N for better resolution of the hill
x, dx = build_Grid(L, N)


# V(x) = +2.0 * sech^2(x)
a1 = 0.0
a2 = -2.0 # Negative a2 in formula => Positive Potential
nu = 1.0 

x_mat, p_mat, energies, eigenvecs = get_operators_in_energy_basis(x, dx, nu, a1, a2)

# Define Initial States (Gaussian Wavepackets)
# State 1: UNSTABLE (Balanced on the peak)
psi_unstable_x = gaussian_state(x, x0=0.0, p0=0.0, sigma=1.0)
# State 2: STABLE (Far away / Flat region)
psi_stable_x = gaussian_state(x, x0=5.0, p0=0.0, sigma=1.0)

# Project to Energy Basis
psi_unstable_E = eigenvecs.T @ psi_unstable_x
psi_unstable_E /= np.linalg.norm(psi_unstable_E) # Ensure normalization

psi_stable_E = eigenvecs.T @ psi_stable_x
psi_stable_E /= np.linalg.norm(psi_stable_E)

# Time Evolution
t_max = 8.0 
t_values = np.linspace(0, t_max, 100)

otoc_unstable = calculate_otoc_static(t_values, x_mat, p_mat, energies, psi_unstable_E)
otoc_stable = calculate_otoc_static(t_values, x_mat, p_mat, energies, psi_stable_E)

# --- 3. Plotting ---
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(t_values, otoc_unstable, 'r-', linewidth=2.5, label='Unstable Saddle ($x_0=0$)')
plt.plot(t_values, otoc_stable, 'b--', linewidth=2, label='Stable Region ($x_0=5$)')

plt.yscale('log')
plt.title('Pathway 2: Saddle-Point Scrambling (Transient Instability)', fontsize=16)
plt.xlabel('Time ($t$)', fontsize=14, fontweight='bold')
plt.ylabel('OTOC Intensity $C(t)$', fontsize=14, fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=12, frameon=True, shadow=True)
plt.show()


