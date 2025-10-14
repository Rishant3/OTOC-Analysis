Quantum Chaos Simulation: OTOC for the Pöschl-Teller Potential
This project explores quantum chaos by simulating the Out-of-Time-Ordered Correlator (OTOC) for the Pöschl-Teller potential, a special exactly solvable system in quantum mechanics. The OTOC is a key indicator of chaotic behavior and scrambling of information in quantum systems.

The simulation is implemented in a Jupyter Notebook using standard Python scientific libraries like numpy, scipy, and matplotlib.

##### Physics Background ######
The Pöschl-Teller Potential
The potential is defined by the equation:
$$ V(x) = -\frac{\lambda(\lambda-1)}{2} \text{sech}^2(x) $$
The integer parameter λ determines the depth of the potential well and the number of bound energy states.

The Out-of-Time-Ordered Correlator (OTOC)
The OTOC measures how a small, early perturbation affects a later measurement. For this simulation, we compute the squared commutator of the position and momentum operators:
$$ C(t) = -\langle[\hat{x}(t), \hat{p}(0)]^2\rangle $$
Its early-time exponential growth is a signature of quantum chaos, characterized by a Lyapunov exponent.

##### Features #####
Exact Solutions: Computes the exact bound state wavefunctions and energy eigenvalues for the Pöschl-Teller potential.

Operator Representation: Constructs the position ( 
x
^
 ) and momentum ( 
p
^
​
 ) operators in the energy eigenbasis.

Microcanonical OTOC: Calculates the OTOC, C 
n
​
 (t), for individual energy states n.

Thermal OTOC: Calculates the thermally-averaged OTOC, C 
T
​
 (t), for a given inverse temperature β.

High-Quality Visualization: Generates aesthetically pleasing, wide-format plots of the results directly within the notebook, with a style inspired by academic publications.


Requirements
Python 3.x

Jupyter Notebook or JupyterLab

The following Python libraries:

numpy

scipy

matplotlib

pandas

You can install the dependencies using pip:

Bash

pip install numpy scipy matplotlib pandas jupyter
Usage
Clone this repository to your local machine.

Open the Poschl_Teller (1).ipynb file in Jupyter Notebook or JupyterLab.

Navigate to the final cell of the notebook (the "Execution Cell").

Modify the parameters in this cell to configure your simulation.

Run all cells in the notebook (Cell > Run All or Kernel > Restart & Run All). The simulation will run for each specified lambda value, and the plots will be displayed inline.

##### Customization #####
You can easily customize the simulation by changing the parameters in the final cell of the notebook:

lams_to_run: A list of integer λ values to simulate. The simulation will run once for each value in the list.

time_max: The maximum time (t 
max
​
 ) for the OTOC evolution.

time_steps: The number of time points in the simulation grid.

beta_values: A list of inverse temperature (β) values for which to calculate the thermal OTOC.

L_box and N_points: The half-width of the spatial box and the number of grid points for the numerical calculations.
