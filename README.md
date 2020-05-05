# master_thesis
The code written and used for simulations as part of my master's thesis on my fifth year on applied physics and mathematics at NTNU.

The code is used to study different applications of granular gas dynamics and consist of two different simulation methods. The first is an event driven simulation used to conduct molecular dynamics simulations of particles colliding in a three-dimensional cubic box. The second is based on Langevin dynamics, where I have implemented the Euler-Maruyama scheme to solve different Langevin equations modelling the dynamics of different particles. Both methods can be used to study different properties of a molecular and a granular gas, including speed distributions, evolution of temperature and Brownian motion. The latter is done by computing the so-called mean squared displacement (MSD) and comparing with theory from kinetic gas theory.

All code is run from problems.py, which has been implemented with argparse to allow command line input for different parameters. Run python3 problems.py -h to see which parameters can be set etc.. Remember that we need initial conditions for system in terms of initial positions and velocities. These can be created if one wants to use other systems than given in the folder initial_values.

The code performs simulations in parallel by creating new initial states based on data from file and use a unique run_number to write the results to file. The results have then been visualization by using jupyter notebook. The code can also be run on high performance computing (HPC) clusters with some minor changes to some of the implementation of the event driven simulation. The use of event driven simulation to compute MSD have been tested on one of the HPC clusters available at NTNU.

The general structure of the code can be summarized as follows:

particle_box.py: contains a class ParticleBox which has the variables, arrays and functionality for the system of particles.

simulation.py: contains a class Simulation which is a set of different implemenatations of an event driven simulation used for different applications. Each implementation has the same general idea, which is to increment time from collision to collision while updating the data of a ParticleBox.

sde_solver.py: contains a class SDESolver which are used to solve SDEs numerically by applying the Euler-Maruyama method. Can also compute the time averaged MSD, used to look at the ergodic property for different systems.

utility_functions.py: implementation of different help functions used to do perform different simulations.

problems.py: the main code used to run different problems e.g event driven simulation to compute MSD or solve SDEs. The code here calls on different utility functions that use either Simulation or SDESolver to do simulations.
