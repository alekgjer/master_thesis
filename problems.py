import time
import numpy as np

import utility_functions as util_funcs

# File used to solve different problems by using the functions in utility_functions

start_time = time.time()
# choose what problem one want to solve by simulating particle collisions in 2D
problem = {1: 'Speed distribution',
           2: 'Energy development',
           }[2]
print(f"Problem: {problem}")

# set parameters
N = 2000
xi = 0.9
v0 = 0.2
radius = 0.007

if problem == 'Speed distribution':
    # let system evolve in time until enough collisions has occurred to assume equilibrium has been reached.
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_function=util_funcs.speed_distribution,
                                           number_of_cores=4,
                                           number_of_runs=30)
elif problem == 'Energy development':
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_function=util_funcs.energy_development,
                                           number_of_cores=4,
                                           number_of_runs=4)

print(f"Time used: {time.time() - start_time}")
