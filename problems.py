import time
import numpy as np

import utility_functions as util_funcs

# File used to solve different problems by using the functions in utility_functions

start_time = time.time()
# choose what problem one want to solve by simulating particle collisions in 2D
problem = {0: 'Testing',
           1: 'Visualization',
           2: 'Speed distribution',
           3: 'Energy development',
           }[3]
print(f"Problem: {problem}")

# set particle parameters
N = 2000
xi = 0.8
v0 = 0.2
radius = 0.007

if problem == 'Testing':
    util_funcs.check_speed_distribution()
    # util_funcs.histogram_positions()
    # util_funcs.test_inelastic_collapse()
elif problem == 'Visualization':
    t_stop = 100
    timestep = 5
    tc = 1e-05
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.create_visualization_system,
                                           number_of_cores=2,
                                           number_of_runs=2)
elif problem == 'Speed distribution':
    # let system evolve in time until enough collisions has occurred to assume equilibrium has been reached.
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[],
                                           simulation_function=util_funcs.speed_distribution,
                                           number_of_cores=4,
                                           number_of_runs=30)
elif problem == 'Energy development':
    t_stop = 100
    timestep = 0.1
    tc = 1e-01
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.energy_development,
                                           number_of_cores=4,
                                           number_of_runs=4)

print(f"Time used: {time.time() - start_time}")
