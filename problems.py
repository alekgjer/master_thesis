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
           4: 'Mean square displacement'
           }[4]
print(f"Problem: {problem}")

# set particle parameters
N = 1000
xi = 0.9
v0 = 0.2
radius = 0.01

if problem == 'Testing':
    # util_funcs.check_speed_distribution(number_of_particles=5000, r=0.004)
    # util_funcs.histogram_positions()
    util_funcs.test_inelastic_collapse()
elif problem == 'Visualization':
    t_stop = 500
    timestep = 1
    tc = 0
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.create_visualization_system,
                                           number_of_cores=1,
                                           number_of_runs=1)
elif problem == 'Speed distribution':
    # let system evolve in time until enough collisions has occurred to assume equilibrium has been reached.
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[],
                                           simulation_function=util_funcs.speed_distribution,
                                           number_of_cores=1,
                                           number_of_runs=1)
elif problem == 'Energy development':
    t_stop = 30
    timestep = 0.1
    tc = 0
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.energy_development,
                                           number_of_cores=4,
                                           number_of_runs=4)
elif problem == 'Mean square displacement':
    t_stop = 200
    timestep = 0.1
    tc = 0
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.mean_square_displacement,
                                           number_of_cores=4,
                                           number_of_runs=20)

print(f"Time used: {time.time() - start_time}")
