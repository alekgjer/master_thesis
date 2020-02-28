import time
import os
import numpy as np

import utility_functions as util_funcs

from config import init_folder

# File used to solve different problems by using the functions in utility_functions

start_time = time.time()
# choose what problem one want to solve by simulating particle collisions in 3D
problem = {0: 'Testing',
           1: 'Visualization',
           2: 'Speed distribution',
           3: 'Energy development',
           4: 'Mean square displacement'
           }[4]
print(f"Problem: {problem}")

# set particle parameters
N = 1000  # number of particles
xi = 0.5  # restitution coefficient
v0 = np.sqrt(2)  # initial speed. Only used if all particles start with the same speed.
radius = 1/40  # radius of each particle

if problem == 'Testing':
    # util_funcs.check_speed_distribution(number_of_particles=N, rad=radius)
    # util_funcs.test_inelastic_collapse()
    # util_funcs.random_positions_for_given_radius(N, radius, 3)
    pos = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}_3d.npy'))
    util_funcs.plot_positions_3d(pos, radius)
elif problem == 'Visualization':
    # TODO: not possible atm due to 3d simulations
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
    average_number_of_collisions_stop = 0.1*N
    timestep = 1
    tc = 0
    do_many = True  # if True: do it for speed distribution. If not do to create eq. state
    if do_many:
        util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                               simulation_parameters=[average_number_of_collisions_stop, timestep, tc],
                                               simulation_function=util_funcs.speed_distribution,
                                               number_of_cores=4,
                                               number_of_runs=20)
    else:
        util_funcs.speed_distribution(particle_parameters=[N, xi, v0, radius],
                                      simulation_parameters=[average_number_of_collisions_stop, timestep, tc],
                                      run_number=-1)
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
    t_stop = 1000
    timestep = 0.1
    tc = 0
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.mean_square_displacement,
                                           number_of_cores=4,
                                           number_of_runs=4)

print(f"Time used: {time.time() - start_time}")
