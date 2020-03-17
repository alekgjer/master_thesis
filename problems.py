import time
import os
import numpy as np

import utility_functions as util_funcs
import sde_solver as sde

from config import init_folder

# File used to solve different problems by using the functions in utility_functions

start_time = time.time()
# choose what problem one want to solve by simulating particle collisions in 3D
problem = {0: 'Testing',
           1: 'Visualization 2D',
           2: 'Speed distribution',
           3: 'Energy development',
           4: 'Mean square displacement',
           5: 'SDE solver',
           6: 'Mean free path',
           }[6]
print(f"Problem: {problem}")

# set particle parameters
N = 1000  # number of particles
xi = 1  # restitution coefficient
v0 = np.sqrt(2)  # initial speed. Only used if all particles start with the same speed.
radius = 0.03  # radius of each particle

if problem == 'Testing':
    # util_funcs.check_speed_distribution(number_of_particles=N, rad=radius)
    # util_funcs.test_inelastic_collapse()
    # util_funcs.random_positions_for_given_radius(N, radius, 2)
    pos = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}_2d.npy'))
    util_funcs.validate_positions(pos, radius)
    # util_funcs.plot_positions_3d(pos, radius)
elif problem == 'Visualization 2D':
    t_stop = 10
    timestep = 0.1
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
    dim = 3
    do_many = True  # if True: do it for speed distribution. If not do to create eq. state
    if do_many:
        util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                               simulation_parameters=[average_number_of_collisions_stop, timestep, dim],
                                               simulation_function=util_funcs.speed_distribution,
                                               number_of_cores=4,
                                               number_of_runs=20)
    else:
        util_funcs.speed_distribution(particle_parameters=[N, xi, v0, radius],
                                      simulation_parameters=[average_number_of_collisions_stop, timestep, dim],
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
elif problem == 'SDE solver':
    dt = 0.01
    t_stop = 10
    if xi == 1:
        sde.solve_underdamped_langevin_equation(N, dt, t_stop)
    elif xi == 0.8:
        sde.solve_udsbm_langevin_equation(N, dt, t_stop)
    else:
        print('SDE is currently solved for xi=1 or xi=0.8. Change parameters for other xi!!')
elif problem == 'Mean free path':
    timestep = 1
    t_stop = 10
    util_funcs.mean_free_path(particle_parameters=[N, xi, v0, radius],
                              simulation_parameters=[t_stop, timestep, 0],
                              run_number=-1)

print(f"Time used: {time.time() - start_time}")
