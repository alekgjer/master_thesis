import time
import os
import numpy as np

import utility_functions as util_funcs

from sde_solver import SDESolver

from config import init_folder

# File used to solve different problems by using the functions in utility_functions

start_time = time.time()
# choose what problem one want to solve by simulating particle collisions in 3D
problem = {0: 'Testing',
           1: 'Visualization 2D',
           2: 'Simulation statistics',
           3: 'Speed distribution',
           4: 'Mean square displacement',
           5: 'SDE solver',
           6: 'Mean free path',
           7: 'Disease outbreak',
           }[5]
print(f"Problem: {problem}")

# set particle parameters
N = 1000  # number of particles
xi = 0.8  # restitution coefficient
v0 = np.sqrt(2)  # initial speed. Only used if all particles start with the same speed.
radius = 1/40  # radius of each particle

if problem == 'Testing':
    # util_funcs.check_speed_distribution(number_of_particles=N, rad=radius)
    # util_funcs.test_inelastic_collapse()
    util_funcs.random_positions_for_given_radius(N, radius, 2)
    # pos = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}_2d.npy'))
    # util_funcs.validate_positions(pos, radius)
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
elif problem == 'Simulation statistics':
    t_stop = 10
    timestep = 0.1
    tc = 0
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.get_simulation_statistics,
                                           number_of_cores=4,
                                           number_of_runs=4)
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
elif problem == 'Mean square displacement':
    t_stop = 10000
    timestep = 1
    tc = 0
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.mean_square_displacement,
                                           number_of_cores=4,
                                           number_of_runs=4)
elif problem == 'SDE solver':
    dt = 0.1
    t_stop = 2000
    if xi == 1:
        gamma0, d0, tau = 11.43, 0.058, np.inf
        sde_solver = SDESolver(t_start=0, t_stop=t_stop, dt=dt, number_of_particles=N, constants=[gamma0, d0, tau])
        sde_solver.ensemble_msd(sde_solver.friction_underdamped_langevin_equation,
                                sde_solver.diffusivity_underdamped_langevin_equation)
    elif xi == 0.8:
        gamma0, d0, tau = 9.26, 0.072, 0.97
        sde_solver = SDESolver(t_start=0, t_stop=t_stop, dt=dt, number_of_particles=N, constants=[gamma0, d0, tau])
        sde_solver.ensemble_msd(sde_solver.friction_udsbm, sde_solver.diffusivity_udsbm)
    else:
        print('SDE is currently solved for xi=1 or xi=0.8. Change parameters for other xi!!')
elif problem == 'Mean free path':
    timestep = 1
    t_stop = 10
    util_funcs.mean_free_path(particle_parameters=[N, xi, v0, radius],
                              simulation_parameters=[t_stop, timestep, 0],
                              run_number=-1)
elif problem == 'Disease outbreak':
    t_stop = 3
    timestep = 0.01
    tc = 0
    util_funcs.simulate_disease_outbreak(particle_parameters=[N, xi, v0, radius],
                                         simulation_parameters=[t_stop, timestep, tc])

print(f"Time used: {time.time() - start_time}")
