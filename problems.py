import time
import os
import argparse
import numpy as np

import utility_functions as util_funcs

from sde_solver import SDESolver

from config import init_folder

# File used to solve different problems by using the functions in utility_functions

parser = argparse.ArgumentParser(description='Python script used to study particles in a box by conducting event driven'
                                             ' simulations of a molecular or a granular gas. Script can also be used to'
                                             ' solve SDEs of Brownian motion using the Euler-Maruyama scheme. Running'
                                             ' script with -p 4/5 for MSD/SDE solution works on HPCs as well. NB! Only'
                                             ' event driven simulation uses several cores. No need for SDE since we'
                                             ' compute for all particles at all times using numpy.')
parser.add_argument('-p', metavar='Problem number', type=int,
                    help='Problem is given from 1-7, see Problems.py. Default: 0',
                    default=0)
parser.add_argument('-N', metavar='Number of particles', type=int,
                    help='Needs to match a set of initial values. Default: 1000.',
                    default=1000)
parser.add_argument('-xi', metavar='Restitution coefficient', type=float,
                    help='Float between 0 and 1. Default: 1, indicating a molecular gas.',
                    default=1.0)
parser.add_argument('-r', metavar='Particle radius', type=float,
                    help='Needs to match a set of initial values. Default: 0.025.',
                    default=0.025)
parser.add_argument('-sc', metavar='Stopping criterion', type=float,
                    help='Stopping criterion in time of average collisions before stop. Default: 1.',
                    default=1)
parser.add_argument('-dt', metavar='Timestep value', type=float,
                    help='Output timestep in simulation, or discretization for numerical solution of SDE. Default: 0.1',
                    default=0.1)
parser.add_argument('-tc', metavar='Duration of contact', type=float,
                    help='Used to implement the TC model. Default: 0, which is equal to not using TC model.',
                    default=0)
parser.add_argument('-nc', metavar='Number of cores', type=int,
                    help='Using joblib we run simulations in parallel and compute mean values. Default: 1.',
                    default=1)
parser.add_argument('-nr', metavar='Number of runs', type=int,
                    help='The number of simulations to be run in parallel. Default: 1.',
                    default=1)
args = vars(parser.parse_args())
# set particle and simulation parameters from default value or from command line
p = args['p']  # problem choice. Default: 0
N = args['N']  # number of particles. Default: 1000
xi = args['xi']  # restitution coefficient. Default: 1 -> molecular gas
assert 0 <= xi <= 1  # check if xi in correct range. Script will stop otherwise
radius = args['r']
# check if there exist initial value files for the given particle data
assert os.path.isfile(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}_3d.npy')) or \
       os.path.isfile(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}_2d.npy'))
stopping_criterion = args['sc']  # stopping criterion. Given as t_stop or avg_numb_coll_stop. Default: 1
dt = args['dt']  # Timestep value. Used for discretization of time of outputs or sde solution. Default: 0.1
tc = args['tc']  # duration of contact used to implement TC model. Default: 0
number_of_cores = args['nc']  # number of cores to use for parallelization. Default: 1
number_of_runs = args['nr']  # number of runs to use for parallelization. Default: 1

# choose what problem one want to solve by simulating particle collisions in 3D or solving SDE
problem = {0: 'Testing',
           1: 'Visualization 2D',
           2: 'Simulation statistics',
           3: 'Speed distribution',
           4: 'Mean square displacement',
           5: 'SDE solver',
           6: 'Mean free path',
           7: 'Disease outbreak',
           }[p]
print(f"Problem: {problem}")

v0 = np.sqrt(2)  # initial speed. Only used if all particles start with the same speed.

start_time = time.time()

if problem == 'Testing':
    # util_funcs.random_positions_for_given_radius(N, radius, 2)
    # pos = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}_2d.npy'))
    pos = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}_3d.npy'))
    util_funcs.validate_positions(pos, radius)
elif problem == 'Visualization 2D':
    t_stop = int(stopping_criterion)
    timestep = dt
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.create_visualization_system,
                                           number_of_cores=1,
                                           number_of_runs=1)
elif problem == 'Simulation statistics':
    t_stop = int(stopping_criterion)
    timestep = dt
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.get_simulation_statistics,
                                           number_of_cores=number_of_cores,
                                           number_of_runs=number_of_runs)
elif problem == 'Speed distribution':
    # let system evolve in time until enough collisions has occurred to assume equilibrium has been reached.
    average_number_of_collisions_stop = stopping_criterion
    timestep = dt
    dim = 3
    do_many = True  # if True: do it for speed distribution. If not do to create eq. state
    if do_many:
        util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                               simulation_parameters=[average_number_of_collisions_stop, timestep, dim],
                                               simulation_function=util_funcs.speed_distribution,
                                               number_of_cores=number_of_cores,
                                               number_of_runs=number_of_runs)
    else:
        util_funcs.speed_distribution(particle_parameters=[N, xi, v0, radius],
                                      simulation_parameters=[average_number_of_collisions_stop, timestep, dim],
                                      run_number=-1)
elif problem == 'Mean square displacement':
    t_stop = int(stopping_criterion)
    timestep = dt
    util_funcs.run_simulations_in_parallel(particle_parameters=[N, xi, v0, radius],
                                           simulation_parameters=[t_stop, timestep, tc],
                                           simulation_function=util_funcs.mean_square_displacement,
                                           number_of_cores=number_of_cores,
                                           number_of_runs=number_of_cores)
elif problem == 'SDE solver':
    t_stop = int(stopping_criterion)
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
        print('SDE is currently solved for xi=1 or xi=0.8. Change parameters for other values of xi!!')
elif problem == 'Mean free path':
    t_stop = int(stopping_criterion)
    timestep = dt
    util_funcs.mean_free_path(particle_parameters=[N, xi, v0, radius],
                              simulation_parameters=[t_stop, timestep, 0],
                              run_number=-1)
elif problem == 'Disease outbreak':
    t_stop = int(stopping_criterion)
    timestep = dt
    util_funcs.simulate_disease_outbreak(particle_parameters=[N, xi, v0, radius],
                                         simulation_parameters=[t_stop, timestep, tc])

print(f"Time used: {time.time() - start_time}")
