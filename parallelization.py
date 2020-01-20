import numpy as np
import os
import time

from scipy.linalg import norm
from joblib import Parallel, delayed

from particle_box import ParticleBox
from simulation import Simulation
from config import results_folder, init_folder

import utility_functions as util_funcs


def run_simulation(N, xi, v0, radius, run_number):
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}.npy'))
    radii = np.ones(N) * radius  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass

    energy_matrix = np.zeros((N, 2))  # array with mass and speed to save to file
    energy_matrix[:, 0] = mass

    average_number_of_collisions_stop = N*0.02

    velocities = util_funcs.random_uniformly_distributed_velocities(N, v0)

    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii)

    simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=average_number_of_collisions_stop)
    simulation.simulate_until_given_number_of_collisions('test_parall', output_timestep=0.1, save_positions=False)

    energy_matrix[:, 1] = norm(simulation.box_of_particles.velocities, axis=1)

    np.save(file=os.path.join(results_folder, f'distributionEqParticles_N_{N}_eq_energy_matrix_{run_number}'),
            arr=energy_matrix)


def problem(number_of_cores):
    N = 2000  # number of particles
    xi = 1  # restitution coefficient
    v_0 = 0.2  # initial speed
    radius = 0.007  # radius of all particles

    number_of_runs = 30

    Parallel(n_jobs=number_of_cores)(delayed(run_simulation)(N, xi, v_0, radius, run_number)
                                     for run_number in range(number_of_runs))


# data_matrix = np.ones((4, 2))
# for i in range(1, 5):
#     start_time = time.time()
#     problem(number_of_cores=i)
#     data_matrix[i-1, 0] = i
#     data_matrix[i-1, 1] = time.time()-start_time
# print(data_matrix)

problem(number_of_cores=4)
