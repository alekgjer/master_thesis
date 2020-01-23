import numpy as np
import os

from scipy.linalg import norm
from joblib import Parallel, delayed

from particle_box import ParticleBox
from simulation import Simulation
from config import results_folder, init_folder

import matplotlib.pyplot as plt
# Various utility functions


def random_positions_for_given_radius(number_of_particles, radius, y_max=1.0, brownian_particle=False):
    """
        Function to create random positions for number_of_particles with a given radius.
    :param number_of_particles: int giving the amount of particles to create positions for
    :param radius: radius of the particles
    :param y_max: max limit on the y-axis. Can be used to create wall on bottom half or particles everywhere.
    :param brownian_particle: boolean value used to give if to create a bp in the middle with r=3r0
    :return: uniformly distributed positions as a 2D array with shape (number_of_particles, 2)
    """
    positions = np.zeros((number_of_particles, 2))  # output shape
    # get random positions in the specified region. Need to make the positions not overlapping with the walls
    # do create more points than the output since not all are accepted since they are too close to each other
    x_pos = np.random.uniform(low=radius * 1.001, high=(1 - radius), size=number_of_particles ** 2)
    y_pos = np.random.uniform(low=radius * 1.001, high=(y_max - radius), size=number_of_particles ** 2)
    counter = 0  # help variable to accept positions that are not too close

    if brownian_particle:
        positions[0, :] = [0.5, 0.5]
        counter += 1

    for i in range(len(x_pos)):
        if counter == len(positions):  # Check if there is enough accepted positions
            print('Done')
            break

        random_position = np.array([x_pos[i], y_pos[i]])  # pick a random position
        # create a distance vector to all accepted positions
        diff = positions - np.tile(random_position, reps=(len(positions), 1))
        diff = norm(diff, axis=1)  # compute distance as the norm of each distance vector
        if brownian_particle:
            diff[0] -= 2*radius
        boolean = diff <= (2 * radius)  # boolean array to indicate if new position is closer than 2*radius to another
        # check of boolean array. If the sum is higher than zero the random position is closer than 2R and is rejected
        if np.sum(boolean) > 0:
            continue
        else:
            # add position to output array
            positions[counter, :] = random_position
            counter += 1
    # remove all slots that did not get a random position
    positions = positions[positions[:, 0] != 0]
    number_of_positions = len(positions)  # number of accepted points -> number of accepted particles
    if y_max == 1:
        # save file for problem where one use the whole region
        if brownian_particle:
            np.save(
                file=os.path.join(init_folder,
                                  f'uniform_pos_around_bp_N_{number_of_positions}_rad_{radius}_bp_rad_{3*radius}'),
                arr=positions)
        else:
            np.save(file=os.path.join(init_folder, f'uniform_pos_N_{number_of_positions}_rad_{radius}'), arr=positions)
    else:
        # save file for problem 5 where these positions in the wall getting hit by a projectile
        np.save(file=os.path.join(init_folder, f'wall_pos_N_{number_of_positions}_rad_{radius}'), arr=positions)


def validate_positions(positions, radius):
    """
        Function to validate the initial positions by checking if the particles are closer than 2r. Not valid for
        positions to a Brownian particle with bigger radius
    :param positions: positions of the particles as a (N, 2) array
    :param radius: radius of the particles
    """
    smallest_distance = np.Inf
    for i in range(len(positions)):
        diff = norm(positions - np.tile(positions[i, :], reps=(len(positions), 1)), axis=1)
        diff = diff[diff != 0]
        smallest_distance = min(smallest_distance, np.min(diff))
    if smallest_distance > (2*radius):
        print('Smallest distance between particles greater than 2r')
    else:
        print('Overlapping positions!!')
    print(f'Smallest dist: {smallest_distance} 2r: {2 * radius}')


def check_speed_distribution(number_of_particles=2000, r=0.007):
    velocities = np.load(os.path.join(init_folder, f'eq_velocity_N_{number_of_particles}_rad_{r}.npy'))
    speeds = norm(velocities, axis=1)
    plt.hist(speeds, bins=100, density=True)
    plt.title(r'$\langle v \rangle = {}$'.format(np.mean(speeds)))
    plt.show()


def random_uniformly_distributed_velocities(N, v0):
    """
        Function that creates a random set of velocity vectors for N particles with speed v0. First one create random
        angles uniformly distributed between (0, 2*pi). Then one use that the velocity in the x direction is equal
        to the cosine of the random angles multiplied by the speed. Same for velocity in y direction, but by using sine.
    :param N: number of particles
    :param v0: initial speed of all particles
    :return: uniformly distributed velocities as a 2D array with shape (number_of_particles, 2)
    """
    random_angles = np.random.uniform(low=0, high=2 * np.pi, size=N)  # random angles in range(0, 2pi)
    velocities = np.zeros((N, 2))
    velocities[:, 0], velocities[:, 1] = np.cos(random_angles), np.sin(random_angles)  # take cosine and sine
    velocities *= v0  # multiply with speed
    return velocities


def run_simulations_in_parallel(particle_parameters, simulation_parameters, simulation_function, number_of_cores,
                                number_of_runs):
    Parallel(n_jobs=number_of_cores)(delayed(simulation_function)(particle_parameters, simulation_parameters,
                                                                  run_number) for run_number in range(number_of_runs))


def create_visualization_system(particle_parameters, simulation_parameters, run_number):
    N, xi, v0, r = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    t_stop, timestep, tc = simulation_parameters[0], simulation_parameters[1], simulation_parameters[2]
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{r}.npy'))
    radii = np.ones(N) * r  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass

    velocities = np.load(os.path.join(init_folder, f'eq_velocity_N_{N}_rad_{r}.npy'))
    np.random.shuffle(velocities)

    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii)

    simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=t_stop, tc=tc)
    simulation.simulate_statistics_until_given_time(f'visualization_{run_number}', output_timestep=timestep,
                                                    save_positions=True)


def speed_distribution(particle_parameters, simulation_parameters, run_number):
    N, xi, v0, r = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{r}.npy'))
    radii = np.ones(N) * r  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass

    energy_matrix = np.zeros((N, 2))  # array with mass and speed to save to file
    energy_matrix[:, 0] = mass

    average_number_of_collisions_stop = N * 0.02

    velocities = random_uniformly_distributed_velocities(N, v0)

    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii)

    simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=average_number_of_collisions_stop)
    simulation.simulate_until_given_number_of_collisions('speed_distribution',
                                                         output_timestep=0.1,
                                                         save_positions=False)

    energy_matrix[:, 1] = norm(simulation.box_of_particles.velocities, axis=1)

    np.save(file=os.path.join(results_folder, f'distributionEqParticles_N_{N}_eq_energy_matrix_{run_number}'),
            arr=energy_matrix)
    # np.save(file=os.path.join(init_folder, f'eq_uniform_pos_N_{N}_rad_{r}.npy'),
    #         arr=simulation.box_of_particles.positions)
    # np.save(file=os.path.join(init_folder, f'eq_velocity_N_{N}_rad_{r}.npy'),
    #         arr=simulation.box_of_particles.velocities)


def energy_development(particle_parameters, simulation_parameters, run_number):
    N, xi, v0, r = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    t_stop, timestep, tc = simulation_parameters[0], simulation_parameters[1], simulation_parameters[2]
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{r}.npy'))
    radii = np.ones(N) * r  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass

    # velocities = random_uniformly_distributed_velocities(N, v0)
    velocities = np.load(os.path.join(init_folder, f'eq_velocity_N_{N}_rad_{r}.npy'))
    np.random.shuffle(velocities)

    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii)

    simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=t_stop, tc=tc)
    time_array, energy_array, speed_array = simulation.simulate_statistics_until_given_time('eng_development',
                                                                                            output_timestep=timestep,
                                                                                            save_positions=False)

    energy_matrix = np.zeros((len(time_array), 3))
    energy_matrix[:, 0] = time_array
    energy_matrix[:, 1] = energy_array
    energy_matrix[:, 2] = speed_array

    np.save(file=os.path.join(results_folder, f'energy_development_N_{N}_xi_{xi}_tstop_{t_stop}_{run_number}'),
            arr=energy_matrix)
    # np.save(file=os.path.join(results_folder, f'positions_t_{t_stop}_N_{N}_xi_{xi}_tstop_{run_number}'),
    #         arr=simulation.box_of_particles.positions)
