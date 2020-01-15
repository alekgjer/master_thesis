import numpy as np
import os

from scipy.linalg import norm

from particle_box import ParticleBox
from simulation import Simulation
from config import results_folder, init_folder

# Various utility functions


def random_positions_and_radius(number_particles, min_value=1e-04, y_max=1):
    """
        Function to create random positions in a desired area of a square box with lines at x = 0, x=1, y=0, y=1
    :param number_particles: number of particles in the box
    :param min_value: minimum position away from an axis
    :param y_max: max position away from y-axis
    :return: random positions decided by input parameters and min possible radius for the particles to not overlap each
    other or a wall
    """
    # random positions making sure that the positions do not get closer than min_value to the walls
    x_pos = np.random.uniform(low=min_value, high=(1-min_value), size=number_particles)
    y_pos = np.random.uniform(low=min_value, high=(y_max-min_value), size=number_particles)
    positions = np.zeros((number_particles, 2))  # save positions as a (N, 2) array.
    positions[:, 0] = x_pos
    positions[:, 1] = y_pos

    min_distance_from_zero_x = np.min(positions[:, 0])  # closest particle distance to y_axis
    min_distance_from_zero_y = np.min(positions[:, 1])  # closest particle distance to x_axis
    min_distance_from_x_max = np.min(1 - positions[:, 0])  # closest particle distance to x=1
    min_distance_from_y_max = np.min(y_max - positions[:, 1])  # closest particle distance to y_max

    min_distance = min(min_distance_from_x_max, min_distance_from_y_max,
                       min_distance_from_zero_x, min_distance_from_zero_y)  # closest particle distance to any wall

    for i in range(np.shape(positions)[0]):  # loop through all positions
        random_position = positions[i, :]  # pick out a random position
        diff = positions - np.tile(random_position, reps=(len(positions), 1))  # find distance vectors to other pos
        diff = norm(diff, axis=1)  # find distance from all distance vectors
        number_of_zeros = np.sum(diff == 0)  # compute number of zero distance. Should only be one(itself!)

        if number_of_zeros > 1:  # Can happen(maybe?)
            print('Two particles randomly generated at same point')
            exit()
        diff = diff[diff != 0]  # remove the distance a particle has with itself
        min_distance = min(min_distance, np.min(diff))  # find the closest distance between wall and nearest particle
    print(min_distance)
    min_radius = min_distance/2  # divide by two since the distance should equal at least radius*2
    min_radius *= 0.99  # in order to have no particles connected with either wall or another particle
    return positions, min_radius


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
