import numpy as np
import os

from scipy.linalg import norm
from joblib import Parallel, delayed

from particle_box import ParticleBox
from simulation import Simulation
from config import results_folder, init_folder

# Various utility functions used to solve the problems in problems.py.


def random_positions_for_given_radius(number_of_particles, radius, dimensions):
    """
        Function to create random positions for number_of_particles with a given radius.
    :param number_of_particles: int giving the amount of particles to create positions for.
    :param radius: radius of the particles.
    :param dimensions: int which tell the dimensions of the system. Can be 2 or 3.
    :return: uniformly distributed positions as a 2D or 3D array with shape (number_of_particles, 2 or 3).
    """
    positions = np.zeros((number_of_particles, dimensions))  # output shape
    # get random positions in the specified region. Need to make the positions not overlapping with the walls
    # do create more points than the output since not all are accepted since they are too close to each other
    x_pos = np.random.uniform(low=radius * 1.001, high=(1 - radius), size=number_of_particles ** 2)
    y_pos = np.random.uniform(low=radius * 1.001, high=(1 - radius), size=number_of_particles ** 2)
    z_pos = np.random.uniform(low=radius * 1.001, high=(1 - radius), size=number_of_particles ** 2)
    counter = 0  # help variable to accept positions that are not too close

    for i in range(len(x_pos)):
        if counter == len(positions):  # Check if there is enough accepted positions
            print('Done')
            break
        random_position = np.zeros(3)
        if dimensions == 2:
            random_position = np.array([x_pos[i], y_pos[i]])  # pick a random position in 2D
        elif dimensions == 3:
            random_position = np.array([x_pos[i], y_pos[i], z_pos[i]])  # pick a random position in 3D
        else:
            print('Dimensions need to be 2 or 3!')
            exit()

        # create a distance vector to all accepted positions
        diff = positions - np.tile(random_position, reps=(len(positions), 1))
        diff = norm(diff, axis=1)  # compute distance as the norm of each distance vector

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
    if dimensions == 2:
        np.save(file=os.path.join(init_folder, f'uniform_pos_N_{number_of_positions}_rad_{radius}_2d'), arr=positions)
    else:
        np.save(file=os.path.join(init_folder, f'uniform_pos_N_{number_of_positions}_rad_{radius}_3d'), arr=positions)


def validate_positions(positions, radius):
    """
        Function to validate the initial positions by checking if the particle centers are closer than 2r.
    :param positions: positions of the particles as a (N, 2 or 3) array.
    :param radius: radius of the particles.
    """
    smallest_distance = np.Inf  # help value
    for i in range(len(positions)):  # iterate through all particles
        # compute distance from all particle to one particle
        diff = norm(positions - np.tile(positions[i, :], reps=(len(positions), 1)), axis=1)
        diff = diff[diff != 0]  # remove distance to itself
        smallest_distance = min(smallest_distance, np.min(diff))  # update with the smallest distance
    if smallest_distance > (2*radius):
        print('Smallest distance between particles greater than 2r')
    else:
        print('Overlapping positions!!')
    print(f'Smallest dist: {smallest_distance} 2r: {2 * radius}')


def random_uniformly_distributed_velocities(number_of_particles, v0, dimensions):
    """
        Function that creates a random set of velocity vectors for N particles with speed v0. Works in 2D and 3D by
        using the definition of phi in [0, 2pi] and theta in [0, pi] for the polar(2D) and spherical coordinates(3D)
         definition In 2D: theta = pi/2 -> vz = 0.
    :param number_of_particles: number of particles.
    :param v0: initial speed of all particles.
    :param dimensions: int which tell the dimensions of the system. Can be 2 or 3.
    :return: uniformly distributed velocities as a 2D or 3D array with shape (number_of_particles, 2 or 3).
    """
    velocities = np.zeros((number_of_particles, dimensions))
    if dimensions == 2:
        phi = np.random.uniform(low=0, high=2 * np.pi, size=number_of_particles)  # random angles in range(0, 2pi)
        velocities[:, 0], velocities[:, 1] = np.cos(phi), np.sin(phi)  # polar coordinates
    elif dimensions == 3:
        phi = np.random.uniform(low=0, high=2 * np.pi, size=number_of_particles)  # random angles in range(0, 2pi)
        theta = np.random.uniform(low=0, high=np.pi, size=number_of_particles)  # random angles in range(0, pi)
        # spherical coordinates
        velocities[:, 0], velocities[:, 1], velocities[:, 2] = np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)  # take cosine and sine
    else:
        print('Dimensions need to be 2 or 3!')
        exit()
    velocities *= v0  # multiply with speed
    return velocities


def run_simulations_in_parallel(particle_parameters, simulation_parameters, simulation_function, number_of_cores,
                                number_of_runs):
    """
        Help function implemented in order to conduct several simulations in parallel by using the library joblib.
        It works in the following manner: simulation_function is run on number_of_cores cores with the same parameters
        for the particles and the simulation. They are each given different run_numbers in order to save their unique
        data without overwriting each other. This is repeated until number_of_runs are finished. This is used to run
        several simulations at the same time, and the results can be averaged afterwards to achieve mean behaviour.
    :param particle_parameters: array of [N, xi, v0, radius] used to initialize a ParticleBox.
    :param simulation_parameters: array of [stopping_criterion, output_timestep, tc] used to initialize a Simulation.
    :param simulation_function: which function to be run in parallel, e.g mean_square_displacement, speed_distribution.
    :param number_of_cores: the number of cores to run with. Limited by computer hardware. Usually 4 is used.
    :param number_of_runs: the number of times the simulation function will be run.
    """
    Parallel(n_jobs=number_of_cores)(delayed(simulation_function)(particle_parameters, simulation_parameters,
                                                                  run_number) for run_number in range(number_of_runs))


def create_visualization_system(particle_parameters, simulation_parameters, run_number):
    """
        Function used to conduct 2D simulations and thus visualize a system of particles.
    :param particle_parameters: array of [N, xi, v0, radius] used to initialize a ParticleBox.
    :param simulation_parameters: array of [t_stop, output_timestep, tc] used to initialize a Simulation.
    :param run_number: int used such that one can run parallel simulations save different visualizations.
    """
    N, xi, v0, r = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    t_stop, timestep, tc = simulation_parameters[0], simulation_parameters[1], simulation_parameters[2]
    positions = np.zeros((N, 3))
    positions[:, :2] = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{r}_2d.npy'))
    positions[:, 2] = 0.5
    print('Validation before..')
    validate_positions(positions, r)
    radii = np.ones(N) * r  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass

    velocities = np.zeros((N, 3))
    velocities[:, :2] = random_uniformly_distributed_velocities(N, v0, 2)
    velocities -= np.mean(velocities, axis=0)

    # pick all particles inside a circle from center with radius 0.2 by turning mask into boolean array
    distance_to_middle_position = norm((positions - np.tile([0.5, 0.5, 0.5], reps=(len(positions), 1))), axis=1)
    mask = distance_to_middle_position < 0.2

    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii,
                                   pbc=True)

    simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=t_stop, tc=tc)
    simulation.mask = mask
    simulation.simulate_statistics_until_given_time(f'visualization_pbc_{run_number}', output_timestep=timestep,
                                                    save_positions=True)
    print('Validation after..')
    validate_positions(simulation.box_of_particles.positions, r)


def speed_distribution(particle_parameters, simulation_parameters, run_number):
    """
        Function to do an event driven simulation until a given number of collisions and then save
        the speed of all particles in order to later verity that the speeds are given by the Maxwell-
        Boltzmann distribution. It can also save the velocities in order to create equilibrium state which
        later can be used as initial values. Can be used to conduct simulations in either 2 or 3
        dimensions by specifying dimensions. 2D use 3d framwork by using zi=z=0.5 and v_zi = 0 for all i.
    :param particle_parameters: array of [N, xi, v0, radius] used to initialize a ParticleBox.
    :param simulation_parameters: array of [avg_coll_stop, output_timestep, dimensions] used to initialize a Simulation.
    :param run_number: int used such that one can run parallel simulations and write results to different files.
    """
    print(f"Run number: {run_number}")
    N, xi, v0, rad = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    avg_collisions_stop, timestep, dimensions = simulation_parameters[0], simulation_parameters[1], int(simulation_parameters[2])

    positions = np.zeros((N, 3))
    if dimensions == 2:
        positions[:, :2] = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{rad}_2d.npy'))
        positions[:, 2] = 0.5
    elif dimensions == 3:
        positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{rad}_3d.npy'))
    else:
        print('Dimensions need to be 2 or 3!')
        exit()

    radii = np.ones(N) * rad  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass

    speed_matrix = np.zeros((N, 2))  # array with mass and speed to save to file
    speed_matrix[:, 0] = mass

    velocities = np.zeros((N, 3))
    if dimensions == 2:
        velocities[:, :2] = random_uniformly_distributed_velocities(N, v0, dimensions)
    elif dimensions == 3:
        velocities = random_uniformly_distributed_velocities(N, v0, dimensions)
    else:
        print('Dimensions need to be 2 or 3!')
        exit()

    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii,
                                   pbc=False)

    simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=avg_collisions_stop)
    simulation.simulate_until_given_number_of_collisions('speed_distribution',
                                                         output_timestep=timestep,
                                                         save_positions=False)
    if run_number == -1:
        np.save(file=os.path.join(init_folder, f'eq_velocity_N_{N}_rad_{rad}_{dimensions}d.npy'),
                arr=simulation.box_of_particles.velocities)
    else:
        offset = 0
        speed_matrix[:, 1] = simulation.box_of_particles.compute_speeds()
        np.save(file=os.path.join(results_folder, f'eq_state_speed_distribution_{dimensions}d_N_{N}_{run_number+offset}'),
                arr=speed_matrix)


def get_simulation_statistics(particle_parameters, simulation_parameters, run_number):
    """
        Function to do an event driven simulation until a given time and save to file the simulation statistics,
        which is all parameters given in a simulation output at each timestep.
    :param particle_parameters: array of [N, xi, v0, radius] used to initialize a ParticleBox.
    :param simulation_parameters: array of [t_stop, output_timestep, tc] used to initialize a Simulation.
    :param run_number: int used such that one can run parallel simulations and save results to different files.
    """
    print(f"Run number: {run_number}")
    N, xi, v0, r = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    t_stop, timestep, tc = simulation_parameters[0], simulation_parameters[1], simulation_parameters[2]
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{r}_3d.npy'))
    radii = np.ones(N) * r  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass

    # velocities = random_uniformly_distributed_velocities(N, v0, 3)
    velocities = np.load(os.path.join(init_folder, f'eq_velocity_N_{N}_rad_{r}_3d.npy'))  # eq state
    np.random.shuffle(velocities)  # shuffle to create different systems for each run
    velocities -= np.mean(velocities, axis=0)

    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii,
                                   pbc=True)

    simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=t_stop, tc=tc)
    time_array, energy_array, length_collision_queue_array, average_number_of_collisions_array = \
        simulation.simulate_statistics_until_given_time('sim_statistics', output_timestep=timestep,
                                                        save_positions=False)

    statistics_matrix = np.zeros((len(time_array), 4))
    statistics_matrix[:, 0] = time_array
    statistics_matrix[:, 1] = energy_array
    statistics_matrix[:, 2] = length_collision_queue_array
    statistics_matrix[:, 3] = average_number_of_collisions_array
    # save results to file
    if tc == 0:  # to avoid np.log10(0) error
        np.save(file=os.path.join(results_folder, f'simulation_statistics_pbc_N_{N}_xi_{xi}_tstop_{t_stop}_lgtc_-inf_'
                                                  f'{run_number}'), arr=statistics_matrix)

    else:
        np.save(file=os.path.join(results_folder, f'simulation_statistics_pbc_N_{N}_xi_{xi}_tstop_{t_stop}_lgtc_'
                                                  f'{np.log10(tc)}_{run_number}'), arr=statistics_matrix)


def mean_square_displacement(particle_parameters, simulation_parameters, run_number):
    """
        Function to do an event driven simulation until a given time and save to file the msd and mss
        as a function of time for all particles. Can be run on HPC due to no output!
    :param particle_parameters: array of [N, xi, v0, radius] used to initialize a ParticleBox.
    :param simulation_parameters: array of [t_stop, output_timestep, tc] used to initialize a Simulation.
    :param run_number: int used such that one can run parallel simulations and save results to different files.
    """
    print(f"Run number: {run_number}")
    N, xi, v0, r = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    t_stop, timestep, tc = simulation_parameters[0], simulation_parameters[1], simulation_parameters[2]
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{r}_3d.npy'))
    radii = np.ones(N) * r  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass

    # velocities = random_uniformly_distributed_velocities_3d(N, v0, 3)  # all particles start with same speed
    velocities = np.load(os.path.join(init_folder, f'eq_velocity_N_{N}_rad_{r}_3d.npy'))  # eq state
    np.random.shuffle(velocities)  # shuffle to create different systems for each run
    # Nice little trick to shift reference frame. Needed for correct msd due to computing the difference
    # between current positions and the initial positions with pbc. The system in whole moves without correction!
    velocities -= np.mean(velocities, axis=0)  # little correction, but has a huge impact on msd!!

    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii,
                                   pbc=True)

    simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=t_stop, tc=tc)
    time_array, msd_array, mss_array = \
        simulation.simulate_msd_until_given_time(output_timestep=timestep, give_output=False)

    msd_matrix = np.zeros((len(time_array), 3))
    msd_matrix[:, 0] = time_array
    msd_matrix[:, 1] = msd_array
    msd_matrix[:, 2] = mss_array
    offset = 0  # for later runs where one want more runs without deleting the earlier runs
    # save to file
    np.save(file=os.path.join(results_folder, f'msd_pbc_eq_start_N_{N}_r_{r}_xi_{xi}_tstop_{t_stop}_dt_{timestep}'
                                              f'_{run_number+offset}'), arr=msd_matrix)


def mean_free_path(particle_parameters, simulation_parameters, run_number):
    """
        Function to do an event driven simulation until a given time to compute the mfp.
    :param particle_parameters: array of [N, xi, v0, radius] used to initialize a ParticleBox.
    :param simulation_parameters: array of [t_stop, output_timestep, tc] used to initialize a Simulation.
    :param run_number: int used such that one can run parallel simulations and save results to different files.
    """
    print(f"Run number: {run_number}")
    N, xi, v0, r = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    t_stop, timestep, tc = simulation_parameters[0], simulation_parameters[1], simulation_parameters[2]
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{r}_3d.npy'))
    radii = np.ones(N) * r  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass

    # velocities = random_uniformly_distributed_velocities_3d(N, v0, 3)  # all particles start with same speed
    velocities = np.load(os.path.join(init_folder, f'eq_velocity_N_{N}_rad_{r}_3d.npy'))  # eq state
    np.random.shuffle(velocities)  # shuffle to create different systems for each run
    # Nice little trick to shift reference frame. Needed for correct msd due to computing the difference
    # between current positions and the initial positions with pbc. The system in whole moves without correction!
    velocities -= np.mean(velocities, axis=0)  # little correction, but has a huge impact on msd!!

    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii,
                                   pbc=True)

    simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=t_stop, tc=tc)

    simulation.simulate_mean_free_path('mfp', output_timestep=timestep, save_positions=False)


def simulate_disease_outbreak(particle_parameters, simulation_parameters):
    """
        Function used to conduct 2D simulations of a disease outbreak.
    :param particle_parameters: array of [N, xi, v0, radius] used to initialize a ParticleBox.
    :param simulation_parameters: array of [t_stop, output_timestep, tc] used to initialize a Simulation.
    """
    N, xi, v0, r = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    t_stop, timestep, tc = simulation_parameters[0], simulation_parameters[1], simulation_parameters[2]
    positions = np.zeros((N, 3))
    positions[:, :2] = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{r}_2d.npy'))
    positions[:, 2] = 0.5

    print('Validation of positions..')
    validate_positions(positions, r)

    radii = np.ones(N) * r  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass

    home_percents = [0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    for i, home_percent in enumerate(home_percents):
        np.random.shuffle(positions)  # shuffle to get a particle starting out with the infection every time
        mass[:int(home_percent*N)] = 1e+06  # use large mass to easy identify particle who stay at home with v=0

        velocities = np.zeros((N, 3))
        velocities[:, :2] = random_uniformly_distributed_velocities(N, v0, 2)
        velocities[:int(home_percent*N), :] = 0

        box_of_particles = ParticleBox(number_of_particles=N,
                                       restitution_coefficient=xi,
                                       initial_positions=positions,
                                       initial_velocities=velocities,
                                       masses=mass,
                                       radii=radii,
                                       pbc=False)

        simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=t_stop, tc=tc)
        time_array, infected_rate, recovered_rate = \
            simulation.simulate_disease_outbreak(f'disease_outbreak_hp_{home_percent}',
                                                 output_timestep=timestep, save_positions=False)

        disease_matrix = np.zeros((len(time_array), 3))
        disease_matrix[:, 0] = time_array
        disease_matrix[:, 1] = infected_rate
        disease_matrix[:, 2] = recovered_rate

        np.save(file=os.path.join(results_folder, f'disease_matrix_N_{N}_rad_{r}_hp_{home_percent}'),
                arr=disease_matrix)
