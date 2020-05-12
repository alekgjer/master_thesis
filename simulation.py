import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import heapq
import os
import shutil

from scipy.linalg import norm

from config import plots_folder

plt.style.use('bmh')  # for nicer plots


class Simulation:
    """
        Class of a event driven simulation, where it is implemented to let particles in a ParticleBox collide until a
        given stopping criterion. All types of simulations use the ParticleBox to do the same general simulation, but
        since one is interested in different things there are several implementations. The event driven simulation
        is a systematic approach where time is incremented for each valid event until the stopping criterion is reached.
    """
    def __init__(self, box_of_particles, stopping_criterion, tc=0):
        """
            Initialize a simulation with a ParticleBox object and a stopping_criterion.
        :param box_of_particles: ParticleBox object with a square box of N particles.
        :param stopping_criterion: the stopping criterion used in the simulation. Can be given as a average number of
        collisions, or as a limit in time, or as a given average energy.
        """
        if box_of_particles is None:
            self.time_at_previous_collision = []  # must now be set when specifying the box_of_particles later
        else:
            self.time_at_previous_collision = np.zeros(box_of_particles.N)  # time at the previous collision

        self.box_of_particles = box_of_particles  # ParticleBox object
        self.simulation_time = 0
        self.max_length_collisions_queue = 1e+8  # if collision queue exceed this number, it is reinitialized.
        self.tc = tc  # variable used in the TC model to avoid inelastic collapse. tc=0 => TC model not used
        # boolean array used to indicate the set of particles to plot in red instead of standard blue. Default: None
        # mask varaible is essentially a boolean array to indicate what particles to use when computing quantities
        self.mask = None  # variable used to indicate whether or not to plot specific particles in separate color

        # simulation will run until given stopping criterion. Can be given as numb_collisions, time or energy
        self.stopping_criterion = stopping_criterion
        # characteristics which are printed as output: average number of collisions/events and average particle energy
        self.average_number_of_collisions = 0
        self.average_energy = self.box_of_particles.compute_energy()

    def print_output(self):
        """
            Function to print desired output from the simulation at each timestep.
        """
        print('--- Output ---')
        print(f"Simulation time: {self.simulation_time}")
        print(f"Priority queue elements: {len(self.box_of_particles.collision_queue)}")
        print(f"Avg energy: {self.average_energy}")
        print(f"Average number of collisions: {self.average_number_of_collisions}")

    def create_simulation_folder(self, simulation_label, output_timestep):
        """
            Function used to create a folder for the plots produced by save_particle_positions. The function creates a
            name based on the simulation_label and some simulation parameters. If the folder already exists, it is
            deleted and made again in order to save new plots.
        :param simulation_label: string given in order to identify simulation results, e.g diffProperties etc.
        :param output_timestep: parameter used to determine how often do to an output in the simulation.
        :return simulation_folder as a string.
        """
        # create detailed name for the simulation, to ensure easy identification
        simulation_folder = os.path.join(plots_folder, 'simulation_' + simulation_label +
                                         f'_N_{self.box_of_particles.N}'
                                         f'_xi_{self.box_of_particles.restitution_coefficient}_dt_{output_timestep}')
        # if not a directory: make directory. If a directory, delete it recursively and make a new
        if not os.path.isdir(simulation_folder):
            os.mkdir(simulation_folder)
        else:
            shutil.rmtree(simulation_folder)
            os.mkdir(simulation_folder)
        return simulation_folder

    def save_particle_positions(self, simulation_folder, output_number):
        # TODO: Consider making one for 3D simulations as well. But 3D plots are not so illustrative in a report...
        """
            Function to save particle positions as a png image at a output time. NB: Only possible for 2D(x, y) atm!
        :param simulation_folder: folder to save png images.
        :param output_number: int parameters stating what picture is saved in order to keep order easily.
        """
        fig, ax = plt.subplots()
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k')
        if self.mask is None:
            coll = matplotlib.collections.EllipseCollection(self.box_of_particles.radii * 2,
                                                            self.box_of_particles.radii * 2,
                                                            np.zeros_like(self.box_of_particles.radii),
                                                            offsets=self.box_of_particles.positions[:, :2], units='width',
                                                            transOffset=ax.transData)
            ax.add_collection(coll)

        else:
            coll_1 = matplotlib.collections.EllipseCollection(self.box_of_particles.radii[~self.mask] * 2,
                                                              self.box_of_particles.radii[~self.mask] * 2,
                                                              np.zeros_like(self.box_of_particles.radii[~self.mask]),
                                                              offsets=self.box_of_particles.positions[~self.mask, :2],
                                                              units='width',
                                                              transOffset=ax.transData)
            coll_2 = matplotlib.collections.EllipseCollection(self.box_of_particles.radii[self.mask] * 2,
                                                              self.box_of_particles.radii[self.mask] * 2,
                                                              np.zeros_like(self.box_of_particles.radii[self.mask]),
                                                              offsets=self.box_of_particles.positions[self.mask, :2],
                                                              units='width',
                                                              transOffset=ax.transData, facecolors='red')
            ax.add_collection(coll_1)
            ax.add_collection(coll_2)

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_aspect('equal')
        plt.savefig(os.path.join(simulation_folder, f"{output_number}.png"))
        plt.close()

    def perform_collision(self, time_at_collision, collision_tuple, t_max=None, elastic_walls=True):
        """
            FUnction that from a collision tuple, performs the collision. Performing a collision consist of updating
            the velocity of the involved particle(s) and update the parameters like collision count, average number of
            collisions, time at prev collision. Will also update the collision queue by adding new possible collisions
            for the involved particle(s).
        :param time_at_collision: time indicating the moment when the collision will occur.
        :param collision_tuple: tuple with information: (coll_time, coll entities, coll_count_comp_coll, box_comp_coll).
        :param t_max: the stopping criterion of the simulation is given by time. Used to not add collisions occurring
        after the stopping criterion if one have used a stopping criterion based on time.
        :param elastic_walls: bool value to indicate if the collisions with the different walls occur with xi=1.
        """
        dt = time_at_collision - self.simulation_time  # the increment in time until the collision
        # update positions and simulation_time by incrementing time until the collision
        self.box_of_particles.positions += self.box_of_particles.velocities * dt
        self.simulation_time += dt

        object_one = collision_tuple[1][0]  # particle number of particle one
        object_two = collision_tuple[1][1]  # particle number of particle two, or 'hw'/'vw'/'tbw' to indicate wall
        time_since_previous_collision_part_one = time_at_collision - self.time_at_previous_collision[object_one]

        # update velocities by letting a collision happen
        if object_two == 'hw':
            # update velocity of particle colliding with hw
            if elastic_walls:
                self.box_of_particles.collision_horizontal_wall(object_one, 1)
            else:
                if time_since_previous_collision_part_one < self.tc:
                    # set xi equal to 1 to avoid inelastic collapse by using the TC model
                    self.box_of_particles.collision_horizontal_wall(object_one, 1)
                else:
                    self.box_of_particles.collision_horizontal_wall(object_one,
                                                                    self.box_of_particles.restitution_coefficient)
        elif object_two == 'vw':
            # update velocity of particle in colliding with vw
            if elastic_walls:
                self.box_of_particles.collision_vertical_wall(object_one, 1)
            else:
                if time_since_previous_collision_part_one < self.tc:
                    # set xi equal to 1 to avoid inelastic collapse by using the TC model
                    self.box_of_particles.collision_vertical_wall(object_one, 1)
                else:
                    self.box_of_particles.collision_vertical_wall(object_one,
                                                                  self.box_of_particles.restitution_coefficient)
        elif object_two == 'tbw':
            # update velocity of particle in colliding with tbw
            if elastic_walls:
                self.box_of_particles.collision_tb_wall(object_one, 1)
            else:
                if time_since_previous_collision_part_one < self.tc:
                    # set xi equal to 1 to avoid inelastic collapse by using the TC model
                    self.box_of_particles.collision_tb_wall(object_one, 1)
                else:
                    self.box_of_particles.collision_tb_wall(object_one,
                                                            self.box_of_particles.restitution_coefficient)
        else:
            time_since_previous_collision_part_two = \
                time_at_collision - self.time_at_previous_collision[object_two]
            # update velocity of the two particles in the collision by particle indices, xi and box particle two
            if time_since_previous_collision_part_one < self.tc or \
                    time_since_previous_collision_part_two < self.tc:
                # in order to avoid inelastic collapse use xi=1 and use the TC model
                self.box_of_particles.collision_particles(object_one, object_two, 1, collision_tuple[3][1])
            else:
                self.box_of_particles.collision_particles(object_one, object_two,
                                                          self.box_of_particles.restitution_coefficient,
                                                          collision_tuple[3][1])

        self.box_of_particles.collision_count_particles[object_one] += 1  # update collision count

        if object_two not in ['hw', 'vw', 'tbw']:  # if there is a second particle involved
            self.box_of_particles.collision_count_particles[object_two] += 1  # update collision count
            # get new collisions for object two
            self.box_of_particles.update_queue_new_collisions_particle(object_two, self.simulation_time, t_max)
            self.time_at_previous_collision[object_two] = time_at_collision  # add time at collision
        # get new collisions for object one
        self.box_of_particles.update_queue_new_collisions_particle(object_one, self.simulation_time, t_max)
        self.time_at_previous_collision[object_one] = time_at_collision  # add time at collision
        # update average number of collisions/events since one or two particles have been in a collision
        self.average_number_of_collisions = np.mean(self.box_of_particles.collision_count_particles)

        # if the collision_queue has too many entries, it is reset and initialized as in the start of the simulations
        if len(self.box_of_particles.collision_queue) > self.max_length_collisions_queue:
            self.box_of_particles.collision_queue = []
            self.box_of_particles.create_initial_priority_queue(t_max)

    def simulate_until_given_number_of_collisions(self, simulation_label, output_timestep=1.0, save_positions=False):
        """
            Implementation of the event driven simulation where the stopping criterion is given as a limit of
            how much one want the average number of collisions to be. Is useful when wanting to create equilibrium
            situations and look at parameters after the particles have collided until a given threshold.
        :param simulation_label: string containing information about the simulation to identify simulation.
        :param output_timestep: parameter used to determine how often do to an output in the simulation.
        :param save_positions: boolean variable used to indicate if one want to save positions. Since saving takes some
        capacity and it is not so interesting when doing multiple runs one has the option to not save positions.
        """
        print('Simulate until a given average number of collisions..')
        print(f'N: {self.box_of_particles.N} and xi: {self.box_of_particles.restitution_coefficient}..')
        print('---------------------')
        # create folder in order to save particle positions as a png files throughout the simulation
        simulation_folder = ""
        if save_positions:
            simulation_folder = self.create_simulation_folder(simulation_label, output_timestep)

        print('Creating initial queue..')
        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output
        self.box_of_particles.create_initial_priority_queue()  # Initialize the queue with all starting collisions

        self.average_energy = self.box_of_particles.compute_energy()  # initial energy for all particles

        # give initial output and save particle positions
        self.print_output()
        if save_positions:
            self.save_particle_positions(simulation_folder, output_number)
        next_output_time += output_timestep
        output_number += 1

        print('Event driven simulation in progress..')
        # run until the average number of collisions has reached the stopping criterion
        while self.average_number_of_collisions < self.stopping_criterion:
            collision_tuple = heapq.heappop(self.box_of_particles.collision_queue)  # pop the earliest event/collision
            time_at_collision = collision_tuple[0]  # extract time at collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.box_of_particles.positions += self.box_of_particles.velocities * dt
                self.simulation_time += dt

                self.average_energy = self.box_of_particles.compute_energy()  # compute average energy

                # give output and save particle positions
                self.print_output()
                if save_positions:
                    self.save_particle_positions(simulation_folder, output_number)

                next_output_time += output_timestep
                output_number += 1
            # if valid collision -> do it! If not discard it and try the next earliest etc.
            if self.box_of_particles.valid_collision(collision_tuple):
                self.perform_collision(time_at_collision, collision_tuple)

        print('Simulation done!')
        print('---------------------')

    def simulate_statistics_until_given_time(self, simulation_label, output_timestep=1.0, save_positions=False):
        """
            Implementation of the event driven simulation where the stopping criterion is given as a limit of
            how much one want the simulation time to be. It is useful for looking at a property as a function of time.
            Atm computes mean energy of all particles, priority queue elements and the average number of collisions
             at all output times. This is the information given in a output of the simulation.
        :param simulation_label: string containing information about the simulation to identify simulation.
        :param output_timestep: parameter used to determine how often do to an output in the simulation.
        :param save_positions: boolean variable used to indicate if one want to save positions. Since saving takes some
        capacity and it is not so interesting when doing multiple runs one have the option to not save positions.
        :return time_array, energy_array, length_priority_queue_array, average_number_of_collisions_array
        """
        print('Simulate until a given simulation time')
        print(f'N: {self.box_of_particles.N} and xi: {self.box_of_particles.restitution_coefficient}..')
        print('---------------------')
        # create folder in order to save particle positions as a png files throughout the simulation
        simulation_folder = ""
        if save_positions:
            simulation_folder = self.create_simulation_folder(simulation_label, output_timestep)

        print('Creating initial queue..')

        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output
        # Initialize the queue with all starting collisions
        self.box_of_particles.create_initial_priority_queue(t_max=self.stopping_criterion)

        # initial energy for all particles
        self.average_energy = self.box_of_particles.compute_energy()

        # give initial output and save particle positions
        self.print_output()
        if save_positions:
            self.save_particle_positions(simulation_folder, output_number)
        next_output_time += output_timestep
        output_number += 1

        time_array = np.zeros(int(self.stopping_criterion/output_timestep)+1)  # array for time at all output times
        energy_array = np.zeros_like(time_array)  # array for average energy of at particles at all output times
        length_priority_queue_array = np.zeros_like(time_array)  # array for elements in the queue at all output times
        average_number_of_collisions_array = np.zeros_like(time_array)  # array for average number of collisions

        energy_array[0] = self.average_energy
        length_priority_queue_array[0] = len(self.box_of_particles.collision_queue)

        print('Event driven simulation in progress..')
        # run until the simulation time has reached the stopping criterion
        while self.simulation_time < self.stopping_criterion:
            collision_tuple = heapq.heappop(self.box_of_particles.collision_queue)  # pop the earliest collisions/event
            time_at_collision = collision_tuple[0]  # extract time at collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.box_of_particles.positions += self.box_of_particles.velocities * dt
                self.simulation_time += dt

                time_array[output_number] = self.simulation_time

                # average energy for all particles, m0 particles and m particles
                self.average_energy = self.box_of_particles.compute_energy()

                energy_array[output_number] = self.average_energy
                length_priority_queue_array[output_number] = len(self.box_of_particles.collision_queue)
                average_number_of_collisions_array[output_number] = self.average_number_of_collisions

                # give output and save particle positions
                self.print_output()
                if save_positions:
                    self.save_particle_positions(simulation_folder, output_number)

                next_output_time += output_timestep
                output_number += 1
            # if valid collision -> do it! If not discard it and try the next earliest etc.
            if self.box_of_particles.valid_collision(collision_tuple):
                self.perform_collision(time_at_collision, collision_tuple, t_max=self.stopping_criterion)

        print('Simulation done!')
        print('---------------------')
        return time_array, energy_array, length_priority_queue_array, average_number_of_collisions_array

    def simulate_msd_until_given_time(self, output_timestep=1.0, give_output=True):
        """
            Implementation of the event driven simulation where the stopping criterion is given as a limit of
            how much one want the simulation time to be. Is useful when looking at a property as a function of time.
            Atm computes mean square displacement and mean quadratic speed for all particles. This function can also
            be run on HPC.
        :param output_timestep: parameter used to determine how often do to an output in the simulation.
        :param give_output: boolean variable used to indicate if one want to do simulation output. It does take some
        time, and since we want to use this code on HPC this option is used to reduce the output.
        :return time_array, mean_quadratic_distance_array, mean_quadratic_speed_array.
        """
        if give_output:
            print(f'Simulate until a given t_stop: {self.stopping_criterion} is reached')
            print(f'N: {self.box_of_particles.N} and xi: {self.box_of_particles.restitution_coefficient}..')
            print('---------------------')
            print('Creating initial queue..')

        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output and get next output time
        # Initialize the queue with all starting collisions
        self.box_of_particles.create_initial_priority_queue(t_max=self.stopping_criterion)

        # initial energy for all particles
        self.average_energy = self.box_of_particles.compute_energy()

        if give_output:
            self.print_output()

        output_number += 1
        # choose different setup for output data
        if output_timestep > 0:  # for dt > 0 we use a constant resolution in time
            time_array = np.zeros(int(self.stopping_criterion / output_timestep) + 1)  # array for time
            next_output_time += output_timestep
        else:  # for dt = 0 we use a logspace to get nice plots on logarithmic scales
            if self.stopping_criterion >= 1000:
                number_of_outputs = 100  # good value of point for 1000 and above
            else:
                number_of_outputs = 50  # good value of points for 10 and 100

            time_array = np.zeros(number_of_outputs+1)  # array for time
            # we add the number of outputs output times to the time array list in order to avoid using to time arrays
            time_array[1:] = np.round(np.logspace(-2, np.log10(self.stopping_criterion), number_of_outputs), decimals=3)
            next_output_time = time_array[output_number]

        mean_square_displacement_array = np.zeros_like(time_array)  # array for msd
        mean_square_speed_array = np.zeros_like(time_array)  # array for mss
        mean_square_speed_array[0] = self.box_of_particles.compute_mean_square_speed()

        if give_output:
            print('Event driven simulation in progress..')
        # run until the simulation time has reached the stopping criterion
        while self.simulation_time < self.stopping_criterion:
            collision_tuple = heapq.heappop(self.box_of_particles.collision_queue)  # pop the earliest collision/event
            time_at_collision = collision_tuple[0]  # extract time at collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.box_of_particles.positions += self.box_of_particles.velocities * dt
                self.simulation_time += dt

                time_array[output_number] = self.simulation_time  # save time of output

                # compute mean square displacement from starting position for all particles
                mean_square_displacement_array[output_number] = self.box_of_particles.compute_mean_square_displacement()
                # compute mean_quadratic_speed for all particles
                mean_square_speed_array[output_number] = self.box_of_particles.compute_mean_square_speed()

                # update average energy for all particles
                self.average_energy = self.box_of_particles.compute_energy()

                if give_output:
                    self.print_output()

                output_number += 1

                if output_timestep > 0:  # for constant resolution in time the next output is after dt time
                    next_output_time += output_timestep
                else:
                    if output_number == len(time_array):  # in order to avoid error on last output we use inf as next
                        next_output_time = np.inf
                    else:
                        # for logarithmic timescale the next output is given as the next element in the time array
                        next_output_time = time_array[output_number]

            # if valid collision -> do it! If not discard it and try the next earliest etc.
            if self.box_of_particles.valid_collision(collision_tuple):
                self.perform_collision(time_at_collision, collision_tuple, t_max=self.stopping_criterion)
        if give_output:
            print('Simulation done!')
            print('---------------------')
        return time_array, mean_square_displacement_array, mean_square_speed_array

    def simulate_mean_free_path(self, simulation_label, output_timestep=1.0, save_positions=False):
        """
            Implementation of the event driven simulation to do simulations of the mfp
        :param simulation_label: string containing information about the simulation to identify simulation.
        :param output_timestep: parameter used to determine how often do to an output in the simulation.
        :param save_positions: boolean variable used to indicate if one want to save positions. Since saving takes some
        capacity and it is not so interesting when doing multiple runs one have the option to not save positions.
        """
        print('Simulate until a given simulation time is reached')
        print(f'N: {self.box_of_particles.N} and xi: {self.box_of_particles.restitution_coefficient}..')
        print('---------------------')
        # create folder in order to save particle positions as a png files throughout the simulation
        simulation_folder = ""
        if save_positions:
            simulation_folder = self.create_simulation_folder(simulation_label, output_timestep)

        print('Creating initial queue..')

        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output
        # Initialize the queue with all starting collisions
        self.box_of_particles.create_initial_priority_queue(t_max=self.stopping_criterion)

        # initial energy for all particles
        self.average_energy = self.box_of_particles.compute_energy()

        # give initial output and save particle positions
        self.print_output()
        if save_positions:
            self.save_particle_positions(simulation_folder, output_number)
        next_output_time += output_timestep
        output_number += 1

        distance_between_collisions = np.zeros(self.box_of_particles.N)
        number_of_particle_particle_collisions = np.zeros_like(distance_between_collisions)

        print('Event driven simulation in progress..')
        # run until the simulation time has reached the stopping criterion
        while self.simulation_time < self.stopping_criterion:
            collision_tuple = heapq.heappop(self.box_of_particles.collision_queue)  # pop the earliest collision/event
            time_at_collision = collision_tuple[0]  # extract time at collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.box_of_particles.positions += self.box_of_particles.velocities * dt
                self.simulation_time += dt

                # update average energy for all particles
                self.average_energy = self.box_of_particles.compute_energy()
                # give output and save particle positions
                self.print_output()
                print(f'mfp: {np.mean(distance_between_collisions / number_of_particle_particle_collisions)}')
                if save_positions:
                    self.save_particle_positions(simulation_folder, output_number)

                next_output_time += output_timestep
                output_number += 1
            # if valid collision -> do it! If not discard it and try the next earliest etc.
            if self.box_of_particles.valid_collision(collision_tuple):
                object_one = collision_tuple[1][0]
                object_two = collision_tuple[1][1]
                if object_two not in ['hw', 'vw', 'tbw']:  # make sure it is a particle particle collision
                    # get speed. Update distance as speed times time difference. Add collision
                    speed_one = norm(self.box_of_particles.velocities[object_one, :])
                    distance_between_collisions[object_one] += speed_one * (
                                time_at_collision - self.time_at_previous_collision[object_one])
                    speed_two = norm(self.box_of_particles.velocities[object_two, :])
                    distance_between_collisions[object_two] += speed_two * (
                            time_at_collision - self.time_at_previous_collision[object_two])
                    number_of_particle_particle_collisions[object_one] += 1
                    number_of_particle_particle_collisions[object_two] += 1
                self.perform_collision(time_at_collision, collision_tuple, t_max=self.stopping_criterion)

        print('Simulation done!')
        print('---------------------')
        # mfp as the mean (tot_distance/number_of_collisions)
        print(f'mfp: {np.mean(distance_between_collisions/number_of_particle_particle_collisions)}')

    def perform_outbreak_collision(self, time_at_collision, collision_tuple, t_max=None):
        """
            FUnction that from a collision tuple, performs the collision. Performing a collision consist of updating
            the velocity of the involved particle(s) and update the parameters like collision count, average number of
            collisions, time at prev collision. Will also update the collision queue by adding new possible collisions
            for the involved particle(s). Is used for simulation of disease outbreak, where the use of high mass is
            applied to serve as good citizens who do not go outside.
        :param time_at_collision: time indicating the moment when the collision will occur.
        :param collision_tuple: tuple with information: (coll_time, coll entities, coll_count_comp_coll, box_comp_coll).
        :param t_max: the stopping criterion of the simulation is given by time. Used to not add collisions occurring
        after the stopping criterion if one have used a stopping criterion based on time.
        """
        dt = time_at_collision - self.simulation_time  # the increment in time until the collision
        # update positions and simulation_time by incrementing time until the collision
        self.box_of_particles.positions += self.box_of_particles.velocities * dt
        self.simulation_time += dt

        object_one = collision_tuple[1][0]  # particle number of particle one
        object_two = collision_tuple[1][1]  # particle number of particle two, or 'hw'/'vw'/'tbw' to indicate wall

        # update velocities by letting a collision happen
        if object_two == 'hw':
            # update velocity of particle colliding with hw
            self.box_of_particles.collision_horizontal_wall(object_one, 1)

        elif object_two == 'vw':
            # update velocity of particle in colliding with vw
            self.box_of_particles.collision_vertical_wall(object_one, 1)

        elif object_two == 'tbw':
            print('This wall should get hit in 3D..')
            exit()
        else:
            # update velocity of the two particles in the collision by particle indices, xi and box particle two
            self.box_of_particles.collision_particles(object_one, object_two,
                                                      self.box_of_particles.restitution_coefficient,
                                                      collision_tuple[3][1])

            if self.box_of_particles.masses[object_one] > 1e+05:
                self.box_of_particles.velocities[object_one, :] = 0
            if self.box_of_particles.masses[object_two] > 1e+05:
                self.box_of_particles.velocities[object_two, :] = 0

        self.box_of_particles.collision_count_particles[object_one] += 1  # update collision count

        if object_two not in ['hw', 'vw', 'tbw']:  # if there is a second particle involved
            self.box_of_particles.collision_count_particles[object_two] += 1  # update collision count
            # get new collisions for object two
            self.box_of_particles.update_queue_new_collisions_particle(object_two, self.simulation_time, t_max)
            self.time_at_previous_collision[object_two] = time_at_collision  # add time at collision
        # get new collisions for object one
        self.box_of_particles.update_queue_new_collisions_particle(object_one, self.simulation_time, t_max)
        self.time_at_previous_collision[object_one] = time_at_collision  # add time at collision
        # update average number of collisions/events since one or two particles have been in a collision
        self.average_number_of_collisions = np.mean(self.box_of_particles.collision_count_particles)

        # if the collision_queue has too many entries, it is reset and initialized as in the start of the simulations
        if len(self.box_of_particles.collision_queue) > self.max_length_collisions_queue:
            self.box_of_particles.collision_queue = []
            self.box_of_particles.create_initial_priority_queue(t_max)

    def simulate_disease_outbreak(self, simulation_label, output_timestep=1.0, save_positions=False):
        """
            Implementation of the event driven simulation to do simulations of a disease outbreak
        :param simulation_label: string containing information about the simulation to identify simulation.
        :param output_timestep: parameter used to determine how often do to an output in the simulation.
        :param save_positions: boolean variable used to indicate if one want to save positions. Since saving takes some
        capacity and it is not so interesting when doing multiple runs one have the option to not save positions.
        return: infected_particles, time_of_infection, infection_count arrays containing information about each
        particle. First is 0/1 for healthy/sick, time of infection in simulation of how many did the particle infect.
        """
        print('Simulate until a given simulation time is reached')
        print(f'N: {self.box_of_particles.N} and xi: {self.box_of_particles.restitution_coefficient}..')
        print('---------------------')
        # create folder in order to save particle positions as a png files throughout the simulation
        simulation_folder = ""
        if save_positions:
            simulation_folder = self.create_simulation_folder(simulation_label, output_timestep)

        infected_particles = np.zeros(self.box_of_particles.N)  # array used to tell if 0: healthy or 1: sick
        recovered_particles = np.zeros_like(infected_particles)  # array used to tell if 0: acceptable or 1: recovered
        time_of_infection = np.ones_like(infected_particles) * np.inf  # array used to save the time of infection
        infection_count = np.zeros_like(infected_particles)  # number of how many other particles the particles infected
        infected_particles[-1] = 1  # the last particle starts out infected. Due to [:home_perfect*N] are at home
        time_of_infection[-1] = 0
        self.mask = infected_particles == 1  # mask used to get other color (red) for the sick particles

        print('Creating initial queue..')

        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output
        # Initialize the queue with all starting collisions
        self.box_of_particles.create_initial_priority_queue(t_max=self.stopping_criterion)

        # initial energy for all particles
        self.average_energy = self.box_of_particles.compute_energy()

        # give initial output and save particle positions
        self.print_output()
        if save_positions:
            self.save_particle_positions(simulation_folder, output_number)
        next_output_time += output_timestep
        output_number += 1

        disease_period = 0.2

        time_array = np.arange(int(self.stopping_criterion/output_timestep)+1)*output_timestep
        infected_rate = np.zeros_like(time_array)
        recovered_rate = np.ones_like(time_array)
        infected_rate[0] = np.mean(infected_particles)
        recovered_rate[0] = 0

        print('Event driven simulation in progress..')
        # run until the simulation time has reached the stopping criterion
        while self.simulation_time < self.stopping_criterion:
            collision_tuple = heapq.heappop(
                self.box_of_particles.collision_queue)  # pop the earliest collision/event
            time_at_collision = collision_tuple[0]  # extract time at collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.box_of_particles.positions += self.box_of_particles.velocities * dt
                self.simulation_time += dt

                infected_rate[output_number] = np.mean(infected_particles)
                recovered_rate[output_number] = np.mean(recovered_particles)

                # update average energy for all particles
                self.average_energy = self.box_of_particles.compute_energy()
                # give output and save particle positions
                self.print_output()
                self.mask = infected_particles == 1  # update mask of infected particles
                if save_positions:
                    self.save_particle_positions(simulation_folder, output_number)

                next_output_time += output_timestep
                output_number += 1
            # if valid collision -> do it! If not discard it and try the next earliest etc.
            if self.box_of_particles.valid_collision(collision_tuple):
                object_one = collision_tuple[1][0]
                object_two = collision_tuple[1][1]
                if self.simulation_time > (disease_period + time_of_infection[object_one]) and recovered_particles[
                   object_one] == 0:
                    recovered_particles[object_one] = 1
                    infected_particles[object_one] = 0
                if object_two not in ['vw', 'hw', 'tbw']:
                    if self.simulation_time > (disease_period + time_of_infection[object_two]) and recovered_particles[
                       object_two] == 0:
                        recovered_particles[object_two] = 1
                        infected_particles[object_two] = 0
                    if infected_particles[object_one] == 1 and recovered_particles[object_two] == 0:
                        if infected_particles[object_two] == 0:
                            infected_particles[object_two] = 1
                            time_of_infection[object_two] = time_at_collision
                            infection_count[object_one] += 1
                    elif infected_particles[object_two] == 1 and recovered_particles[object_one] == 0:
                        if infected_particles[object_one] == 0:
                            infected_particles[object_one] = 1
                            time_of_infection[object_one] = time_at_collision
                            infection_count[object_two] += 1

                self.perform_outbreak_collision(time_at_collision, collision_tuple, t_max=self.stopping_criterion)
                if np.sum(recovered_particles) == self.box_of_particles.N:
                    # early stop when have been infected and then recovered
                    print('Everybody got infected :(..')
                    return time_array, infected_rate, recovered_rate

        print('Simulation done!')
        print('---------------------')
        return time_array, infected_rate, recovered_rate
