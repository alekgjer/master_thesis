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
        is a systematic approach where time is incremented for each valid event.
    """
    def __init__(self, box_of_particles, stopping_criterion):
        """
            Initialize a simulation with a ParticleBox object and a stopping_criterion
        :param box_of_particles: ParticleBox object with a square box of N particles
        :param stopping_criterion: the stopping criterion used in the simulation. Can be given as a average number of
        collisions, or as a limit in time, or as a given average energy.
        """
        if box_of_particles is None:
            self.time_at_previous_collision = []  # must now be set when specifying the box_of_particles later
        else:
            self.time_at_previous_collision = np.zeros(box_of_particles.N)  # time at the previous collision

        self.box_of_particles = box_of_particles  # ParticleBox object
        self.simulation_time = 0
        self.tc = 0  # variable used in the TC model to avoid inelastic collapse. tc=0 => TC model not used
        # boolean array used to indicate the set of particles to plot in red instead of standard blue. Default: None
        # mask varaible is essentially a boolean array to indicate what particles to use when computing quantities
        self.mask = None  # variable used to indicate whether or not to plot specific particles in separate color

        # simulation will run until given stopping criterion. Can be given as numb_collisions, time or energy
        self.stopping_criterion = stopping_criterion
        self.average_number_of_collisions = 0

    def reset(self):
        """
            Function to reinitialize some values in the simulation. It used such than one can use Simulation multiple
            times by making sure to update the system given in box_of_particles.
        """
        self.simulation_time = 0
        self.time_at_previous_collision = np.zeros(self.box_of_particles.N)
        self.average_number_of_collisions = 0

    def print_output(self, average_number_of_collisions, avg_energy):
        """
            Function to print desired output from the simulation
        """
        print('--- Output ---')
        print(f"Simulation time: {self.simulation_time}")
        print(f"Priority queue elements: {len(self.box_of_particles.collision_queue)}")
        print(f"Avg energy: {avg_energy}")
        print(f"Average number of collisions: {average_number_of_collisions}")

    def create_simulation_folder(self, simulation_label):
        """
            Function used to create a folder for the plots produced by save_particle_positions. The function creates a
            name based on the simulation_label and some simulation parameters. If the folder already exists, it is
            deleted and made again in order to save new plots.
        :param simulation_label: string given in order to identify simulation results, e.g diffProperties etc
        :return simulation_folder as a string
        """
        simulation_folder = os.path.join(plots_folder, 'simulation_' + simulation_label +
                                         f'_N_{self.box_of_particles.N}'
                                         f'_xi_{self.box_of_particles.restitution_coefficient}')
        if not os.path.isdir(simulation_folder):
            os.mkdir(simulation_folder)
        else:
            shutil.rmtree(simulation_folder)
            os.mkdir(simulation_folder)
        return simulation_folder

    def save_particle_positions(self, simulation_folder, picture_number):
        """
            Function to save particle positions as a png image at a output time
        :param simulation_folder: folder to save png images
        :param picture_number: int parameters stating what picture is saved in order to keep order easily
        """
        fig, ax = plt.subplots()
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k')
        if self.mask is None:
            coll = matplotlib.collections.EllipseCollection(self.box_of_particles.radii * 2,
                                                            self.box_of_particles.radii * 2,
                                                            np.zeros_like(self.box_of_particles.radii),
                                                            offsets=self.box_of_particles.positions, units='width',
                                                            transOffset=ax.transData)
            ax.add_collection(coll)

        else:
            coll_1 = matplotlib.collections.EllipseCollection(self.box_of_particles.radii[~self.mask] * 2,
                                                              self.box_of_particles.radii[~self.mask] * 2,
                                                              np.zeros_like(self.box_of_particles.radii[~self.mask]),
                                                              offsets=self.box_of_particles.positions[~self.mask, :],
                                                              units='width',
                                                              transOffset=ax.transData)
            coll_2 = matplotlib.collections.EllipseCollection(self.box_of_particles.radii[self.mask] * 2,
                                                              self.box_of_particles.radii[self.mask] * 2,
                                                              np.zeros_like(self.box_of_particles.radii[self.mask]),
                                                              offsets=self.box_of_particles.positions[self.mask, :],
                                                              units='width',
                                                              transOffset=ax.transData, facecolors='red')
            ax.add_collection(coll_1)
            ax.add_collection(coll_2)

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        plt.savefig(os.path.join(simulation_folder, f"{picture_number}.png"))
        plt.close()

    def perform_collision(self, time_at_collision, collision_tuple, t_max=None):
        """
            FUnction that from a collision tuple, performs the collision. Performing a collision consist of updating
            the velocity of the involved particle(s) and update the parameters like collision count, average number of
            collisions, time at prev collision. Will also update the collision queue by adding new possible collisions
            for the involved particle(s).
        :param time_at_collision: time indicating the moment when the collision will occur
        :param collision_tuple: tuple with information: (coll_time, coll entities, coll_count_comp_coll)
        :param t_max: the stopping criterion of the simulation is given by time. Used to not add collisions occurring
        after the stopping criterion if one have used a stopping criterion based on time.
        """
        dt = time_at_collision - self.simulation_time  # the increment in time until the collision
        # update positions and simulation_time by incrementing time until the collision
        self.box_of_particles.positions += self.box_of_particles.velocities * dt
        self.simulation_time += dt

        object_one = collision_tuple[1][0]  # particle number of particle one
        object_two = collision_tuple[1][1]  # particle number of particle two, or 'hw'/'vw' to indicate wall
        time_since_previous_collision_part_one = time_at_collision - self.time_at_previous_collision[object_one]

        # update velocities by letting a collision happen
        if object_two == 'hw':
            # update velocity of particle colliding with hw
            if time_since_previous_collision_part_one < self.tc:
                # set xi equal to 1 to avoid inelastic collapse by using the TC model
                self.box_of_particles.collision_horizontal_wall(object_one, 1)
            else:
                self.box_of_particles.collision_horizontal_wall(object_one,
                                                                self.box_of_particles.restitution_coefficient)
        elif object_two == 'vw':
            # update velocity of particle in colliding with vw
            if time_since_previous_collision_part_one < self.tc:
                # set xi equal to 1 to avoid inelastic collapse by using the TC model
                self.box_of_particles.collision_vertical_wall(object_one, 1)
            else:
                self.box_of_particles.collision_vertical_wall(object_one,
                                                              self.box_of_particles.restitution_coefficient)
        else:
            time_since_previous_collision_part_two = \
                time_at_collision - self.time_at_previous_collision[object_two]
            # update velocity of the two particles in the collision
            if time_since_previous_collision_part_one < self.tc or \
                    time_since_previous_collision_part_two < self.tc:
                # in order to avoid inelastic collapse use xi=1 and use the TC model
                self.box_of_particles.collision_particles(object_one, object_two, 1)
            else:
                self.box_of_particles.collision_particles(object_one, object_two,
                                                          self.box_of_particles.restitution_coefficient)

        self.box_of_particles.collision_count_particles[object_one] += 1  # update collision count
        if object_two not in ['hw', 'vw']:  # if there is a second particle involved
            self.box_of_particles.collision_count_particles[object_two] += 1  # update collision count
            # get new collisions for object two
            self.box_of_particles.update_queue_new_collisions_particle(object_two, self.simulation_time, t_max)
            self.time_at_previous_collision[object_two] = time_at_collision  # add time at collision
        # get new collisions for object one
        self.box_of_particles.update_queue_new_collisions_particle(object_one, self.simulation_time, t_max)
        self.time_at_previous_collision[object_one] = time_at_collision  # add time at collision
        # update average number of collisions since one or two particles have been in a collision
        self.average_number_of_collisions = np.mean(self.box_of_particles.collision_count_particles)

    def simulate_until_given_number_of_collisions(self, simulation_label, output_timestep=1.0, save_positions=True):
        """
            Implementation of the event driven simulation where the stopping criterion is given as a limit of
            how much one want the average number of collisions to be. Is useful when wanting to create equilibrium
            situations and look at parameters after the particles have collided until a given threshold.
        :param simulation_label: string containing information about the simulation to identify simulation
        :param output_timestep: parameter used to determine how often do to an output in the simulation
        :param save_positions: boolean variable used to indicate if one want to save positions. Since saving takes some
        capacity and it is not so interesting when doing multiple runs one have the option to not save positions.
        """
        print('Simulate until a given average number of collisions..')
        print(f'N: {self.box_of_particles.N} and xi: {self.box_of_particles.restitution_coefficient}..')
        print('---------------------')
        # create folder in order to save particle positions as a png files throughout the simulation
        simulation_folder = ""
        if save_positions:
            simulation_folder = self.create_simulation_folder(simulation_label)

        print('Creating initial queue..')
        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output
        self.box_of_particles.create_initial_priority_queue()  # Initialize the queue with all starting collisions

        avg_energy = self.box_of_particles.compute_energy()  # initial energy for all particles

        # give initial output and save particle positions
        self.print_output(0, avg_energy)
        if save_positions:
            self.save_particle_positions(simulation_folder, output_number)
        next_output_time += output_timestep
        output_number += 1

        print('Event driven simulation in progress..')
        # run until the average number of collisions has reached the stopping criterion
        while self.average_number_of_collisions < self.stopping_criterion:
            collision_tuple = heapq.heappop(self.box_of_particles.collision_queue)  # pop the earliest element
            time_at_collision = collision_tuple[0]  # extract time at collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.box_of_particles.positions += self.box_of_particles.velocities * dt
                self.simulation_time += dt

                avg_energy = self.box_of_particles.compute_energy()  # compute average energy

                # give output and save particle positions
                self.print_output(self.average_number_of_collisions, avg_energy)
                if save_positions:
                    self.save_particle_positions(simulation_folder, output_number)

                next_output_time += output_timestep
                output_number += 1

            if self.box_of_particles.valid_collision(collision_tuple):
                self.perform_collision(time_at_collision, collision_tuple)

        print('Simulation done!')
        print('---------------------')

    def simulate_statistics_until_given_time(self, simulation_label, output_timestep=1.0, save_positions=True):
        """
            Implementation of the event driven simulation where the stopping criterion is given as a limit of
            how much one want the simulation time to be. Is useful when looking at a property as a function of time.
            Atm computes mean energy and speed of all particles at all output times. Essentially gives statistics
        :param simulation_label: string containing information about the simulation to identify simulation
        :param output_timestep: parameter used to determine how often do to an output in the simulation
        :param save_positions: boolean variable used to indicate if one want to save positions. Since saving takes some
        capacity and it is not so interesting when doing multiple runs one have the option to not save positions.
        :return time_array, energy_array, mean_speed_array
        """
        print('Simulate until a given simulation time')
        print(f'N: {self.box_of_particles.N} and xi: {self.box_of_particles.restitution_coefficient}..')
        print('---------------------')
        # create folder in order to save particle positions as a png files throughout the simulation
        simulation_folder = ""
        if save_positions:
            simulation_folder = self.create_simulation_folder(simulation_label)

        print('Creating initial queue..')

        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output
        # Initialize the queue with all starting collisions
        self.box_of_particles.create_initial_priority_queue(t_max=self.stopping_criterion)

        # initial energy for all particles
        avg_energy = self.box_of_particles.compute_energy()

        # give initial output and save particle positions
        self.print_output(self.average_number_of_collisions, avg_energy)
        if save_positions:
            self.save_particle_positions(simulation_folder, output_number)
        next_output_time += output_timestep
        output_number += 1

        time_array = np.zeros(int(self.stopping_criterion/output_timestep)+1)  # array for time at all output times
        energy_array = np.zeros_like(time_array)  # array for average energy of at particles at all output times
        mean_speed_array = np.zeros_like(time_array)  # array for average speed at all output times

        energy_array[0] = avg_energy
        mean_speed_array[0] = np.mean(norm(self.box_of_particles.velocities, axis=1))

        print('Event driven simulation in progress..')
        # run until the simulation time has reached the stopping criterion
        while self.simulation_time < self.stopping_criterion:
            collision_tuple = heapq.heappop(self.box_of_particles.collision_queue)  # pop the earliest element
            time_at_collision = collision_tuple[0]  # extract time at collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.box_of_particles.positions += self.box_of_particles.velocities * dt
                self.simulation_time += dt

                time_array[output_number] = self.simulation_time

                # average energy for all particles, m0 particles and m particles
                avg_energy = self.box_of_particles.compute_energy()

                energy_array[output_number] = avg_energy
                mean_speed_array[output_number] = np.mean(norm(self.box_of_particles.velocities, axis=1))

                # give output and save particle positions
                self.print_output(self.average_number_of_collisions, avg_energy)
                if save_positions:
                    self.save_particle_positions(simulation_folder, output_number)

                next_output_time += output_timestep
                output_number += 1

            if self.box_of_particles.valid_collision(collision_tuple):
                self.perform_collision(time_at_collision, collision_tuple, t_max=self.stopping_criterion)

        print('Simulation done!')
        print('---------------------')
        return time_array, energy_array, mean_speed_array

    def simulate_until_given_energy(self, simulation_label, output_timestep=1.0, save_positions=True):
        """
            Implementation of the event driven simulation where the stopping criterion is given as a limit of
            how much one want the average energy time to be. Is useful when looking at the end result of a system where
            the energy is reduced(xi<1). Can be used to simulate impact of projectile and create fractal by enabling
            sticky particles.
            Atm computes mean energy and speed of all particles at all output times.
        :param simulation_label: string containing information about the simulation to identify simulation
        :param output_timestep: parameter used to determine how often do to an output in the simulation
        :param save_positions: boolean variable used to indicate if one want to save positions. Since saving takes some
        capacity and it is not so interesting when doing multiple runs one have the option to not save positions.
        """
        print('Simulate until a given energy is reached')
        print(f'N: {self.box_of_particles.N} and xi: {self.box_of_particles.restitution_coefficient}..')
        print('---------------------')
        # create folder in order to save particle positions as a png files throughout the simulation
        simulation_folder = ""
        if save_positions:
            simulation_folder = self.create_simulation_folder(simulation_label)

        print('Creating initial queue..')

        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output
        self.box_of_particles.create_initial_priority_queue()  # Initialize the queue with all starting collisions

        # initial average energy for all particles
        avg_energy = self.box_of_particles.compute_energy()

        # give initial output and save particle positions
        self.print_output(self.average_number_of_collisions, avg_energy)
        if save_positions:
            self.save_particle_positions(simulation_folder, output_number)
        next_output_time += output_timestep
        output_number += 1

        print('Event driven simulation in progress..')
        # run until the simulation time has reached the stopping criterion
        while avg_energy > self.stopping_criterion:
            collision_tuple = heapq.heappop(self.box_of_particles.collision_queue)  # pop the earliest element
            time_at_collision = collision_tuple[0]  # extract time at collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.box_of_particles.positions += self.box_of_particles.velocities * dt
                self.simulation_time += dt

                # average energy for all particles
                avg_energy = self.box_of_particles.compute_energy()

                # give output and save particle positions
                self.print_output(self.average_number_of_collisions, avg_energy)
                if save_positions:
                    self.save_particle_positions(simulation_folder, output_number)

                next_output_time += output_timestep
                output_number += 1
            if self.box_of_particles.valid_collision(collision_tuple):
                self.perform_collision(time_at_collision, collision_tuple)
                # recompute the avg_energy for the stopping criterion
                avg_energy = self.box_of_particles.compute_energy()

        print('Simulation done!')
        print('---------------------')

    def simulate_until_given_time_mask_quantities(self, simulation_label, output_timestep=1.0,
                                                  update_positions=False, save_positions=True):
        """
            Implementation of the event driven simulation where the stopping criterion is given as a limit of
            how much one want the simulation time to be. Is useful when looking at a property as a function of time.
            Atm will compute quantities for a given mask, such as mean square displacement and mean quadratic speed
        :param simulation_label: string containing information about the simulation to identify simulation
        :param output_timestep: parameter used to determine how often do to an output in the simulation
        :param update_positions: bool parameters used to indicate whether or not to update old_positions at collisions
        and update the distance_to_collision array in box_of_particles to compute mean free path.
        :param save_positions: boolean variable used to indicate if one want to save positions. Since saving takes some
        capacity and it is not so interesting when doing multiple runs one have the option to not save positions.
        :return time_array, mean_quadratic_distance_array, mean_quadratic_speed_array
        """
        print('Simulate until a given simulation time is reached')
        print(f'N: {self.box_of_particles.N} and xi: {self.box_of_particles.restitution_coefficient}..')
        print('---------------------')
        # create folder in order to save particle positions as a png files throughout the simulation
        simulation_folder = ""
        if save_positions:
            simulation_folder = self.create_simulation_folder(simulation_label)

        print('Creating initial queue..')

        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output
        # Initialize the queue with all starting collisions
        self.box_of_particles.create_initial_priority_queue(t_max=self.stopping_criterion)

        # initial energy for all particles, m0 particles and m particles
        avg_energy = self.box_of_particles.compute_energy()

        if self.mask is None:
            # pick all particles inside a circle from center with radius 0.1 by turning mask into boolean array
            distance_to_middle_position = norm((self.box_of_particles.positions -
                                                np.tile([0.5, 0.5], reps=(len(self.box_of_particles.positions), 1))),
                                               axis=1)
            self.mask = distance_to_middle_position < 0.2

        # give initial output and save particle positions
        self.print_output(self.average_number_of_collisions, avg_energy)
        if save_positions:
            self.save_particle_positions(simulation_folder, output_number)
        next_output_time += output_timestep
        output_number += 1

        time_array = np.zeros(int(self.stopping_criterion / output_timestep) + 1)  # array for time
        mean_quadratic_distance_array = np.zeros_like(time_array)  # array for msd / mqd
        mean_quadratic_speed_array = np.zeros_like(time_array)
        mean_quadratic_speed_array[0] = np.mean(norm(self.box_of_particles.velocities[self.mask], axis=1)**2)

        print('Event driven simulation in progress..')
        # run until the simulation time has reached the stopping criterion
        while self.simulation_time < self.stopping_criterion:
            collision_tuple = heapq.heappop(self.box_of_particles.collision_queue)  # pop the earliest element
            time_at_collision = collision_tuple[0]  # extract time at collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.box_of_particles.positions += self.box_of_particles.velocities * dt
                self.simulation_time += dt

                time_array[output_number] = self.simulation_time

                # compute mean quadratic distance from starting position for masked particles
                mean_quadratic_distance_array[output_number] = \
                    np.mean(
                        norm(self.box_of_particles.positions[self.mask]-self.box_of_particles.old_positions[self.mask],
                             axis=1)**2)
                # compute mean_quadratic_speed for masked particles
                mean_quadratic_speed_array[output_number] = \
                    np.mean(norm(self.box_of_particles.velocities[self.mask], axis=1)**2)

                # average energy for all particles, m0 particles and m particles
                avg_energy = self.box_of_particles.compute_energy()

                # give output and save particle positions
                self.print_output(self.average_number_of_collisions, avg_energy)
                if save_positions:
                    self.save_particle_positions(simulation_folder, output_number)

                next_output_time += output_timestep
                output_number += 1

            if self.box_of_particles.valid_collision(collision_tuple):
                self.perform_collision(time_at_collision, collision_tuple, t_max=self.stopping_criterion)
                if update_positions:
                    object_one = collision_tuple[1][0]
                    object_two = collision_tuple[1][1]
                    distance_moved_object = norm(
                        self.box_of_particles.positions[object_one, :] - self.box_of_particles.old_positions[
                                                                         object_one, :])
                    self.box_of_particles.distance_to_collisions[object_one] += distance_moved_object
                    self.box_of_particles.old_positions[object_one, :] = self.box_of_particles.positions[object_one, :]
                    if object_two not in ["hw", "vw"]:
                        distance_moved_object = norm(
                            self.box_of_particles.positions[object_two, :] - self.box_of_particles.old_positions[
                                                                             object_two, :])
                        self.box_of_particles.distance_to_collisions[object_two] += distance_moved_object

                        self.box_of_particles.old_positions[object_two, :] =\
                            self.box_of_particles.positions[object_two, :]

        print('Simulation done!')
        print('---------------------')
        return time_array, mean_quadratic_distance_array, mean_quadratic_speed_array

    def simulate_until_given_time_bp(self, simulation_label, output_timestep=1.0, save_positions=True):
        """
            Implementation of the event driven simulation where the stopping criterion is given as a limit of
            how much one want the simulation time to be. Is useful when looking at a property as a function of time.
            This function stores the position and velocity of the Brownian particle with output_timestep resolution
            until t_stop.
        :param simulation_label: string containing information about the simulation to identify simulation
        :param output_timestep: parameter used to determine how often do to an output in the simulation
        and update the distance_to_collision array in box_of_particles to compute mean free path.
        :param save_positions: boolean variable used to indicate if one want to save positions. Since saving takes some
        capacity and it is not so interesting when doing multiple runs one have the option to not save positions.
        :return time_array, bp_position_array, bp_velocity_array
        """
        print('Simulate until a given simulation time is reached')
        print(f'N: {self.box_of_particles.N} and xi: {self.box_of_particles.restitution_coefficient}..')
        print('---------------------')
        # create folder in order to save particle positions as a png files throughout the simulation
        simulation_folder = ""
        if save_positions:
            simulation_folder = self.create_simulation_folder(simulation_label)

        print('Creating initial queue..')

        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output
        # Initialize the queue with all starting collisions
        self.box_of_particles.create_initial_priority_queue(t_max=self.stopping_criterion)

        # initial energy for all particles, m0 particles and m particles
        avg_energy = self.box_of_particles.compute_energy()

        # give initial output and save particle positions
        self.print_output(self.average_number_of_collisions, avg_energy)
        if save_positions:
            self.save_particle_positions(simulation_folder, output_number)
        next_output_time += output_timestep
        output_number += 1

        time_array = np.zeros(int(self.stopping_criterion / output_timestep) + 1)  # array for time
        bp_position_array = np.zeros((len(time_array), 2))
        bp_velocity_array = np.zeros((len(time_array), 2))
        bp_position_array[0, :] = self.box_of_particles.positions[self.mask]
        bp_velocity_array[0, :] = self.box_of_particles.velocities[self.mask]

        print('Event driven simulation in progress..')
        # run until the simulation time has reached the stopping criterion
        while self.simulation_time < self.stopping_criterion:
            collision_tuple = heapq.heappop(self.box_of_particles.collision_queue)  # pop the earliest element
            time_at_collision = collision_tuple[0]  # extract time at collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.box_of_particles.positions += self.box_of_particles.velocities * dt
                self.simulation_time += dt

                time_array[output_number] = self.simulation_time

                bp_position_array[output_number, :] = self.box_of_particles.positions[self.mask]
                bp_velocity_array[output_number, :] = self.box_of_particles.velocities[self.mask]

                # average energy for all particles, m0 particles and m particles
                avg_energy = self.box_of_particles.compute_energy()

                # give output and save particle positions
                self.print_output(self.average_number_of_collisions, avg_energy)
                if save_positions:
                    self.save_particle_positions(simulation_folder, output_number)

                next_output_time += output_timestep
                output_number += 1

            if self.box_of_particles.valid_collision(collision_tuple):
                self.perform_collision(time_at_collision, collision_tuple, t_max=self.stopping_criterion)

        print('Simulation done!')
        print('---------------------')
        return time_array, bp_position_array, bp_velocity_array
