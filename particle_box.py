import numpy as np
import heapq

from scipy.linalg import norm


class ParticleBox:
    """
        Class which implements a cubic box containing a number of particles with a given radius and mass. The class
        has functionality to implement collisions with vertical, horizontal and top/bottom walls and between two
        particles. ParticleBox is a help class used by the class Simulation, which conducts a event driven simulation
        of force-free granular gases. ParticleBox can handle periodic boundary conditions (pbc) and hard walls in 3D.
    """

    def __init__(self, number_of_particles, restitution_coefficient, initial_positions, initial_velocities, masses,
                 radii, pbc):
        """
            Initialize the ParticleBox class.
        :param number_of_particles: number of particles in the box.
        :param restitution_coefficient: number giving how much energy is lost during a collision.
        :param initial_positions: Array with shape = (number_of_particles, 3) giving x, y and z coordinates.
        :param initial_velocities: Array with shape = (number_of_particles, 3) giving velocity along x-, y- and z-axis
        :param masses: Array with length number_of_particles giving the masses of each particle.
        :param radii: Array with length number_of_particles giving the radius of each particle.
        :param pbc: bool value used to indicate whether or not to use pbc.
        """
        self.N = number_of_particles  # amount of particles
        self.restitution_coefficient = restitution_coefficient  # coefficient determining the energy lost in collisions
        # initialize variables used in the class
        self.positions = np.zeros((self.N, 3))  # positions of particles
        self.initial_positions = np.zeros((self.N, 3))  # help variable to compute mean square displacement
        self.velocities = np.zeros((self.N, 3))  # velocities of particles
        self.masses = np.zeros(self.N)  # mass of each particle
        self.radii = np.zeros(self.N)  # radius of each particle
        self.collision_count_particles = np.zeros(self.N)  # array keeping track of the number of collisions

        # set parameters equal to the input to the class. Use .copy() such that the parameters can be used in outer loop
        self.positions = initial_positions.copy()
        self.initial_positions = initial_positions.copy()
        self.velocities = initial_velocities.copy()
        self.masses = masses
        self.radii = radii
        # a priority queue / heap queue of tuples of (time_collision, collision_entities, collision_count when
        # computing the collision, box number of the particles). The collision count at computation is used to
        # ignore non-valid collisions due to the involved particles being in other collisions between computation and
        # collision. Box number is needed for the pbc.
        self.collision_queue = []  # heap queue needs list structure to work

        # In order to create 27 copies for pbc in three dimensions one need to known their relation to the original
        # box. These are given by offsets. Offsets is also used to correct positions of particles colliding in
        # different boxes (due to the pbc).
        self.offsets = [(-1, 1, 1), (0, 1, 1), (1, 1, 1), (-1, 0, 1), (0, 0, 1), (1, 0, 1), (-1, -1, 1), (0, -1, 1),
                        (1, -1, 1), (-1, 1, 0), (0, 1, 0), (1, 1, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (-1, -1, 0),
                        (0, -1, 0), (1, -1, 0), (-1, 1, -1), (0, 1, -1), (1, 1, -1), (-1, 0, -1), (0, 0, -1),
                        (1, 0, -1), (-1, -1, -1), (0, -1, -1), (1, -1, -1)]
        # Crossings is used to compute current positions due to the periodic boundary conditions. It essentially get
        # updated every time a particle cross the edge in the x-, y- or z-direction.
        self.crossings = np.zeros((self.N, 3))

        self.pbc = pbc  # periodic boundary conditions

    def collision_horizontal_wall(self, particle_number, restitution_coefficient):
        """
            Function to solve a collision with a particle and a horizontal wall by updating the velocity vector if
            hard walls. If pbc: move particle to other edge along correct axis and update crossings.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param restitution_coefficient: degree of inelasticity in the system, float value in [0,1].
        """
        # check for pbc
        if self.pbc:
            # for pbc: move particle to the other edge along the correct axis and update crossings
            if self.positions[particle_number, 1] > 0.5:
                self.positions[particle_number, 1] -= 1
                self.crossings[particle_number, 1] += 1
            else:
                self.positions[particle_number, 1] += 1
                self.crossings[particle_number, 1] -= 1
        else:
            # for hard walls, the particle collides with a wall (particle with inf mass)
            self.velocities[particle_number, :] *= restitution_coefficient * np.array([1, -1, 1])

    def collision_vertical_wall(self, particle_number, restitution_coefficient):
        """
            Function to solve a collision with a particle and a vertical wall by updating the velocity vector if
            hard walls. If pbc: move particle to other edge along correct axis and update crossings.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param restitution_coefficient: degree of inelasticity in the system, float value in [0,1].
        """
        # check for pbc
        if self.pbc:
            # for pbc: move particle to the other edge along the correct axis and update crossings
            if self.positions[particle_number, 0] > 0.5:
                self.positions[particle_number, 0] -= 1
                self.crossings[particle_number, 0] += 1
            else:
                self.positions[particle_number, 0] += 1
                self.crossings[particle_number, 0] -= 1
        else:
            # for hard walls, the particle collides with a wall (particle with inf mass)
            self.velocities[particle_number, :] *= restitution_coefficient * np.array([-1, 1, 1])

    def collision_tb_wall(self, particle_number, restitution_coefficient):
        """
            Function to solve a collision with a particle and a top/bottom wall by updating the velocity vector if
            hard walls. If pbc: move particle to other edge along correct axis and update crossings.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param restitution_coefficient: degree of inelasticity in the system, float value in [0,1].
        """
        # check for pbc
        if self.pbc:
            # for pbc: move particle to the other edge along the correct axis and update crossings
            if self.positions[particle_number, 2] > 0.5:
                self.positions[particle_number, 2] -= 1
                self.crossings[particle_number, 2] += 1
            else:
                self.positions[particle_number, 2] += 1
                self.crossings[particle_number, 2] -= 1
        else:
            # for hard walls, the particle collides with a wall (particle with inf mass)
            self.velocities[particle_number, :] *= restitution_coefficient * np.array([1, 1, -1])

    def collision_particles(self, particle_one, particle_two, restitution_coefficient, box_particle_two):
        """
            Function to solve a collision between two particles by updating the velocity vector for both particles.
        :param particle_one: the index of particle number one.
        :param particle_two: the index of particle number two.
        :param restitution_coefficient: degree of inelasticity in the system, float value in [0,1].
        :param box_particle_two: the box number of the second particle. Used to handle pbc.
        """
        pos_particle_two = self.positions[particle_two, :].copy()  # get position of second particle
        # check if the box of the second particle is not 13, which is system box (offset=(0, 0, 0)).
        if box_particle_two != 13:
            # collision is through a wall and one must use offset to correct collision position to get correct dx
            # meaning that particle_two is in one of the copied systems due to pbc
            pos_particle_two += [self.offsets[box_particle_two][0], self.offsets[box_particle_two][1],
                                 self.offsets[box_particle_two][2]]

        mass_particle_one, mass_particle_two = self.masses[particle_one], self.masses[particle_two]  # get masses
        delta_x = pos_particle_two - self.positions[particle_one, :]  # difference in position
        delta_v = self.velocities[particle_two, :] - self.velocities[particle_one, :]  # difference in velocity
        r_squared = (self.radii[particle_one] + self.radii[particle_two]) ** 2  # distance from center to center

        # update velocities of the particles
        self.velocities[particle_one, :] += delta_x*((1+restitution_coefficient)*mass_particle_two*np.dot(delta_v, delta_x)/((mass_particle_one+mass_particle_two)*r_squared))
        self.velocities[particle_two, :] -= delta_x*((1+restitution_coefficient)*mass_particle_one*np.dot(delta_v, delta_x)/((mass_particle_one+mass_particle_two)*r_squared))

    def time_at_collision_vertical_wall(self, particle_number, simulation_time):
        """
            Function that computes at what time a particle will collide with a vertical wall
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation.
        :return: the time when particle particle_number will collide with a vertical wall.
        """
        velocity_x = self.velocities[particle_number, 0]  # velocity in the x-direction for the particle
        position_x = self.positions[particle_number, 0]  # x-position of the particle
        # check for pbc
        if self.pbc:
            # if pbc: set radius equal to 0 in order to get the time when the center touches the wall
            radius = 0
        else:
            radius = self.radii[particle_number]  # radius of the particle
        # compute time until collision based on the sign of the velocity
        if velocity_x > 0:
            time_until_collision = (1-radius-position_x) / velocity_x
        elif velocity_x < 0:
            time_until_collision = (radius-position_x) / velocity_x
        else:
            time_until_collision = np.inf
        return time_until_collision + simulation_time

    def time_at_collision_horizontal_wall(self, particle_number, simulation_time):
        """
            Function that computes at what time a particle will collide with a horizontal wall.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation.
        :return: the time when particle particle_number will collide with a horizontal wall.
        """
        velocity_y = self.velocities[particle_number, 1]  # velocity in the y-direction of the particle
        position_y = self.positions[particle_number, 1]  # y position of the particle
        # check for pbc
        if self.pbc:
            # if pbc: set radius equal to 0 in order to get the time when the center touches the wall
            radius = 0
        else:
            radius = self.radii[particle_number]  # radius of the particle
        # compute time until collision based on the sign of the velocity
        if velocity_y > 0:
            time_until_collision = (1 - radius - position_y) / velocity_y
        elif velocity_y < 0:
            time_until_collision = (radius - position_y) / velocity_y
        else:
            time_until_collision = np.inf
        return time_until_collision + simulation_time

    def time_at_collision_tb_wall(self, particle_number, simulation_time):
        """
            Function that computes at what time a particle will collide with a top/bottom wall.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation.
        :return: the time when particle particle_number will collide with a horizontal wall.
        """
        velocity_z = self.velocities[particle_number, 2]  # velocity in the z-direction of the particle
        position_z = self.positions[particle_number, 2]  # z position of the particle
        # check for pbc
        if self.pbc:
            # if pbc: set radius equal to 0 in order to get the time when the center touches the wall
            radius = 0
        else:
            radius = self.radii[particle_number]  # radius of the particle
        # compute time until collision based on the sign of the velocity
        if velocity_z > 0:
            time_until_collision = (1 - radius - position_z) / velocity_z
        elif velocity_z < 0:
            time_until_collision = (radius - position_z) / velocity_z
        else:
            time_until_collision = np.inf
        return time_until_collision + simulation_time

    def time_at_collision_particles(self, particle_number, simulation_time):
        """
            Function that computes the time until a particle collides with all other particles.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation.
        :return: the time when particle particle_number will collide with all of the other particles.
        """
        # difference from particle particle_number to all other particles
        delta_x = self.positions - np.tile(self.positions[particle_number, :], reps=(len(self.positions), 1))
        # difference in velocity from particle particle_number to all other particles
        delta_v = self.velocities - np.tile(self.velocities[particle_number, :], reps=(len(self.velocities), 1))
        r_squared = (self.radii[particle_number] + self.radii) ** 2  # array of center to center distances
        dvdx = np.sum(delta_v*delta_x, axis=1)  # dot product between delta_v and delta_x
        dvdv = np.sum(delta_v*delta_v, axis=1)  # dot product between delta_v and delta_v
        d = dvdx ** 2 - dvdv * (norm(delta_x, axis=1) ** 2 - r_squared)  # help array quantity
        time_until_collisions = np.ones(self.N)*np.inf  # assume no particles is going to collide
        boolean = np.logical_and(dvdx < 0, d > 0)  # both these conditions must be valid particle-particle collision
        # check if there exist some valid particle-particle collisions for particle particle_number
        if np.sum(boolean) > 0:
            # compute time until collision
            time_until_collisions[boolean] = -1 * ((dvdx[boolean] + np.sqrt(d[boolean])) / (dvdv[boolean]))
        return time_until_collisions + simulation_time

    def time_at_collision_particles_pbc(self, particle_number, simulation_time):
        """
            Similar function as time_at_collision_particles, but is modified for pbc. The function computes the time
            until particle particle_number collides with the other particles and their 27 copies (pbc in 3D).
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation.
        :return: the time when particle particle_number will collide with all of the other particles.
        """
        positions = np.zeros((len(self.positions)*27, 3))  # need 27 boxes/copies of the system
        # set correct positions of the particles in all boxes with all 27 offsets
        for i, offset in enumerate(self.offsets):
            # position of the particles in box i is given as 'positions + offset[i]'
            positions[i*len(self.positions):(i+1)*len(self.positions)] = \
                self.positions + np.array([offset[0], offset[1], offset[2]])
        # difference from particle particle_number to all other particles
        delta_x = positions - np.tile(self.positions[particle_number, :], reps=(len(positions), 1))
        # difference in velocity from particle particle_number to all other particles
        delta_v = self.velocities - np.tile(self.velocities[particle_number, :], reps=(len(self.velocities), 1))
        delta_v = np.tile(delta_v, reps=(27, 1))  # all copies have the same velocity as the original particles
        r_squared = (self.radii[particle_number] + self.radii) ** 2  # array of center to center distance
        r_squared = np.tile(r_squared, reps=(27, ))  # r_squares is the same for all copies
        dvdx = np.sum(delta_v * delta_x, axis=1)  # dot product between delta_v and delta_x
        dvdv = np.sum(delta_v * delta_v, axis=1)  # dot product between delta_v and delta_v
        d = dvdx ** 2 - dvdv * (norm(delta_x, axis=1) ** 2 - r_squared)  # help array quantity
        time_until_collisions = np.ones(self.N*27) * np.inf  # assume no particles is going to collide
        boolean = np.logical_and(dvdx < 0, d > 0)  # both these conditions must be valid for particle-particle collision
        # check if there exist some valid particle-particle collisions for particle particle_number
        if np.sum(boolean) > 0:
            # compute time until collision
            time_until_collisions[boolean] = -1 * ((dvdx[boolean] + np.sqrt(d[boolean])) / (dvdv[boolean]))
        return time_until_collisions + simulation_time

    def add_collision_horizontal_wall_to_queue(self, particle_number, simulation_time):
        """
            Help function to compute time at collision with horizontal wall for a given particle, create collision
            tuple and push the tuple into the heap queue.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation.
        """
        time_hw = self.time_at_collision_horizontal_wall(particle_number, simulation_time)  # time at collision
        # create collision tuple on desired form
        tuple_hw = (time_hw, [particle_number, 'hw'], [self.collision_count_particles[particle_number]], [13])
        # push to heap queue
        heapq.heappush(self.collision_queue, tuple_hw)

    def add_collision_vertical_wall_to_queue(self, particle_number, simulation_time):
        """
            Help function to compute time at collision with vertical wall for a given particle, create collision
            tuple and push the tuple into the heap queue.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation.
        """
        time_vw = self.time_at_collision_vertical_wall(particle_number, simulation_time)  # time at collision
        # create collision tuple on desired form
        tuple_vw = (time_vw, [particle_number, 'vw'], [self.collision_count_particles[particle_number]], [13])
        # push to heap queue
        heapq.heappush(self.collision_queue, tuple_vw)

    def add_collision_tb_wall_to_queue(self, particle_number, simulation_time):
        """
            Help function to compute time at collision with top/bottom wall for a given particle, create collision
            tuple and push the tuple into the heap queue.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation.
        """
        time_vw = self.time_at_collision_tb_wall(particle_number, simulation_time)  # time at collision
        # create collision tuple on desired form
        tuple_vw = (time_vw, [particle_number, 'tbw'], [self.collision_count_particles[particle_number]], [13])
        # push to heap queue
        heapq.heappush(self.collision_queue, tuple_vw)

    def add_collisions_particle_to_queue(self, particle_number, simulation_time, t_max):
        """
            Help function to compute time at collision with all particles for a given particle, create collision
            tuples and push valid tuples into the heap queue.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation.
        :param t_max: is a float containing the stopping criterion of the simulation in time. Is used to neglect
        collisions if they occur later than t_max*1.01. Default to None: use all. Exist if simulation until t_stop.
        """
        # check for pbc
        if self.pbc:
            # get time of collisions between particle particle_number and all other particles, for all 27 boxes
            time_at_collisions = self.time_at_collision_particles_pbc(particle_number, simulation_time)
            # list of possible collision particles. Repeated for each copy of the system
            collision_particles = np.tile(np.arange(self.N), reps=(27, ))
            # the box number of each of the particles
            boxes = np.repeat(np.arange(0, 27), self.N)
        else:
            # get time of collisions between particle particle_number and all other particles
            time_at_collisions = self.time_at_collision_particles(particle_number, simulation_time)
            collision_particles = np.arange(self.N)  # create a list of possible collision candidates
            # get box number of each of the particles. With hard walls all particles are in box 13
            boxes = np.repeat(13, self.N)
        # only regard valid collisions by removing all entries which are np.inf
        if t_max is None:
            boolean = time_at_collisions != np.inf
        else:
            # also neglect collisions occurring after 1.01*t_max
            boolean = np.logical_and(time_at_collisions != np.inf, time_at_collisions < t_max*1.01)

        # pick out the correct particles, times and boxes with the use of the boolean array
        collision_particles = collision_particles[boolean]
        time_at_collisions = time_at_collisions[boolean]
        boxes = boxes[boolean]
        # check if there are any valid collisions
        if len(time_at_collisions) > 0:
            # iterate through all valid collisions
            for i in range(len(time_at_collisions)):
                # create collision tuple of valid form
                tuple_particle_collision = (time_at_collisions[i], [particle_number, collision_particles[i]],
                                            [self.collision_count_particles[particle_number],
                                             self.collision_count_particles[collision_particles[i]]],
                                            [13, boxes[i]])
                # push tuple to heap queue
                heapq.heappush(self.collision_queue, tuple_particle_collision)

    def create_initial_priority_queue(self, t_max=None):
        """
            Help function that initialize the heap queue by iterating though all particles and push all possible
            collisions to the heap queue.
        :param t_max: is a float containing the stopping criterion of the simulation in time. Is used to neglect
        collisions if they occur later than t_max*1.01. Default to None: use all. Exist if simulation until t_stop.
        """
        for particle_number in range(self.N):  # iterate through each particle
            self.add_collision_horizontal_wall_to_queue(particle_number, 0)  # add collision with horizontal wall
            self.add_collision_vertical_wall_to_queue(particle_number, 0)  # add collision with vertical wall
            self.add_collision_tb_wall_to_queue(particle_number, 0)  # add collision with tb wall
            self.add_collisions_particle_to_queue(particle_number, 0, t_max)  # add collisions with other particles

    def valid_collision(self, collision_tuple):
        """
            Function that validates a proposed new collision by looking at the collision tuple information.
        :param collision_tuple: tuple with information: (coll_time, coll entities, coll_count_comp_coll, box_comp_coll).
        """
        # check if first entity has not been in collision since computation of collision
        if collision_tuple[2][0] == self.collision_count_particles[collision_tuple[1][0]]:
            # if there is a second particle and it has been in another collision -> False
            if len(collision_tuple[2]) == 2 and \
                    collision_tuple[2][1] != self.collision_count_particles[collision_tuple[1][1]]:
                return False
            # Accept if there is only one particle, or the second particle has not been in another collision
            else:
                return True
        else:
            return False

    def update_queue_new_collisions_particle(self, particle_number, simulation_time, t_max):
        """
            Help function that add all new possible collisions for a particle after being part of a collision.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data.
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation.
        :param t_max: is a float containing the stopping criterion of the simulation in time. Is used to neglect.
        collisions if they occur later than t_max*1.01. Default to None: use all. Exist if simulation until t_stop.
        """
        self.add_collision_horizontal_wall_to_queue(particle_number, simulation_time)
        self.add_collision_vertical_wall_to_queue(particle_number, simulation_time)
        self.add_collision_tb_wall_to_queue(particle_number, simulation_time)
        self.add_collisions_particle_to_queue(particle_number, simulation_time, t_max)

    def compute_energy(self):
        """
            Help function to compute the average particle energy in the system.
        :return: average energy of all particles
        """
        energy = 0.5 * self.masses * np.sum(self.velocities * self.velocities, axis=1)
        avg_energy = np.mean(energy)  # average kinetic energy of all particles
        return avg_energy

    def compute_speeds(self):
        """
            Help function to compute the speed of the particles in the system.
        :return: the speed of all N particles as a array
        """
        # compute and return the speed as the norm of all velocity vectors
        return norm(self.velocities, axis=1)

    def compute_mean_square_speed(self):
        """
            Help function to compute the mean square speed of the particles in the system
        :return: mean square speed as a float
        """
        speeds = self.compute_speeds()  # speed of all particles
        return np.mean(speeds**2)  # mean square speed

    def compute_mean_square_displacement(self):
        """
            Help function to compute the mean square displacement of the particles in the system
        :return: mean square displacement as a float
        """
        # compute the current_positions by adding the number of crossings of the system
        current_positions = self.positions + self.crossings
        # get the dx vector between the current position and the initial positions for all particles
        dx = current_positions - self.initial_positions
        # compute and return the mean square displacement
        return np.mean(norm(dx, axis=1)**2)
