import numpy as np
import heapq

from scipy.linalg import norm


class ParticleBox:
    """
        Class which implements a square box containing a number of particles with a given radius and mass. The class
        has functionality to implement collisions with vertical and horizontal walls and between two particles. The main
        function is simulate which is a implementation of a event driven simulation of particles colliding in a box.
    """

    def __init__(self, number_of_particles, restitution_coefficient, initial_positions, initial_velocities, masses,
                 radii):
        """
            Initialize the ParticleBox class
        :param number_of_particles: number of particles in the box
        :param restitution_coefficient: number giving how much energy is lost during a collision.
        :param initial_positions: Array with shape = (number_of_particles, 2) giving x and y coordinates
        :param initial_velocities: Array with shape = (number_of_particles, 2) giving x and y velocities
        :param masses: Array with length number_of_particles giving the masses of each particle
        :param radii: Array with length number_of_particles giving the radius of each particle
        """
        self.N = number_of_particles  # amount of particles
        self.restitution_coefficient = restitution_coefficient  # coefficient determining the energy lost in collisions
        # initialize variables used in the class
        self.positions = np.zeros((self.N, 2))  # positions of particles
        self.old_positions = np.zeros((self.N, 2))  # help variable to calculate distance from start and mean free path
        self.velocities = np.zeros((self.N, 2))  # velocities of particles
        self.masses = np.zeros(self.N)  # mass of each particle
        self.radii = np.zeros(self.N)  # radius of each particles
        self.collision_count_particles = np.zeros(self.N)  # array keeping track of the number of collisions
        self.distance_to_collisions = np.zeros(self.N)  # array keeping track of the distance travelled for collisions

        # set parameters equal to the input to the class. Use .copy() such that the parameters can be used in outer loop
        self.positions = initial_positions.copy()
        self.old_positions = initial_positions.copy()
        self.velocities = initial_velocities.copy()
        self.masses = masses
        self.radii = radii
        # a priority queue / heap queue of tuples of (time_collision, collision_entities, collision_count when
        # computing the collision). The collision count at computation is used to ignore non-valid collisions due to
        # the involved particles being in other collisions between computation and collision.
        self.collision_queue = []  # heap queue needs list structure to work

    def collision_horizontal_wall(self, particle_number, restitution_coefficient):
        """
            Function to solve a collision with a particle and a horizontal wall by updating the velocity vector
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param restitution_coefficient: number giving how much energy is lost during a collision.
        """
        self.velocities[particle_number, :] *= restitution_coefficient*np.array([1, -1])

    def collision_vertical_wall(self, particle_number, restitution_coefficient):
        """
            Function to solve a collision with a particle and a vertical wall by updating the velocity vector
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param restitution_coefficient: number giving how much energy is lost during a collision.
        """
        self.velocities[particle_number, :] *= restitution_coefficient * np.array([-1, 1])

    def collision_particles(self, particle_one, particle_two, restitution_coefficient):
        """
            Function to solve a collision between two particles by updating the velocity vector for both particles
        :param particle_one: the index of particle number one
        :param particle_two: the index of particle number two
        :param restitution_coefficient: number giving how much energy is lost during a collision.
        """
        mass_particle_one, mass_particle_two = self.masses[particle_one], self.masses[particle_two]  # get masses
        delta_x = self.positions[particle_two, :] - self.positions[particle_one, :]  # difference in position
        delta_v = self.velocities[particle_two, :] - self.velocities[particle_one, :]  # difference in velocity
        r_squared = (self.radii[particle_one] + self.radii[particle_two]) ** 2  # distance from center to center
        # update velocities of the particles
        self.velocities[particle_one, :] += delta_x*((1+restitution_coefficient)*mass_particle_two*np.dot(delta_v, delta_x)/((mass_particle_one+mass_particle_two)*r_squared))
        self.velocities[particle_two, :] -= delta_x*((1+restitution_coefficient)*mass_particle_one*np.dot(delta_v, delta_x)/((mass_particle_one+mass_particle_two)*r_squared))

    def time_at_collision_vertical_wall(self, particle_number, simulation_time):
        """
            Function that computes at what time a particle will collide with a vertical wall
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation
        :return: the time when particle particle_number will collide with a vertical wall
        """
        velocity_x = self.velocities[particle_number, 0]  # velocity in the x-direction for the particle
        position_x = self.positions[particle_number, 0]  # x-position of the particle
        radius = self.radii[particle_number]  # radius of the particle
        # compute time until collision
        if velocity_x > 0:
            time_until_collision = (1-radius-position_x) / velocity_x
        elif velocity_x < 0:
            time_until_collision = (radius-position_x) / velocity_x
        else:
            time_until_collision = np.inf
        return time_until_collision + simulation_time

    def time_at_collision_horizontal_wall(self, particle_number, simulation_time):
        """
            Function that computes at what time a particle will collide with a horizontal wall
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation
        :return: the time when particle particle_number will collide with a horizontal wall
        """
        velocity_y = self.velocities[particle_number, 1]  # velocity in the y-direction of the particle
        position_y = self.positions[particle_number, 1]  # y position of the particle
        radius = self.radii[particle_number]  # radius of the particle
        # compute time until collision
        if velocity_y > 0:
            time_until_collision = (1 - radius - position_y) / velocity_y
        elif velocity_y < 0:
            time_until_collision = (radius - position_y) / velocity_y
        else:
            time_until_collision = np.inf
        return time_until_collision + simulation_time

    def time_at_collision_particles(self, particle_number, simulation_time):
        """
            Function that computes the time until a particle collides with all other particles
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation
        :return: the time when particle particle_number will collide with all of the other particles
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

    def add_collision_horizontal_wall_to_queue(self, particle_number, simulation_time):
        """
            Help function to compute time at collision with horizontal wall for a given particle, create collision
            tuple and push the tuple into the heap queue.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation
        """
        time_hw = self.time_at_collision_horizontal_wall(particle_number, simulation_time)  # time at collision
        # create collision tuple on desired form
        tuple_hw = (time_hw, [particle_number, 'hw'], [self.collision_count_particles[particle_number]])
        # push to heap queue
        heapq.heappush(self.collision_queue, tuple_hw)

    def add_collision_vertical_wall_to_queue(self, particle_number, simulation_time):
        """
            Help function to compute time at collision with vertical wall for a given particle, create collision
            tuple and push the tuple into the heap queue.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation
        """
        time_vw = self.time_at_collision_vertical_wall(particle_number, simulation_time)  # time at collision
        # create collision tuple on desired form
        tuple_vw = (time_vw, [particle_number, 'vw'], [self.collision_count_particles[particle_number]])
        # push to heap queue
        heapq.heappush(self.collision_queue, tuple_vw)

    def add_collisions_particle_to_queue(self, particle_number, simulation_time, t_max):
        """
            Help function to compute time at collision with all particles for a given particle, create collision
            tuples and push valid tuples into the heap queue.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation
        :param t_max: is a float containing the stopping criterion of the simulation in time. Is used to neglect
        collisions if they occur later than t_max*1.01. Default to None: use all. Exist if simulation until t_stop.
        """
        time_at_collisions = self.time_at_collision_particles(particle_number, simulation_time)  # get time collisions
        collision_particles = np.arange(self.N)  # create a list of possible collision candidates
        # only regard valid collisions by removing all entries which are np.inf
        if t_max is None:
            boolean = time_at_collisions != np.inf
            collision_particles = collision_particles[boolean]
            time_at_collisions = time_at_collisions[boolean]
        else:
            boolean = np.logical_and(time_at_collisions != np.inf, time_at_collisions < t_max*1.01)
            collision_particles = collision_particles[boolean]
            time_at_collisions = time_at_collisions[boolean]

        # check if there are any valid collisions
        if len(time_at_collisions) > 0:
            # iterate through all valid collisions
            for i in range(len(time_at_collisions)):
                # create collision tuple of valid form
                tuple_particle_collision = (time_at_collisions[i], [particle_number, collision_particles[i]],
                                            [self.collision_count_particles[particle_number],
                                             self.collision_count_particles[collision_particles[i]]])
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
            self.add_collisions_particle_to_queue(particle_number, 0, t_max)  # add collisions with other particles

    def valid_collision(self, collision_tuple):
        """
            Function that validates a proposed new collision by looking at the collision tuple information
        :param collision_tuple: tuple with information: (coll_time, coll entities, coll_count_comp_coll)
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
            Help function that add all new possible collisions for a particle after being part of a collision
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param simulation_time: is a float of the simulation time, used to get time for collisions in the simulation
        :param t_max: is a float containing the stopping criterion of the simulation in time. Is used to neglect
        collisions if they occur later than t_max*1.01. Default to None: use all. Exist if simulation until t_stop.
        """
        self.add_collision_horizontal_wall_to_queue(particle_number, simulation_time)
        self.add_collision_vertical_wall_to_queue(particle_number, simulation_time)
        self.add_collisions_particle_to_queue(particle_number, simulation_time, t_max)

    def compute_energy(self, equal_particles):
        """
            Function to compute the energy in the system of particles. Based on boolean input can compute average
            kinetic energy of all particles or compute average for all, for m0 and for m.
        :param equal_particles: bool value indicating if there exist similar type of particles. If false: the different
        halves of particles contain two different particle masses, m0 and m.
        :return: average energy of all particles or average energy of all particles, m0 particles and m particles
        """
        energy = 0.5 * self.masses * np.sum(self.velocities * self.velocities, axis=1)
        avg_energy = np.mean(energy)  # average kinetic energy of all particles
        if equal_particles:
            return avg_energy
        else:
            avg_energy_m0 = np.mean(energy[:int(self.N/2)])  # average kinetic energy of m0 particles
            avg_energy_m = np.mean(energy[int(self.N/2):])  # average kinetic energy of m(=4m0) particles
            return avg_energy, avg_energy_m0, avg_energy_m
