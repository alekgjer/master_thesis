import numpy as np

from particle_box import ParticleBox
from simulation import Simulation

N = 20
xi = 0.9
rad = 0.01

pos = np.ones((N, 2))*0.5
pos[:, 0] = np.linspace(0.05, 0.95, N)
#pos = np.array([[0.9*rad, 0.5], [0.8, 0.5]])

#vel = np.array([[0, 0], [0.2, 0]])
vel = np.zeros((N, 2))
vel[:, 0] = np.random.uniform(-0.2, 0.2, size=N)

mass = np.ones(N)
radii = np.ones(N)*rad

box_of_particles = ParticleBox(number_of_particles=N,
                               restitution_coefficient=xi,
                               initial_positions=pos,
                               initial_velocities=vel,
                               masses=mass,
                               radii=radii,
                               pbc=True)

simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=10)
simulation.simulate_until_given_number_of_collisions('test_pbc', output_timestep=0.1, save_positions=True)
