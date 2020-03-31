import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import utility_functions as util_funcs

from scipy.linalg import norm
from scipy.integrate import simps

from config import results_folder, init_folder

# Some functionality implemented to solve SDE numerically. The SDE of interest are the Langevin equation, which can be
# used to approximate the effects in a molecular gas and in a granular gas.


class SDESolver:
    """
        Class used to solve different SDEs by applying some numerical discretization methods. Is mainly used to solve
        the underdamped Langevin equation and underdamped scaled brownian motion by applying the Euler-Maruyama method.
         It does the Euler-Maruyama method for a given number of particles and return ensemble averages of the MSD.
    """
    def __init__(self, t_start, t_stop, dt, number_of_particles, constants):
        """
        Initialize a SDESolver object with the needed parameters of when to start, when to stop, timestep value,
        number of particles to use and constant achieved from theory.
        :param t_start: start time, which is always given as 0
        :param t_stop: stop time for the numerical iterative schemes
        :param dt: timestep value. Choose how often to update the positions and velocities of the particles
        :param number_of_particles: the number of particles to use for ensemble averages
        :param constants: the value used for [gamma0, d0, tau]. Is used to get correct diffusivity and friction.
        """
        self.N = number_of_particles
        self.t_0 = t_start
        self.t_stop = t_stop
        self.dt = dt
        self.number_of_timesteps = int((self.t_stop - self.t_0) / self.dt)
        self.N = number_of_particles
        self.constants = constants

    @staticmethod
    def diffusivity_underdamped_langevin_equation(times, params):
        """
            Help function to compute the diffusivity at all times for the underdamped Langevin equation, which off
            course is constant.
        :param times: array of different times to get the diffusivity
        :param params: [gamma0, d0, tau]
        return an array of d0 with the same shape as times
        """
        return np.ones_like(times)*params[1]

    @staticmethod
    def diffusivity_udsbm(times, params):
        """
            Help function to compute the diffusivity at all times for UDSBM, which decays as a function of time.
        :param times: array of different times to get the diffusivity
        :param params: [gamma0, d0, tau]
        return an array of the diffusivity at different times
        """
        return params[1]/(1+times/params[2])

    @staticmethod
    def friction_underdamped_langevin_equation(times, params):
        """
            Help function to compute the friction coefficient at all times for the underdamped Langevin equation,
            which off course is constant.
        :param times: array of different times to get the friction coefficient
        :param params: [gamma0, d0, tau]
        return an array of gamma0 with the same shape as times
        """
        return np.ones_like(times) * params[0]

    @staticmethod
    def friction_udsbm(times, params):
        """
            Help function to compute the friction coefficient at all times for UDSBM, which decays as a function
            of time.
        :param times: array of different times to get the friction coefficient
        :param params: [gamma0, d0, tau]
        return an array of the friction coefficient at different times
        """
        return params[0] / (1 + times / params[2])

    def euler_maruyama_method(self, friction_func, diffusivity_func):
        """
            Implementation of the Euler-Maruyama iterative scheme for a general SDE describing Brownian motion. It can
            be used for both the underdamped Langevin equation and UDSBM which differ due to the time dependence of the
            friction coefficient and the diffusivity. Other than that they are equal.
        :param friction_func: function giving the friction coefficient at all times. Is given as friction_coefficient_
        underdamped_langevin_equation or friction_udsbm for the two different SDEs.
        :param diffusivity_func: function giving the diffusivity at all times. Is given as diffusivity_underdamped_
        langevin_equation or diffusivity_udsbm for the two different SDEs.
        return times, positions_at_all_times, velocities_at_all_times
        """
        v0 = util_funcs.random_uniformly_distributed_velocities(self.N, np.sqrt(2), 3)  # initial velocities
        x0 = np.tile([0.5, 0.5, 0.5], reps=(self.N, 1))  # initial positions
        # create arrays for all positions and velocities at all timesteps
        positions = np.zeros((self.number_of_timesteps + 1, self.N, 3))
        velocities = np.zeros_like(positions)
        # initialize with initial values
        positions[0, :, :] = x0
        velocities[0, :, :] = v0

        times = np.arange(self.number_of_timesteps + 1) * self.dt

        gamma_t = friction_func(times, self.constants)
        d_t = diffusivity_func(times, self.constants)

        # compute the Euler-Maruyama iterative scheme for all particles simultaneously
        for i in range(self.number_of_timesteps):
            # the random force is a Wiener process, drawn from a normal distribution with mu = 0, sigma = sqrt(dt)
            dW = np.random.normal(loc=0, scale=np.sqrt(self.dt), size=(self.N, 3))
            positions[i + 1, :, :] = positions[i, :, :] + velocities[i, :, :] * self.dt
            velocities[i + 1, :, :] = velocities[i, :, :] - gamma_t[i] * velocities[i, :, :] * self.dt + \
                                      np.sqrt(2 * d_t[i]) * gamma_t[i] * dW

        return times, positions, velocities

    def strong_taylor_method(self, friction_func, diffusivity_func):
        """
            Implementation of the Strong Taylor iterative scheme for a general SDE describing Brownian motion. It can
            be used for both the underdamped Langevin equation and UDSBM which differ due to the time dependence of the
            friction coefficient and the diffusivity. Other than that they are equal.
        :param friction_func: function giving the friction coefficient at all times. Is given as friction_coefficient_
        underdamped_langevin_equation or friction_udsbm for the two different SDEs.
        :param diffusivity_func: function giving the diffusivity at all times. Is given as diffusivity_underdamped_
        langevin_equation or diffusivity_udsbm for the two different SDEs.
        return times, positions_at_all_times, velocities_at_all_times
        """
        v0 = util_funcs.random_uniformly_distributed_velocities(self.N, np.sqrt(2), 3)  # initial velocities
        x0 = np.tile([0.5, 0.5, 0.5], reps=(self.N, 1))  # initial positions
        # create arrays for all positions and velocities at all timesteps
        positions = np.zeros((self.number_of_timesteps + 1, self.N, 3))
        velocities = np.zeros_like(positions)
        # initialize with initial values
        positions[0, :, :] = x0
        velocities[0, :, :] = v0

        times = np.arange(self.number_of_timesteps + 1) * self.dt

        gamma_t = friction_func(times, self.constants)
        d_t = diffusivity_func(times, self.constants)

        # compute the Strong taylor iterative scheme for all particles simultaneously
        for i in range(self.number_of_timesteps):
            random_numb_1 = np.random.randn(self.N, 3)
            random_numb_2 = np.random.randn(self.N, 3)
            dW = random_numb_1*np.sqrt(self.dt)
            dZ = (random_numb_1+random_numb_2/np.sqrt(3))*self.dt**(3/2)/2
            positions[i + 1, :, :] = \
                positions[i, :, :] + velocities[i, :, :] * self.dt + velocities[i, :, :] * self.dt**2/2
            velocities[i + 1, :, :] = velocities[i, :, :] - gamma_t[i] * velocities[i, :, :] * self.dt + \
                                      np.sqrt(2 * d_t[i]) * gamma_t[i] * dW - np.sqrt(2 * d_t[i]) * gamma_t[i]**2 * dZ \
                                      + gamma_t[i]**2*velocities[i, :, :]*self.dt**2/2

        return times, positions, velocities

    def ensemble_msd(self, friction_func, diffusivity_func):
        """
            Help function to compute the ensemble mean square displacement and the ensemble mean square speed from the
            solution to a SDE for Brownian motion and save to file.
        :param friction_func: function giving the friction coefficient at all times. Is given as friction_coefficient_
        underdamped_langevin_equation or friction_udsbm for the two different SDEs.
        :param diffusivity_func: function giving the diffusivity at all times. Is given as diffusivity_underdamped_
        langevin_equation or diffusivity_udsbm for the two different SDEs.
        """
        times, positions, velocities = self.euler_maruyama_method(friction_func, diffusivity_func)
        dx = positions - positions[0, :, :]  # compute the dx vector from the initial position for all particles
        msd = np.mean(norm(dx, axis=2) ** 2, axis=1)  # compute the mean square displacement
        mss = np.mean(norm(velocities, axis=2) ** 2, axis=1)  # compute the mean square speed

        msd_matrix = np.zeros((len(times), 3))
        msd_matrix[:, 0] = times
        msd_matrix[:, 1] = msd
        msd_matrix[:, 2] = mss
        if self.constants[2] == np.inf:  # for a molecular gas tau is equal to inf
            np.save(os.path.join(results_folder, f"msd_sde_N_{self.N}_xi_1_tstop_{self.t_stop}_dt_{self.dt}"),
                    arr=msd_matrix)
        else:  # granular gas for which we have only considered xi = 0.8.
            np.save(os.path.join(results_folder, f"msd_sde_N_{self.N}_xi_0.8_tstop_{self.t_stop}_dt_{self.dt}"),
                    arr=msd_matrix)

    # TODO: fix the time average msd
    def A(self, times, delta, gamma0, T0):
        return -T0 * (np.exp(-gamma0 * delta) - 1 + np.exp(-gamma0 * times) - np.exp(-gamma0 * (times + delta))) / (
                    gamma0 ** 2)

    def time_average_msd(self, x, times, dt):
        delta_values = times
        number_of_timesteps = len(times) - 1
        msd = np.zeros_like(delta_values)
        dx = norm(x - x[0, :, :], axis=2)
        msd_values = np.mean(dx ** 2, axis=1)
        gamma0, T0 = 11.43, 2 / 3
        for counter, delta in enumerate(delta_values[1:-1], 1):
            print(counter, times[-1] - delta)
            # dx_t_delta = dx[counter:]
            # dx_t = dx[:-counter]
            # integrand = dx_t_delta - dx_t
            # integrand = np.mean(integrand**2, axis=1)
            # msd[counter] = simps(integrand, dx=dt)/(times[-1]-delta)
            msd_t_delta = msd_values[counter:]
            msd_t = msd_values[:-counter]
            a = self.A(times[:-counter], delta, gamma0, T0)
            integrand = msd_t_delta - msd_t - 2 * a
            msd[counter] = simps(integrand, dx=dt) / (times[-1] - delta)

        ensemble_msd = np.mean(dx ** 2, axis=1)
        plt.loglog(delta_values[1:-1], msd[1:-1], label='Time averaged MSD')
        plt.loglog(times, ensemble_msd, label='Ensemble MSD')
        plt.legend()
        plt.show()


def trajectory_ex(t_start=0, t_stop=1, dt=2**(-4), x0=np.array([1])):
    """
        Function to do the example from the section in Kloeden & Platen about the Euler-Maruyama method
    :param t_start: starting time
    :param t_stop: stopping time
    :param dt: timestep value
    :param x0: initial starting condition
    """
    parameters = [1.5, 1]
    number_of_timesteps = int((t_stop - t_start) / dt)
    x = np.zeros((number_of_timesteps + 1, len(x0)))
    x_exact = np.zeros_like(x)
    x[0, :] = x0
    x_exact[0, :] = x0
    time = np.linspace(t_start, t_stop, number_of_timesteps + 1)

    random_force = np.random.normal(scale=np.sqrt(dt), size=(number_of_timesteps, 1))

    for step in range(number_of_timesteps):
        x[step + 1, :] = x[step, :] + parameters[0]*x[step, :]*dt + parameters[1] * x[step, :] * random_force[step]
        x_exact[step+1, :] = x0*np.exp((parameters[0]-parameters[1]**2/2)*time[step]+parameters[1]*np.sum(random_force[:step+1]))

    plt.plot(time, x, label='Numerical solution')
    plt.plot(time, x_exact, label='Exact solution')
    plt.legend()
    plt.show()
