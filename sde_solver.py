import os
import numpy as np
import matplotlib.pyplot as plt

import utility_functions as util_funcs

from scipy.linalg import norm
from scipy.integrate import simps

from config import results_folder

# Some functionality implemented to solve SDE numerically. The SDE of interest are the Langevin equation, which can be
# used to approximate the effects in a molecular gas and in a granular gas.


class SDESolver:
    """
        Class used to solve different SDEs by applying some numerical discretization methods. Is mainly used to solve
        the underdamped Langevin equation and underdamped scaled brownian motion by applying the Euler-Maruyama method.
        It does the Euler-Maruyama method for a given number of particles and return ensemble mean square displacement
        and the mean square speed. It can also compute the time averaged msd. Can also perform the Strong Taylor
        method, but since the results from the Euler-Maruyama method is so good higher order methods are not needed.
    """
    def __init__(self, t_start, t_stop, dt, number_of_particles, constants, method='em'):
        """
        Initialize a SDESolver object with the needed parameters of when to start, when to stop, timestep value,
        number of particles to use and constants for gamma_0, d_0 and tau_0 achieved from theory.
        :param t_start: start time, which is always given as 0
        :param t_stop: stop time for the numerical iterative schemes
        :param dt: timestep value. Choose how often to update the positions and velocities of the particles
        :param number_of_particles: the number of particles to use for ensemble averages
        :param constants: the value used for [gamma_0, d_0, tau_0]. Is used to get correct diffusivity and friction.
        :param method: string containing the method of choice, e.g "em" or "st".
        """
        self.N = number_of_particles
        self.t_0 = t_start
        self.t_stop = t_stop
        self.dt = dt
        self.number_of_timesteps = int((self.t_stop - self.t_0) / self.dt)
        self.N = number_of_particles
        self.constants = constants
        self.method = method

    @staticmethod
    def a_underdamped_langevin_equation(z, t, params):
        """
            Function giving a in the SDE given as the underdamped Langevin equation.
        :param z: the velocity of all the particles as a (N, 3) array.
        :param t: time. Is not used here as a is not dependent explicitly on time.
        :param params: [gamma_0, d_0, tau_0].
        return a as a function of the velocity of the particles
        """
        return - z * params[0]

    @staticmethod
    def da_dz_underdamped_langevin_equation(z, t, params):
        """
            Function giving da_dz in the SDE given as the underdamped Langevin equation.
        :param z: the velocity of all the particles as a (N, 3) array.
        :param t: time. Is not used here as da_dz is not dependent explicitly on time.
        :param params: [gamma_0, d_0, tau_0].
        return da_dz as a constant
        """
        return -params[0]

    @staticmethod
    def b_underdamped_langevin_equation(z, t, params):
        """
            Function giving b in the SDE given as the underdamped Langevin equation.
        :param z: the velocity of all the particles as a (N, 3) array.
        :param t: time. Is not used here as b is not dependent explicitly on time.
        :param params: [gamma_0, d_0, tau_0].
        return b as a constant
        """
        return np.sqrt(2*params[1])*params[0]

    @staticmethod
    def a_udsbm(z, t, params):
        """
            Function giving a in the SDE given as UDSBM.
        :param z: the velocity of all the particles as a (N, 3) array.
        :param t: time. Is used here as a is dependent explicitly on time.
        :param params: [gamma_0, d_0, tau_0].
        return a as a function of the velocity of the particles and time
        """
        return - z * params[0]/(1+t/params[2])

    @staticmethod
    def da_dz_udsbm(z, t, params):
        """
            Function giving da_dz in the SDE given as UDSBM.
        :param z: the velocity of all the particles as a (N, 3) array.
        :param t: time. Is used here as da_dz is dependent explicitly on time.
        :param params: [gamma_0, d_0, tau_0].
        return da_dz as a function of time
        """
        return -params[0]/(1+t/params[2])

    @staticmethod
    def b_udsbm(z, t, params):
        """
            Function giving b in the SDE given as UDSBM.
        :param z: the velocity of all the particles as a (N, 3) array
        :param t: time. Is used here as b is dependent explicitly on time.
        :param params: [gamma_0, d_0, tau_0].
        return b as a function of time
        """
        return np.sqrt(2*params[1]/(1+t/params[2]))*params[0]/(1+t/params[2])

    def euler_maruyama_method(self, z, a, b, t):
        """
            Taking a step in the Euler-Maruyama iterative scheme for a general SDE describing Brownian motion. It can
            be used for both the underdamped Langevin equation and UDSBM which differ due to the time dependence of the
            friction coefficient and the diffusivity giving different a and b. Gives the velocity at next timestep.
        :param z: the velocity of all the particles as a (N, 3) array.
        :param a: function giving the velocity and time dependence of the a term in the SDE.
        :param b: function giving the velocity and time dependence of the b term in the SDE.
        :param t: time, which a and b can be a function of.
        return the velocity at the next timestep
        """
        # the random force is a Wiener process, drawn from a normal distribution with mu = 0, sigma = sqrt(dt)
        dW = np.random.normal(loc=0, scale=np.sqrt(self.dt), size=(self.N, 3))
        # return the velocity at the next timestep as a function of the previous velocity and time
        return z + a(z, t, self.constants)*self.dt + b(z, t, self.constants) * dW

    def strong_taylor_method(self, z, a, b, da_dz, t):
        """
            Taking a step in the strong Taylor iterative scheme for a general SDE describing Brownian motion. It can
            be used for both the underdamped Langevin equation and UDSBM which differ due to the time dependence of the
            friction coefficient and the diffusivity giving different a and b. Gives the velocity at next timestep.
        :param z: the velocity of all the particles as a (N, 3) array.
        :param a: function giving the velocity and time dependence of the a term in the SDE.
        :param b: function giving the velocity and time dependence of the b term in the SDE.
        :param da_dz: function giving the derivative of the a term in the SDE.
        :param t: time, which a, b and da_dz can be a function of.
        return the velocity at the next timestep
        """
        random_numb_1 = np.random.randn(self.N, 3)
        random_numb_2 = np.random.randn(self.N, 3)
        dW = random_numb_1 * np.sqrt(self.dt)
        dZ = (random_numb_1 + random_numb_2 / np.sqrt(3)) * self.dt ** (3 / 2) / 2
        return z + a(z, t, self.constants) * self.dt + b(z, t, self.constants) * dW + \
               b(z, t, self.constants) * da_dz(z, t, self.constants) * dZ + \
               a(z, t, self.constants) * da_dz(z, t, self.constants) * self.dt**2 / 2

    def trajectory(self, a, b, initial_speed):
        """
            Help function to do a iterative scheme of a method in order to solve a SDE describing Brownian motion. From
            input of the function for the a and b is can solve different SDEs, as the underdamped Langevin equation
            or UDSBM. The function creates initial conditions, updates the position from the velocity, and updates the
            velocity by applying a method such as the Euler-Maruyama method or the Strong Taylor method.
        :param a: function giving the velocity and time dependence of the a term in the SDE.
        :param b: function giving the velocity and time dependence of the b term in the SDE.
        :param initial_speed: float giving the initial speed of all the particles, default sqrt(2).
        return the times, positions and velocities computed with given method
        """
        v0 = util_funcs.random_uniformly_distributed_velocities(self.N, initial_speed, 3)  # initial velocities
        x0 = np.tile([0.5, 0.5, 0.5], reps=(self.N, 1))  # initial positions
        # create arrays for all positions and velocities at all timesteps
        positions = np.zeros((self.number_of_timesteps + 1, self.N, 3))
        velocities = np.zeros_like(positions)
        # initialize with initial values
        positions[0, :, :] = x0
        velocities[0, :, :] = v0

        times = np.arange(self.number_of_timesteps + 1) * self.dt  # compute the time at all timesteps

        # choose method and then compute the iterative scheme by updating the position and velocity at each timestep
        if self.method == "em":
            for i in range(self.number_of_timesteps):
                positions[i + 1, :, :] = positions[i, :, :] + velocities[i, :, :] * self.dt
                velocities[i + 1, :, :] = self.euler_maruyama_method(velocities[i, :, :], a, b, times[i])
        elif self.method == "st":
            if self.constants[2] == np.inf:
                da_dz = self.da_dz_underdamped_langevin_equation
            else:
                da_dz = self.da_dz_udsbm
            for i in range(self.number_of_timesteps):
                positions[i + 1, :, :] = positions[i, :, :] + velocities[i, :, :] * self.dt
                velocities[i + 1, :, :] = self.strong_taylor_method(velocities[i, :, :], a, b, da_dz, times[i])

        return times, positions, velocities

    def compute_ensemble_msd(self, a, b, initial_speed):
        """
            Help function to compute the mean square displacement and the mean square speed for the solution to a SDE
            describing Brownian motion. From input of the function for the a and b is can solve different SDEs, as the
            underdamped Langevin equation or UDSBM. The function use the trajectory to create the position and velocity
            at all times before computing the msd and mss and return it.
        :param a: function giving the velocity and time dependence of the a term in the SDE.
        :param b: function giving the velocity and time dependence of the b term in the SDE.
        :param initial_speed: float giving the initial speed of all the particles, default sqrt(2).
        :return time_array, mean_square_distance_array, mean_square_speed_array
        """
        times, positions, velocities = self.trajectory(a, b, initial_speed)  # solve SDE with initialized method
        dx = positions - positions[0, :, :]  # compute the dx vector from the initial position for all particles
        msd = np.mean(norm(dx, axis=2) ** 2, axis=1)  # compute the mean square displacement
        mss = np.mean(norm(velocities, axis=2) ** 2, axis=1)  # compute the mean square speed

        return times, msd, mss

    def compute_ensemble_and_time_averaged_msd(self, a, b, initial_speed):
        """
            Similar function as ensemble_msd, but will additionally compute the time averaged msd which can be used
            to prove properties of Ergodicity. The time averaged values is computed with the Simpson's method.
        :param a: function giving the velocity and time dependence of the a term in the SDE.
        :param b: function giving the velocity and time dependence of the b term in the SDE.
        :param initial_speed: float giving the initial speed of all the particles, default sqrt(2).
        :return time_array/delta_array, ensemble_msd_array, time_averaged_msd_array, mean_square_speed_array
        """
        times, positions, velocities = self.trajectory(a, b, initial_speed)  # solve SDE with initialized method
        delta_values = times  # use the same discretization for delta as for time
        time_averaged_msd = np.zeros_like(delta_values)  # array to store the time averaged msd values
        # iterate through the values for delta and compute the time averaged msd value
        for counter, delta in enumerate(delta_values[1:-1], 1):
            x_t_delta = positions[counter:]  # positions at time t+delta
            x_t = positions[:-counter]  # positions at time t
            integrand = np.mean(np.sum((x_t_delta - x_t)**2, axis=2), axis=1)  # integrand for time integral
            # integrand = np.mean(norm((x_t_delta - x_t), axis=2)**2, axis=1)  # more intuitive, but slower
            # integrate the mean square difference between the position at t+delta and t from 0 to t-delta
            time_averaged_msd[counter] = simps(y=integrand, x=times[:-counter])/(times[-1]-delta)

        dx = positions - positions[0, :, :]  # compute the dx vector from the initial position for all particles
        ensemble_msd = np.mean(norm(dx, axis=2) ** 2, axis=1)  # compute the mean square displacement
        mss = np.mean(norm(velocities, axis=2) ** 2, axis=1)  # compute the mean square speed
        return times, ensemble_msd, time_averaged_msd, mss


# In addition to SDESolver, we give some utility functions that use the SDESolver to compute msd from SDE.


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


def compute_constants(number_of_particles, restitution_coefficient, initial_speed, radius):
    """
        Help function to compute gamma_0, d_0 and tau_0 for a given set of particle parameters.
    :param number_of_particles: the number of particles to solve the SDE for at the same time.
    :param restitution_coefficient: the restitution coefficient of the system.
    :param initial_speed: initial speed used to compute initial temperature.
    :param radius: radius of the particles.
    return gamma_0, d_0, gamma_0
    """
    if restitution_coefficient == 1:
        assert number_of_particles == 1000 and radius == 0.025, \
            "Given values for gamma0, d0 are only valid for N=1000 and r=0.025!"
        gamma0, d0, tau_0 = 11.43, 0.058, np.inf
    else:
        # compute correct gamma0, d0, tau_0 from N, r and xi. See report for equations
        number_density = number_of_particles / 1
        packing_fraction = number_density * 4 * np.pi * radius ** 3 / 3
        # contact value for the equilibrium pair correlation function for hard spheres
        g_2 = (2 - packing_fraction) / (2 * (1 - packing_fraction) ** 3)
        mass = 1
        initial_temperature = initial_speed**2/3
        # initial diffusivity given in rapport
        d0 = 3*np.sqrt(initial_temperature)/((1+restitution_coefficient)**2*np.sqrt(mass*np.pi)*number_density*g_2*(2*radius)**2*2)
        # initial friction coefficient given in rapport
        gamma0 = initial_temperature/(d0*mass)
        # characteristic timescale of the evolution of the granular temperature given in rapport
        tau_0 = (np.sqrt(initial_temperature/mass)*(1-restitution_coefficient**2)/6*4*np.sqrt(np.pi)*g_2*(2*radius)**2*number_density)**(-1)
        # round to 2 or 3 decimals to ensure less number representation issues
        tau_0 = np.round(tau_0, decimals=2)
        gamma0 = np.round(gamma0, decimals=2)
        d0 = np.round(d0, decimals=3)
    return gamma0, d0, tau_0


def mean_square_displacement_from_sde(particle_parameters, sde_parameters, number_of_runs):
    """
        Function used to compute mean square displacement and mean square speed of the particles for the different
        SDEs describing Brownian motion. This function use the class SDESolver to get the results and
        then saves them to file by applying the Euler-Maruyama method. Can be changed to use Strong Taylor as well.
    :param particle_parameters: [N, xi, v0, r] used to get correct constants and size of arrays
    :param sde_parameters: [t_stop, dt] used to do a iterative scheme for SDEs.
    :param number_of_runs: int telling how many different runs to do in order to check convergence from clt.
    """
    N, xi, v0, r = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    gamma_0, d_0, tau_0 = compute_constants(N, xi, v0, r)
    t_stop, dt = sde_parameters[0], sde_parameters[1]

    # initialize parameters for memory concern
    number_of_values = int(t_stop/dt)+1
    times = np.zeros(number_of_values)
    msd = np.zeros_like(times)
    mss = np.zeros_like(times)

    for run_number in range(number_of_runs):
        print(f'Run number: {run_number}')
        sde_solver = SDESolver(t_start=0,
                               t_stop=t_stop,
                               dt=dt,
                               number_of_particles=N,
                               constants=[gamma_0, d_0, tau_0],
                               method='em')
        if xi == 1:
            times, msd, mss = sde_solver.compute_ensemble_msd(sde_solver.a_underdamped_langevin_equation,
                                                              sde_solver.b_underdamped_langevin_equation, v0)
        else:
            # the value of v0 below has better agreement with theory, but do not match initial T0..
            # v0 = np.sqrt(3*d0*gamma0**2*tau/(gamma0*tau-1))  # for correct initial speed compared to theory
            times, msd, mss = sde_solver.compute_ensemble_msd(sde_solver.a_udsbm, sde_solver.b_udsbm, v0)
        # store data in matrix
        msd_matrix = np.zeros((len(times), 3))
        msd_matrix[:, 0] = times
        msd_matrix[:, 1] = msd
        msd_matrix[:, 2] = mss
        # save matrix to file
        np.save(file=os.path.join(results_folder, f'msd_sde_N_{N}_r_{r}_xi_{xi}_tstop_{t_stop}_dt_{dt}_{run_number}'),
                arr=msd_matrix)


def ensemble_and_time_averaged_msd_from_sde(particle_parameters, sde_parameters, run_number):
    """
        Similar function as mean_square_displacement_from_sde, but will in addition compute the time averaged msd. With
        this additional information we can look if the system exhibit the principle of Ergodicity. After computing
        the results, this function saves the results.
    :param particle_parameters: [N, xi, v0, r] used to get correct constants and size of arrays
    :param sde_parameters: [t_stop, dt] used to do a iterative scheme for SDEs.
    :param run_number: int used such that one can run parallel simulations and save results to different files.
    """
    N, xi, v0, r = int(particle_parameters[0]), particle_parameters[1], particle_parameters[2], particle_parameters[3]
    gamma_0, d_0, tau_0 = compute_constants(N, xi, v0, r)
    t_stop, dt = sde_parameters[0], sde_parameters[1]

    print(f'Run number: {run_number}')
    sde_solver = SDESolver(t_start=0,
                           t_stop=t_stop,
                           dt=dt,
                           number_of_particles=N,
                           constants=[gamma_0, d_0, tau_0],
                           method='em')
    if xi == 1:
        times, ensemble_msd, time_averaged_msd, mss = \
            sde_solver.compute_ensemble_and_time_averaged_msd(sde_solver.a_underdamped_langevin_equation,
                                                              sde_solver.b_underdamped_langevin_equation, v0)
    else:
        # the value of v0 below has better agreement with theory, but do not match initial T0..
        # v0 = np.sqrt(3*d0*gamma0**2*tau/(gamma0*tau-1))  # for correct initial speed compared to theory
        times, ensemble_msd, time_averaged_msd, mss = \
            sde_solver.compute_ensemble_and_time_averaged_msd(sde_solver.a_udsbm, sde_solver.b_udsbm, v0)
    # store data in matrix
    msd_matrix = np.zeros((len(times), 4))
    msd_matrix[:, 0] = times
    msd_matrix[:, 1] = ensemble_msd
    msd_matrix[:, 2] = time_averaged_msd
    msd_matrix[:, 3] = mss
    # save matrix to file
    np.save(file=os.path.join(results_folder, f'ergodicity_msd_sde_N_{N}_r_{r}_xi_{xi}_tstop_{t_stop}_dt_{dt}_'
                                              f'{run_number}'), arr=msd_matrix)
