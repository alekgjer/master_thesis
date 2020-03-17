import os
import numpy as np
import matplotlib.pyplot as plt

import utility_functions as util_funcs

from scipy.linalg import norm
from scipy.integrate import simps

from config import results_folder, init_folder

# Some functionality implemented to solve SDE numerically. The SDE of interest are the Langevin equation, which can be
# used to approximate the effects in a molecular gas and in a granular gas.


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


def A(times, delta, gamma0, T0):
    return -T0*(np.exp(-gamma0*delta)-1+np.exp(-gamma0*times)-np.exp(-gamma0*(times+delta)))/(gamma0**2)


def time_average_msd(x, times, dt):
    delta_values = times
    number_of_timesteps = len(times)-1
    msd = np.zeros_like(delta_values)
    dx = norm(x - x[0, :, :], axis=2)
    msd_values = np.mean(dx**2, axis=1)
    gamma0, T0 = 11.43, 2/3
    for counter, delta in enumerate(delta_values[1:-1], 1):
        print(counter, times[-1]-delta)
        # dx_t_delta = dx[counter:]
        # dx_t = dx[:-counter]
        # integrand = dx_t_delta - dx_t
        # integrand = np.mean(integrand**2, axis=1)
        # msd[counter] = simps(integrand, dx=dt)/(times[-1]-delta)
        msd_t_delta = msd_values[counter:]
        msd_t = msd_values[:-counter]
        a = A(times[:-counter], delta, gamma0, T0)
        integrand = msd_t_delta-msd_t-2*a
        msd[counter] = simps(integrand, dx=dt)/(times[-1]-delta)

    ensemble_msd = np.mean(dx**2, axis=1)
    plt.loglog(delta_values[1:-1], msd[1:-1], label='Time averaged MSD')
    plt.loglog(times, ensemble_msd, label='Ensemble MSD')
    plt.legend()
    plt.show()


def solve_underdamped_langevin_equation(number_of_particles, dt, t_stop):
    """
        Function to solve the underdamped Langevin equation numerically by applying the Euler-Maruyama method. Atm it
        saves the results of the mean square displacement and mean square speed to file.
    :param number_of_particles: number_of_particles to solve for at the same time.
    :param dt: timestep value. The discretization in time used in the iterative scheme.
    :param t_stop: the end time of the simulation. Will do t_stop/dt timesteps.
    """
    print('Solving the underdamped Langevin equation..')
    v0 = util_funcs.random_uniformly_distributed_velocities(number_of_particles, np.sqrt(2), 3)  # initial velocities
    x0 = np.tile([0.5, 0.5, 0.5], reps=(number_of_particles, 1))  # initial positions

    t_start = 0

    number_of_timesteps = int((t_stop-t_start)/dt)
    # create arrays for all positions and velocities at all timesteps
    x = np.zeros((number_of_timesteps+1, number_of_particles, 3))
    v = np.zeros((number_of_timesteps+1, number_of_particles, 3))
    # initialize with initial values
    x[0, :, :] = x0
    v[0, :, :] = v0

    # parameters for the diffusion coefficient and the damping/friction coefficient
    gamma0, d0 = 11.43, 0.058  # found from 3d simulations of molecular gas with xi=1 and eta = 0.065.
    # the random force is a Wiener process which is drawn from a normal distribution with mu = 0, sigma = sqrt(dt)

    # compute the Euler-Maruyama iterative scheme for the underdamped Langevin equation for all particles simultaneously
    for i in range(number_of_timesteps):
        # the random force is a Wiener process which is drawn from a normal distribution with mu = 0, sigma = sqrt(dt)
        random_force_timestep = np.random.normal(loc=0, scale=np.sqrt(dt), size=(number_of_particles, 3))
        x[i+1, :, :] = x[i, :, :] + v[i, :, :]*dt
        v[i+1, :, :] = v[i, :, :] - gamma0*v[i, :, :]*dt + np.sqrt(2*d0)*gamma0*random_force_timestep

    dx = x-x[0, :, :]  # compute the dx vector from the initial position for all particles
    # msd = np.mean(norm(dx, axis=2)**2, axis=1)  # compute the mean square displacement
    # mss = np.mean(norm(v, axis=2)**2, axis=1)  # compute the mean square speed

    times = np.arange(number_of_timesteps+1)*dt  # array of all times

    # msd_matrix = np.zeros((number_of_timesteps+1, 3))  # matrix to save to file
    # msd_matrix[:, 0] = times
    # msd_matrix[:, 1] = msd
    # msd_matrix[:, 2] = mss

    # plt.loglog(times, msd, label='Numerical values')
    # plt.loglog(times, msd_elasticparticles(times, d0, gamma0), label='Theoretical values')
    # plt.legend()
    # plt.show()
    time_average_msd(x, times, dt)
    # save to file
    # np.save(file=os.path.join(results_folder, f'msd_sde_N_{number_of_particles}_xi_1_tstop_{t_stop}_dt_{dt}'),
    #         arr=msd_matrix)


def gamma(t, gamma0, tau):
    return gamma0/(1+t/tau)


def d(t, d0, tau):
    return d0/(1+t/tau)


def msd_gg(t, d0, gamma0, tau):
    msd_1d = 2*d0*tau*(np.log(1+t/tau)+(1/(tau*gamma0))*((1+t/tau)**(-tau*gamma0)-1))
    return 3*msd_1d


def msd_udsbm(t, d0, gamma0, tau):
    msd_1d = 2*d0*tau**2*gamma0**2*(tau*np.log(1+t/tau)+(tau/(tau*gamma0-1))*((1+t/tau)**(-tau*gamma0+1)-1))/((tau*gamma0-1)**2)
    return 3*msd_1d


def msd_elasticparticles(t, d0, gamma0):
    msd_1d = 2*d0*t+2*d0*(np.exp(-gamma0*t)-1)/gamma0
    return 3*msd_1d


def solve_udsbm_langevin_equation(number_of_particles, dt, t_stop):
    """
        Function to solve the udsbm Langevin equation numerically by applying the Euler-Maruyama method. Atm it
        saves the results of the mean square displacement and mean square speed to file.
    :param number_of_particles: number_of_particles to solve for at the same time.
    :param dt: timestep value. The discretization in time used in the iterative scheme.
    :param t_stop: the end time of the simulation. Will do t_stop/dt timesteps.
    """
    print('Solving the ubsbm Langevin equation..')
    v0 = util_funcs.random_uniformly_distributed_velocities(number_of_particles, np.sqrt(2), 3)  # initial velocities
    x0 = np.tile([0.5, 0.5, 0.5], reps=(number_of_particles, 1))  # initial positions

    t_start = 0

    number_of_timesteps = int((t_stop - t_start) / dt)
    # create arrays for all positions and velocities at all timesteps
    x = np.zeros((number_of_timesteps + 1, number_of_particles, 3))
    v = np.zeros((number_of_timesteps + 1, number_of_particles, 3))
    # initialize with initial values
    x[0, :, :] = x0
    v[0, :, :] = v0

    gamma0, d0, tau = 9.21, 0.072, 0.97  # found from 3d simulations of granular gas with xi=0.8 and eta = 0.065.
    times = np.arange(number_of_timesteps+1)*dt

    gamma_all_times = gamma(times, gamma0, tau)
    d_all_times = d(times, d0, tau)

    for i in range(number_of_timesteps):
        # Euler-Maruyama method
        random_force_timestep = np.random.normal(loc=0, scale=np.sqrt(dt), size=(number_of_particles, 3))
        x[i + 1, :, :] = x[i, :, :] + v[i, :, :] * dt
        v[i + 1, :, :] = v[i, :, :] - gamma_all_times[i] * v[i, :, :] * dt + np.sqrt(2 * d_all_times[i]) * gamma_all_times[i] * random_force_timestep
        # Strong Taylor approx
        # random_numb_1 = np.random.randn(number_of_particles, 3)
        # random_numb_2 = np.random.randn(number_of_particles, 3)
        # dW = random_numb_1*np.sqrt(dt)
        # dZ = (random_numb_1+random_numb_2/np.sqrt(3))*dt**(3/2)/2
        # x[i + 1, :, :] = x[i, :, :] + v[i, :, :] * dt + 1/2 * v[i, :, :] * dt**2
        # v[i + 1, :, :] = v[i, :, :] - gamma_all_times[i] * v[i, :, :] * dt + np.sqrt(2 * d_all_times[i]) * \
        #                  gamma_all_times[i] * dW - np.sqrt(2 * d_all_times[i]) * gamma_all_times[i]**2 * dZ + \
        #                  gamma_all_times[i]**2*v[i, :, :]*dt**2/2


    x = x[::10, :, :]
    # dx = x - x[0, :, :]  # compute the dx vector from the initial position for all particles
    # msd = np.mean(norm(dx, axis=2) ** 2, axis=1)  # compute the mean square displacement
    # mss = np.mean(norm(v, axis=2) ** 2, axis=1)  # compute the mean square speed

    # msd_matrix = np.zeros((number_of_timesteps + 1, 3))  # matrix to save to file
    # msd_matrix[:, 0] = times
    # msd_matrix[:, 1] = msd
    # msd_matrix[:, 2] = mss

    # plt.loglog(times[::10], msd, label='Numerical values')
    # plt.loglog(times[::10], msd_udsbm(times[::10], d0, gamma0, tau), label='Theoretical values')
    # plt.legend()
    # plt.show()
    time_average_msd(x, times[::10], dt*10)
    # save to file
    # np.save(file=os.path.join(results_folder, f'msd_sde_N_{number_of_particles}_xi_0.8_tstop_{t_stop}_dt_{dt}'),
    #         arr=msd_matrix)
