import numpy as np
#from numba import njit
rng = np.random.default_rng()

### no_of_draws is how many parameter sets are passed to the waveform generator

#@njit
def para_con(no_of_draws):

    center_M = 1.2e6
    rel_var_M = 0.05

    center_mu = 1.e2
    rel_var_mu = 0.05

    p0 = 12.0
    e0 = 0.4
    theta = np.pi/3  # polar viewing angle
    phi = np.pi/4  # azimuthal viewing angle
    dt = 10.0 # seconds -- sample frequency
    T = 0.1 # years -- total length of waveform


    # dist = 'random_gauss':
    def random_gauss_dist(center, rel_var, no_of_draws):
       return center + rel_var * center * rng.standard_normal(no_of_draws)
    
    def rnd_uniform_dist(center, rel_var, no_of_draws):
        upper_bound = center + center * rel_var
        lower_bound = center - center * rel_var
        return np.random.uniform(lower_bound, upper_bound, no_of_draws)


    # M_params = random_gauss_dist(center_M, rel_var_M, no_of_draws)
    # mu_params = random_gauss_dist(center_mu, rel_var_mu, no_of_draws)

    M_params = rnd_uniform_dist(center_M, rel_var_M, no_of_draws)
    mu_params = rnd_uniform_dist(center_mu, rel_var_mu, no_of_draws)

    
    def array_constructor(M, mu, p0, e0, theta, phi, dt, T, no_of_draws):
        param_space = []
        for i in range(no_of_draws):
            param_space.append(np.array((M[i], mu[i], p0[i], e0[i], theta[i], phi[i], dt[i], T[i])))

        return np.array(param_space)
        

    param_space =  array_constructor(M_params, mu_params, np.full(no_of_draws, p0), np.full(no_of_draws, e0), np.full(no_of_draws, theta), np.full(no_of_draws, phi), np.full(no_of_draws, dt), np.full(no_of_draws, T), no_of_draws)

    #with open('./run_parameters.txt', "a") as myfile:
    #    myfile.write(str(no_of_draws))
    #    myfile.write(str(list(param_space)))

    return(param_space)
