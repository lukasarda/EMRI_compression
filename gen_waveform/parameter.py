import numpy as np


def para_con(no_of_draws, rng_seed=None):
    rng = np.random.default_rng(rng_seed)

    center_M = 1.2e6
    rel_var_M = 0.05

    center_mu = 1.e2
    rel_var_mu = 0.05

    p0 = 12.0
    e0 = 0.4
    theta = np.pi / 3  # polar viewing angle
    phi = np.pi / 4  # azimuthal viewing angle
    dt = 10.0  # seconds -- sample frequency
    T = 1./12. # years -- total length of waveform

    def rnd_uniform_dist(center, rel_var, no_of_draws):
        upper_bound = center + center * rel_var
        lower_bound = center - center * rel_var
        return rng.uniform(lower_bound, upper_bound, no_of_draws)

    M_params = rnd_uniform_dist(center_M, rel_var_M, no_of_draws)
    mu_params = rnd_uniform_dist(center_mu, rel_var_mu, no_of_draws)
    
    param_space = np.column_stack((
        M_params,
        mu_params,
        np.full(no_of_draws, p0),
        np.full(no_of_draws, e0),
        np.full(no_of_draws, theta),
        np.full(no_of_draws, phi),
        np.full(no_of_draws, dt),
        np.full(no_of_draws, T)
    ))

    return param_space, T