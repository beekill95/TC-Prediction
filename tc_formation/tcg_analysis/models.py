import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def hier_tcg_trend_model(tcg_freq: jnp.ndarray, year: jnp.ndarray, period: jnp.ndarray, nb_period: int = 2):
    """
    Perform TCG trend analysis using hierarchical Bayesian model.

    tcg_freq: jnp.ndarray
        Number of TCG in each year.
    year: jnp.ndarray
        Corresponding year in `tcg_freq`.
    period: jnp.ndarray
        Whether the year belongs to middle of century (2030-2050) or the end of century (2080-2100).
    nb_period: int
        Number of periods in `period`. Current, it has to be 2.
    """
    assert nb_period == 2, 'Right now, only support two periods.'

    tcg_mean_log = jnp.log(jnp.mean(tcg_freq))
    tcg_sd_log = jnp.log(jnp.std(tcg_freq))

    # Normalize the years for each period to prevent divergencies.
    # year_means = _year_means(year, period, nb_period)
    # year_stds = _year_stds(year, period, nb_period)
    # year_Z = jnp.zeros_like(year)
    nb_obs = tcg_freq.shape[0]
    # for i in range(nb_obs):
    #     p = period[i]
    #     z = (year[i] - year_means[p]) / year_stds[p]
    #     year_Z.at[i].set(z)

    # Specify the priors for the baseline base on the scale of the data.
    a0 = numpyro.sample('_a0', dist.Normal(tcg_mean_log, tcg_sd_log * 3))

    # Specify the priors for the coefficients for deflection based on years.
    a1_mean = numpyro.sample('_a1_mean', dist.Normal(0, 1))
    a1_sd = numpyro.sample('_a1_sd', gamma_from_mode_std(1., 1.))

    # Specify the priors for each period.
    # Here, we use a1 = a1_mean + Normal(0, 1) * a1_sd to prevent divergencies.
    a1_Z = numpyro.sample('_a1_Z', dist.Normal(0, 1).expand((nb_period, )))
    a1 = numpyro.deterministic('_a1', a1_mean + a1_Z * a1_sd)

    # We can now specify the observed TC frequencies.
    with numpyro.plate('obs', nb_obs) as idx:
        p = period[idx]
        mean = numpyro.deterministic('mean', jnp.exp(a0 + year[idx] * a1[p]))
        numpyro.sample('tcg_freq', dist.Poisson(mean), obs=tcg_freq[idx])

    # Now, we will convert the baseline and coefficients back to the original scale.
    # FIXME: this is not correct!, a0 is a scalar, while a1 is a vector with `nb_period` elements.
    # numpyro.deterministic('b0', a0 - jnp.sum(a1 * year_means / year_stds))
    # numpyro.deterministic('b1', a1 / year_stds)
    # numpyro.deterministic('b1_mean', a1_mean / year_stds)
    # numpyro.deterministic('b1_sd', a1_sd / year_stds)


def hier_tcg_trend_year_rcp_model(tcg_freq: jnp.ndarray, year: jnp.ndarray, rcp: jnp.ndarray):
    """
    Perform TCG trend analysis using Poisson distribution for tcg_freq,
    and metric predictor `year` and categorical predictor `rcp`.

    tcg_freq ~ Poisson(lambda)
    where lambda = exp(b0 + b1 * year + b2[rcp])

    Also, the model assumes that there are only two scenarios in rcp.
    """
    nb_rcp_scenarios = 2

    # To specify the priors of baseline.
    tcg_mean_log = jnp.log(jnp.mean(tcg_freq))
    tcg_sd_log = jnp.log(jnp.std(tcg_freq))

    # Normalize the years.
    year_mean = jnp.mean(year)
    year_sd = jnp.std(year)
    year_Z = (year - year_mean) / year_sd
    # year_centered = (year - year_mean

    # Specify the priors for the baseline base on the scale of the data.
    a0 = numpyro.sample('_a0', dist.Normal(tcg_mean_log, tcg_sd_log * 3))

    # Specify the priors for the coefficients of the deflection based on years.
    a1 = numpyro.sample('_a1', dist.Normal(0, 2 * tcg_sd_log / year_sd))

    # Specify the priors for the coefficients of the deflection based on rcp scenarios.
    a2_mean = numpyro.sample('_a2_mean', dist.Normal(0, 1))
    a2_sd = numpyro.sample('_a2_sd', gamma_from_mode_std(1., 1.))

    # Use a2 = mean + Normal * std
    a2_Z = numpyro.sample('_a2_Z', dist.Normal(0, 1).expand((nb_rcp_scenarios, )))
    a2 = numpyro.deterministic('_a2', a2_mean + a2_Z * a2_sd)

    # Observations.
    nb_obs = tcg_freq.shape[0]
    with numpyro.plate('obs', nb_obs) as idx:
        mean = numpyro.deterministic(
            'mean', jnp.exp(a0 + a1 * year_Z[idx] + a2[rcp[idx]]))
        numpyro.sample('y', dist.Poisson(mean), obs=tcg_freq[idx])

    # Transform back to b, and impose sum-to-zero constraints.
    a2_mean = jnp.mean(a2)
    numpyro.deterministic('b0', a0 + a2_mean - a1 * year_mean / year_sd)
    numpyro.deterministic('b1', a1 / year_sd)
    numpyro.deterministic('b2', a2 - a2_mean)


def hier_tcg_trend_year_rcp_cluster_model(tcg_freq: jnp.ndarray, year: jnp.ndarray, rcp: jnp.ndarray, cluster: jnp.ndarray):
    """
    Perform TCG trend analysis using Poisson distribution for tcg_freq,
    and metric predictor `year` and categorical predictor `rcp`.

    tcg_freq ~ Poisson(lambda)
    where lambda = exp(b0 + b1 * year + b2[rcp])

    Also, the model assumes that there are only two scenarios in rcp.
    """
    nb_rcp_scenarios = 2
    nb_cluster = 2

    # To specify the priors of baseline.
    tcg_mean_log = jnp.log(jnp.mean(tcg_freq))
    tcg_sd_log = jnp.log(jnp.std(tcg_freq))

    # Normalize the years.
    year_mean = jnp.mean(year)
    year_sd = jnp.std(year)
    year_Z = (year - year_mean) / year_sd
    # year_centered = (year - year_mean

    # Specify the priors for the baseline base on the scale of the data.
    a0 = numpyro.sample('_a0', dist.Normal(tcg_mean_log, tcg_sd_log * 3))

    # Specify the priors for the coefficients of the deflection based on years.
    a1 = numpyro.sample('_a1', dist.Normal(0, 2 * tcg_sd_log / year_sd))

    # Specify the priors for the coefficients of the deflection based on rcp scenarios.
    a2_mean = numpyro.sample('_a2_mean', dist.Normal(0, 1))
    a2_sd = numpyro.sample('_a2_sd', gamma_from_mode_std(1., 1.))

    # Use a2 = mean + Normal * std
    a2_Z = numpyro.sample('_a2_Z', dist.Normal(0, 1).expand((nb_rcp_scenarios, )))
    a2 = numpyro.deterministic('_a2', a2_mean + a2_Z * a2_sd)

    # Specify the priors for the coefficients of the deflection based on rcp scenarios.
    a3_mean = numpyro.sample('_a3_mean', dist.Normal(0, 1))
    a3_sd = numpyro.sample('_a3_sd', gamma_from_mode_std(1., 1.))

    # Use a3 = mean + Normal * std
    a3_Z = numpyro.sample('_a3_Z', dist.Normal(0, 1).expand((nb_cluster, )))
    a3 = numpyro.deterministic('_a3', a3_mean + a3_Z * a3_sd)

    # Observations.
    nb_obs = tcg_freq.shape[0]
    with numpyro.plate('obs', nb_obs) as idx:
        mean = numpyro.deterministic(
            'mean', jnp.exp(a0 + a1 * year_Z[idx] + a2[rcp[idx]] + a3[cluster[idx]]))
        numpyro.sample('y', dist.Poisson(mean), obs=tcg_freq[idx])

    # Transform back to b, and impose sum-to-zero constraints.
    a2_mean = jnp.mean(a2)
    a3_mean = jnp.mean(a3)
    numpyro.deterministic('b0', a0 + a2_mean + a3_mean - a1 * year_mean / year_sd)
    numpyro.deterministic('b1', a1 / year_sd)
    numpyro.deterministic('b2', a2 - a2_mean)
    numpyro.deterministic('b3', a3 - a3_mean)


def gamma_from_mode_std(mode, std):
    std_squared = std**2
    rate = (mode + jnp.sqrt(mode**2 + 4 * std_squared)) / (2 * std_squared)
    shape = 1 + mode * rate

    return dist.Gamma(shape, rate)


def _year_means(year: jnp.ndarray, period: jnp.ndarray, nb_period: int):
    means = []
    for i in range(nb_period):
        means.append(jnp.mean(year, where=(period == i)))

    return jnp.array(means)


def _year_stds(year: jnp.ndarray, period: jnp.ndarray, nb_period: int):
    stds = []
    for i in range(nb_period):
        stds.append(jnp.std(year, where=(period == i)))

    return jnp.array(stds)
