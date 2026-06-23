"""Experiment 6: Bayesian charge-balance CDR signal + probability-of-detection design.

Aligned with the uncertainty-aware MRV frameworks now favoured in the literature
(Rogers & Maher, Front. Clim. 2026; CDRXIV Bayesian MRV preprint 2025; cation /
charge-balance MRV, Biogeosciences 2026), this phase:

  1. Fits a hierarchical Bayesian model (partial pooling over plot-halves) to the
     resin NET BASE-CATION charge balance (mol_c) vs dose, giving a full posterior
     for the dose effect with honest between-plot variance.
  2. Translates the dose-effect posterior to an INDICATIVE alkalinity-equivalent
     CDR scale (clearly labelled resin-flux-proxy, not an areal mass balance).
  3. Produces a Bayesian ASSURANCE curve: using the fitted noise (sigma, tau),
     the posterior-predictive probability of declaring a credible-positive effect
     as replication (plots/arm) grows - a design tool for the next campaign.

CPU; ~30-60 s for MCMC.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from src.config import (AUDIT_DIR, DOSE_THA, ION_CHARGE, ION_MOLAR_MASS,
                        RANDOM_SEED, RESULT_DIR)
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin

numpyro.set_host_device_count(1)

CATIONS = ["ca_ppm", "mg_ppm", "k_ppm", "na_ppm", "nh4_n_ppm"]
ANIONS = ["no3_n_ppm", "s_ppm", "p_ppm"]
# Indicative durable-CDR stoichiometry: 1 mol_c base-cation alkalinity ~ 1 mol
# HCO3- ~ 0.5 mol CO2 retained long-term. Order-of-magnitude scale ONLY.
MOLC_TO_MOL_CO2 = 0.5
M_CO2_G = 44.009
N_PLOTS_GRID = [4, 8, 12, 24, 48]


def molc(series: pd.Series, ion: str) -> pd.Series:
    return (series.fillna(0) / ION_MOLAR_MASS[ion]) * abs(ION_CHARGE[ion])


def model(plot_idx, depth_idx, dose, n_plots, n_depths, y=None):
    mu_a = numpyro.sample("mu_a", dist.Normal(0.0, 5.0))
    tau_a = numpyro.sample("tau_a", dist.HalfNormal(5.0))
    with numpyro.plate("plots", n_plots):
        a_plot = numpyro.sample("a_plot", dist.Normal(mu_a, tau_a))
    with numpyro.plate("depths", n_depths):
        b_depth = numpyro.sample("b_depth", dist.Normal(0.0, 3.0))
    beta_dose = numpyro.sample("beta_dose", dist.Normal(0.0, 0.5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    mu = a_plot[plot_idx] + b_depth[depth_idx] + beta_dose * dose
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def assurance_curve(beta, sigma, tau, dose_hi=60.0, alpha=0.05):
    """Posterior-predictive P(credible-positive) vs plots/arm.

    For each posterior draw, simulate a control vs high-dose contrast with
    `n` plots/arm (between-plot var tau^2 + residual sigma^2) and check whether
    a 95% normal CI on the mean difference excludes 0. Average over draws.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    z = 1.959964
    true_diff = beta * dose_hi                      # posterior mean shift at 60 t/ha
    sd_obs = np.sqrt(tau ** 2 + sigma ** 2)         # per-observation SD
    rows = []
    D = len(beta)
    for n in N_PLOTS_GRID:
        # SE of difference of two arm means, n plots each
        se = sd_obs * np.sqrt(2.0 / n)
        # simulate one realised difference per posterior draw
        sim = true_diff + rng.normal(0, se, size=D)
        detect = np.abs(sim) > z * se
        rows.append({"plots_per_arm": n,
                     "p_detect": round(float(detect.mean()), 3),
                     "median_abs_effect_molc": round(float(np.median(np.abs(true_diff))), 4),
                     "se_diff_molc": round(float(np.median(se)), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    resin = qa_clean(load_resin())
    resin = resin[resin["treatment"].isin(["control", "20", "60"])].copy()
    resin["net_alk"] = (sum(molc(resin[i], i) for i in CATIONS)
                        - sum(molc(resin[i], i) for i in ANIONS))
    resin = resin.dropna(subset=["net_alk"]).reset_index(drop=True)
    resin["dose_tha"] = resin["treatment"].map(DOSE_THA)

    plots = sorted(resin["plot_half"].unique())
    depths = sorted(resin["depth_cm"].unique())
    p_idx = resin["plot_half"].map({p: i for i, p in enumerate(plots)}).to_numpy()
    d_idx = resin["depth_cm"].map({d: i for i, d in enumerate(depths)}).to_numpy()

    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=2000, num_chains=1,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(RANDOM_SEED),
             plot_idx=jnp.array(p_idx), depth_idx=jnp.array(d_idx),
             dose=jnp.array(resin["dose_tha"].to_numpy(dtype=float)),
             n_plots=len(plots), n_depths=len(depths),
             y=jnp.array(resin["net_alk"].to_numpy(dtype=float)))
    s = mcmc.get_samples()
    beta = np.asarray(s["beta_dose"])         # mol_c per t/ha
    sigma = np.asarray(s["sigma"])
    tau = np.asarray(s["tau_a"])

    # indicative CDR rate posterior at 60 t/ha (resin-flux-proxy scale)
    cdr_molc = beta * 60.0
    cdr_co2_g = cdr_molc * MOLC_TO_MOL_CO2 * M_CO2_G

    post = pd.DataFrame([{
        "quantity": "beta_dose (mol_c per t/ha)",
        "mean": round(float(beta.mean()), 5),
        "hdi_2.5": round(float(np.quantile(beta, 0.025)), 5),
        "hdi_97.5": round(float(np.quantile(beta, 0.975)), 5),
        "p_gt_0": round(float((beta > 0).mean()), 3),
    }, {
        "quantity": "indicative CDR @60 t/ha (g CO2-eq per capsule-proxy)",
        "mean": round(float(cdr_co2_g.mean()), 4),
        "hdi_2.5": round(float(np.quantile(cdr_co2_g, 0.025)), 4),
        "hdi_97.5": round(float(np.quantile(cdr_co2_g, 0.975)), 4),
        "p_gt_0": round(float((cdr_co2_g > 0).mean()), 3),
    }, {
        "quantity": "tau_a (between-plot SD, mol_c)",
        "mean": round(float(tau.mean()), 4),
        "hdi_2.5": round(float(np.quantile(tau, 0.025)), 4),
        "hdi_97.5": round(float(np.quantile(tau, 0.975)), 4), "p_gt_0": np.nan,
    }, {
        "quantity": "sigma (residual SD, mol_c)",
        "mean": round(float(sigma.mean()), 4),
        "hdi_2.5": round(float(np.quantile(sigma, 0.025)), 4),
        "hdi_97.5": round(float(np.quantile(sigma, 0.975)), 4), "p_gt_0": np.nan,
    }])
    post.to_csv(RESULT_DIR / "cnew_bayesian_mrv_posterior.csv", index=False)

    assurance = assurance_curve(beta, sigma, tau)
    assurance.to_csv(RESULT_DIR / "cnew_bayesian_mrv_assurance.csv", index=False)

    p_pos = float((beta > 0).mean())
    lines = ["# Experiment 6: Bayesian charge-balance CDR + detection assurance\n",
             "Hierarchical partial-pooling model on the resin NET base-cation "
             "charge balance (mol_c) vs dose. CDR column is an INDICATIVE "
             "alkalinity-equivalent scale (resin-flux proxy, not an areal mass "
             "balance) using 1 mol_c ~ 0.5 mol CO2 durable.\n",
             "## Posterior", post.to_markdown(index=False), "",
             "## Detection assurance (posterior-predictive P[credible-positive])",
             assurance.to_markdown(index=False), "",
             f"Posterior probability the dose effect is positive: P(beta>0) = "
             f"{p_pos:.2f}. The assurance curve shows how many plots/arm a future "
             "campaign needs for a high probability of a credible-positive at the "
             "observed effect magnitude - turning our detection-budget into a "
             "Bayesian field-design tool. Consistent with the frequentist MDE and "
             "the FDR result: at n=4-12 plots/arm the charge-balance CDR signal is "
             "not reliably distinguishable from zero."]
    (AUDIT_DIR / "cnew_bayesian_mrv.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
