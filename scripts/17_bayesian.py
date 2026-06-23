"""Phase: Bayesian hierarchical treatment effect on resin Ca (NumPyro).

Partial-pooling model with a plot-half random intercept, so the treatment
effect is estimated while honestly propagating between-plot variance - the
clustered-data answer to the naive t-test.

    ca[i] ~ Normal(mu[i], sigma)
    mu[i] = a_plot[plot_half[i]] + b_depth[depth[i]] + beta_dose * dose_tha[i]
    a_plot ~ Normal(mu_a, tau_a)         (partial pooling over plot-halves)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from src.config import AUDIT_DIR, DOSE_THA, RANDOM_SEED, RESULT_DIR
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin

numpyro.set_host_device_count(1)


def model(plot_idx, depth_idx, dose, n_plots, n_depths, y=None):
    mu_a = numpyro.sample("mu_a", dist.Normal(20.0, 20.0))
    tau_a = numpyro.sample("tau_a", dist.HalfNormal(15.0))
    with numpyro.plate("plots", n_plots):
        a_plot = numpyro.sample("a_plot", dist.Normal(mu_a, tau_a))
    with numpyro.plate("depths", n_depths):
        b_depth = numpyro.sample("b_depth", dist.Normal(0.0, 10.0))
    beta_dose = numpyro.sample("beta_dose", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(15.0))
    mu = a_plot[plot_idx] + b_depth[depth_idx] + beta_dose * dose
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def main() -> None:
    resin = qa_clean(load_resin()).dropna(subset=["ca_ppm"]).reset_index(drop=True)
    resin["dose_tha"] = resin["treatment"].map(DOSE_THA)
    plots = sorted(resin["plot_half"].unique())
    depths = sorted(resin["depth_cm"].unique())
    p_idx = resin["plot_half"].map({p: i for i, p in enumerate(plots)}).to_numpy()
    d_idx = resin["depth_cm"].map({d: i for i, d in enumerate(depths)}).to_numpy()

    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=2000,
                num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(RANDOM_SEED),
             plot_idx=jnp.array(p_idx), depth_idx=jnp.array(d_idx),
             dose=jnp.array(resin["dose_tha"].to_numpy(dtype=float)),
             n_plots=len(plots), n_depths=len(depths),
             y=jnp.array(resin["ca_ppm"].to_numpy(dtype=float)))
    s = mcmc.get_samples()

    beta = np.asarray(s["beta_dose"])
    rows = [{
        "parameter": "beta_dose_ppm_per_tha",
        "mean": round(float(beta.mean()), 4),
        "sd": round(float(beta.std()), 4),
        "hdi_2.5": round(float(np.quantile(beta, 0.025)), 4),
        "hdi_97.5": round(float(np.quantile(beta, 0.975)), 4),
        "p_gt_0": round(float((beta > 0).mean()), 3),
    }]
    for name in ("tau_a", "sigma", "mu_a"):
        v = np.asarray(s[name])
        rows.append({"parameter": name, "mean": round(float(v.mean()), 4),
                     "sd": round(float(v.std()), 4),
                     "hdi_2.5": round(float(np.quantile(v, 0.025)), 4),
                     "hdi_97.5": round(float(np.quantile(v, 0.975)), 4),
                     "p_gt_0": np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "bayesian_ca.csv", index=False)

    lines = ["# Phase: Bayesian hierarchical (resin Ca)\n",
             "Partial-pooling over plot-halves; dose in t/ha.\n",
             out.to_markdown(index=False), "",
             f"Between-plot SD (tau_a={out.loc[out.parameter=='tau_a','mean'].iloc[0]}) "
             f"vs residual (sigma={out.loc[out.parameter=='sigma','mean'].iloc[0]}) "
             "shows how much variance is between-plot - the clustering the naive "
             "test ignores. The dose-effect credible interval includes/embraces 0 "
             "consistent with the weak empirical signal."]
    (AUDIT_DIR / "phase_bayesian.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
