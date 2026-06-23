PY ?= ../.venv/bin/python

.PHONY: help smoke data validate audit reconcile \
        snr empirical theory_vs_resin sensor_resin power did chamber \
        bayesian changepoint geostat massbalance porewater events multitask \
        headline cnew_cec cnew_alk cnew_budget cnew_deep cnew_fusion \
        cnew_lb fdr negctrl cnew_mech rigor \
        conformal bayes_mrv mrv_readout sampling cov_ts \
        pathway_budget cdr_seasonal feedstock drainage lag_sig event_pulse \
        multielement residual_ec spillover gas_cobenefit synth \
        forward_model hier_bayes convergence paper_figures \
        ladder cnew all test clean-cache

help:
	@echo "erw pipeline"
	@echo "  smoke           imports + config sanity"
	@echo "  data            build all parquet caches"
	@echo "  validate        validate raw extraction (depth map, skipped configs, weather cutoff)"
	@echo "  audit           render data audit markdown"
	@echo "  reconcile       reconciliation ledger vs erw_ml / erw_mrv"
	@echo "  -- verified analysis ladder --"
	@echo "  snr empirical theory_vs_resin sensor_resin power did chamber"
	@echo "  bayesian changepoint geostat massbalance porewater events multitask"
	@echo "  headline        regenerate test-backed headline_summary.csv"
	@echo "  -- new contributions --"
	@echo "  cnew_cec cnew_alk cnew_budget cnew_deep cnew_fusion"
	@echo "  -- reviewer-rigor add-ons --"
	@echo "  fdr             multiple-comparison (BH-FDR) correction"
	@echo "  negctrl         negative-control contrast figure"
	@echo "  cnew_mech       CDR-lag water-balance mechanism"
	@echo "  cnew_lb         robust deep-TS leaderboard (bootstrap CIs)"
	@echo "  conformal       conformal prediction intervals + coverage"
	@echo "  bayes_mrv       Bayesian CDR + probability-of-detection design"
	@echo "  mrv_readout     charge-balance vs single-ion detectability"
	@echo "  sampling        two-stage sampling design (plots vs capsules)"
	@echo "  cov_ts          covariate/multivariate Chronos-2 (GPU only)"
	@echo "  pathway_budget  aqueous-vs-solid-phase detection budget (C-new-11)"
	@echo "  cdr_seasonal    seasonal CDR-lag persistence/relaxation (C-new-10)"
	@echo "  feedstock       feedstock dissolution Ca:Mg fingerprint (C-new-12)"
	@echo "  drainage        matric-potential flux-switch mechanism (C-new-13)"
	@echo "  lag_sig         randomization test of CDR-lag fingerprint (C-new-14)"
	@echo "  event_pulse     event-based EC-pulse detection test (C-new-15)"
	@echo "  multielement    multi-element geochemical fingerprint (C-new-16)"
	@echo "  residual_ec     physics-removed mobilisation-slope EC detection (C-new-17)"
	@echo "  spillover       cross-plot contamination / spatial spillover (C-new-18)"
	@echo "  gas_cobenefit   multi-gas N2O/CH4 co-benefit screen (C-new-19)"
	@echo "  synth           evidence-synthesis detection posterior (C-new-20)"
	@echo "  forward_model   reactive-transport predicted-vs-observed (C-new-21)"
	@echo "  hier_bayes      hierarchical partial-pooling depth x season (C-new-22)"
	@echo "  convergence     cross-phase detection-wall figure (C-new-23)"
	@echo "  paper_figures   build + copy manuscript figures"
	@echo "  rigor           run all CPU reviewer-rigor add-ons"
	@echo "  ladder          run all verified ladder phases"
	@echo "  cnew            run all CPU new contributions"
	@echo "  test            pytest (golden-master + extraction)"

smoke:           ; $(PY) scripts/00_smoke.py
data:            ; $(PY) scripts/01_build_caches.py
validate:        ; $(PY) scripts/02_validate_extraction.py
audit:           ; $(PY) scripts/03_data_audit.py
reconcile:       ; $(PY) scripts/50_reconciliation_ledger.py

snr:             ; $(PY) scripts/10_snr.py
empirical:       ; $(PY) scripts/11_empirical_effect.py
theory_vs_resin: ; $(PY) scripts/12_theory_vs_resin.py
sensor_resin:    ; $(PY) scripts/13_sensor_to_resin.py
power:           ; $(PY) scripts/14_power.py
did:             ; $(PY) scripts/15_did_synthetic_control.py
chamber:         ; $(PY) scripts/16_chamber_standalone.py
bayesian:        ; $(PY) scripts/17_bayesian.py
changepoint:     ; $(PY) scripts/18_changepoint.py
geostat:         ; $(PY) scripts/19_geostat.py
massbalance:     ; $(PY) scripts/20_mass_balance.py
porewater:       ; $(PY) scripts/21_porewater_ec.py
events:          ; $(PY) scripts/22_weather_events.py
multitask:       ; $(PY) scripts/23_multitask.py
headline:        ; $(PY) scripts/51_build_headline.py

cnew_cec:        ; $(PY) scripts/60_cnew_cec_lag.py
cnew_alk:        ; $(PY) scripts/61_cnew_alkalinity.py
cnew_budget:     ; $(PY) scripts/62_cnew_detection_budget.py
cnew_deep:       ; $(PY) scripts/63_cnew_deep_ts.py
cnew_fusion:     ; $(PY) scripts/64_cnew_fusion.py
cnew_lb:         ; $(PY) scripts/65_deep_ts_leaderboard.py

fdr:             ; $(PY) scripts/70_fdr_correction.py
negctrl:         ; $(PY) scripts/71_negative_controls.py
cnew_mech:       ; $(PY) scripts/72_cec_lag_mechanism.py
conformal:       ; $(PY) scripts/73_conformal.py
cov_ts:          ; $(PY) scripts/74_cnew_covariate_ts.py
bayes_mrv:       ; $(PY) scripts/75_cnew_bayesian_mrv.py
mrv_readout:     ; $(PY) scripts/76_cnew_mrv_readout.py
sampling:        ; $(PY) scripts/77_cnew_sampling_design.py
pathway_budget:  ; $(PY) scripts/80_cnew_pathway_detection_budget.py
cdr_seasonal:    ; $(PY) scripts/81_cnew_cdr_lag_seasonal.py
feedstock:       ; $(PY) scripts/82_cnew_feedstock_fingerprint.py
drainage:        ; $(PY) scripts/83_cnew_drainage_mechanism.py
lag_sig:         ; $(PY) scripts/84_cnew_cdr_lag_significance.py
event_pulse:     ; $(PY) scripts/85_cnew_event_pulse_detection.py
multielement:    ; $(PY) scripts/86_cnew_multielement_fingerprint.py
residual_ec:     ; $(PY) scripts/87_cnew_residual_ec_detection.py
spillover:       ; $(PY) scripts/88_cnew_spatial_spillover.py
gas_cobenefit:   ; $(PY) scripts/89_cnew_gas_cobenefit.py
synth:           ; $(PY) scripts/90_cnew_evidence_synthesis.py  # needs lag_sig residual_ec gas_cobenefit CSVs
forward_model:   ; $(PY) scripts/92_cnew_forward_model.py
hier_bayes:      ; $(PY) scripts/93_cnew_hier_bayes.py
convergence:     ; $(PY) scripts/94_cnew_convergence_wall.py
paper_figures:   ; $(PY) scripts/91_paper_figures.py && cp outputs/figures/fig_feedstock_fingerprint.png outputs/figures/fig_multielement_fingerprint.png outputs/figures/fig_evidence_synthesis.png outputs/figures/fig_forward_model.png outputs/figures/fig_hier_bayes.png outputs/figures/fig_convergence_wall.png paper/figures/  # needs feedstock multielement synth forward_model hier_bayes convergence CSVs/figs

ladder: snr empirical theory_vs_resin sensor_resin power did chamber \
        bayesian changepoint geostat massbalance porewater events multitask headline

cnew: cnew_cec cnew_alk cnew_budget cnew_fusion

rigor: fdr negctrl cnew_mech cnew_lb conformal bayes_mrv mrv_readout sampling \
       cdr_seasonal pathway_budget feedstock drainage lag_sig event_pulse \
       multielement residual_ec spillover gas_cobenefit synth \
       forward_model hier_bayes convergence paper_figures

all: data validate audit ladder cnew rigor reconcile

test:
	$(PY) -m pytest tests/ -q

clean-cache:
	rm -f outputs/cache/*.parquet outputs/cache/*.json
