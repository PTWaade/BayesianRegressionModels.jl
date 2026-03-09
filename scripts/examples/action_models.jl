using CSV, DataFrames, Downloads, Random

# Formulas from: ActionModels.jl (ComputationalPsychiatry)
# Source: https://github.com/ComputationalPsychiatry/ActionModels.jl
# Data:   https://github.com/ComputationalPsychiatry/ActionModels.jl/tree/main/docs/example_data
#
# ActionModels uses lme4/brms-style @formula syntax for the population-level regression
# that maps cognitive parameters (learning rates, noise) onto experimental covariates.
# These formulas do NOT conform to brms syntax — the dependent variable is a latent
# cognitive parameter estimated by fitting a cognitive model (e.g. Rescorla-Wagner or
# PVL-Delta) to trial-level behavioural data.

const ACTIONMODELS_DATA_URL = "https://raw.githubusercontent.com/ComputationalPsychiatry/ActionModels.jl/main/docs/example_data/"

# ── Iowa Gambling Task — Ahn et al. (2014) ────────────────────────────────────

"""
name: ahn_igt — Iowa Gambling Task (Ahn et al. 2014)
source: https://github.com/ComputationalPsychiatry/ActionModels.jl/tree/main/docs/example_data/ahn_et_al_2014
----

Decision-making data from Ahn et al. (2014, *Frontiers in Psychology*): 129 participants
across three clinical groups performing the Iowa Gambling Task (100 trials each). Columns:
`trial` (1–100), `deck` (1–4, decks A–D), `gain` (monetary gain per trial), `loss`
(monetary loss per trial, ≤ 0), `reward` (= gain + loss, net outcome), `subjID`
(subject ID), `clinical_group`
("healthy_control" N=48, "heroin" N=43, "amphetamine" N=38).

Reference: Ahn W-Y et al. (2014). Decision-making in stimulant and opiate addicts in
protracted abstinence: evidence from computational modeling with pure users.
"""
function load(::Val{:ahn_igt})
    groups = [
        ("IGTdata_healthy_control.txt", "healthy_control"),
        ("IGTdata_heroin.txt",           "heroin"),
        ("IGTdata_amphetamine.txt",      "amphetamine"),
    ]
    dfs = map(groups) do (fname, gname)
        url = ACTIONMODELS_DATA_URL * "ahn_et_al_2014/" * fname
        df  = CSV.read(Downloads.download(url), DataFrame; delim = '\t')
        df.reward         = df.gain .+ df.loss   # loss is negative; reward = net outcome
        df.clinical_group .= gname
        df
    end
    return vcat(dfs...)
end

# ── JGET — Mikus et al. (2025) ────────────────────────────────────────────────

"""
name: jget_trial — Jumping Gaussian Estimation Task, Trial-Level Data
source: https://github.com/ComputationalPsychiatry/ActionModels.jl/tree/main/docs/example_data/JGET
----

Trial-level data from Mikus et al. (2025) on schizotypy and reinforcement learning.
Columns: `trials` (trial number), `ID` (participant), `session` (1–3), `outcome`
(integer target value), `response` (float continuous estimate), `confidence` (float ∈ [0,1]).
Input to ActionModels for fitting the Rescorla-Wagner model.
"""
load(::Val{:jget_trial}) = CSV.read(
    Downloads.download(ACTIONMODELS_DATA_URL * "JGET/JGET_data_trial_preprocessed.csv"),
    DataFrame,
)

"""
name: jget_sub — Jumping Gaussian Estimation Task, Subject-Level Data
source: https://github.com/ComputationalPsychiatry/ActionModels.jl/tree/main/docs/example_data/JGET
----

Subject-level (per session) data from Mikus et al. (2025). Key columns: `ID`, `session`
(1–3), `pdi_total` (Peters Delusions Inventory normalised total score), `Age`, `Gender`,
`Education`. Used as regression covariates to predict Rescorla-Wagner parameters estimated
from trial-level data.
"""
load(::Val{:jget_sub}) = CSV.read(
    Downloads.download(ACTIONMODELS_DATA_URL * "JGET/JGET_data_sub_preprocessed.csv"),
    DataFrame,
)

# ── Rescorla-Wagner × JGET ────────────────────────────────────────────────────

"""
name: Rescorla-Wagner × JGET — Learning Rate vs. Delusional Ideation
source: https://github.com/ComputationalPsychiatry/ActionModels.jl/blob/main/docs/julia_files/D_tutorials_examples/example_jget.jl
example: rw_jget
dataset: jget_sub
formula: "learning_rate ~ 1 + pdi_total + session + (1 | ID)"
----

Population-level hierarchical regression of the Rescorla-Wagner learning rate (α)
on Peters Delusions Inventory (PDI) total score and session number. `(1|ID)` captures
between-subject variability. α is constrained to [0,1] via logistic transformation;
regression prior: Normal(0, 0.5) on PDI coefficient.

Note: `learning_rate` is a latent parameter estimated by running ActionModels on the
trial-level JGET data before this regression is applied.
"""
function examples(::Val{:rw_jget_lr})
    data = load(Val(:jget_sub))
    return ("learning_rate ~ 1 + pdi_total + session + (1 | ID)", data)
end

"""
name: Rescorla-Wagner × JGET — Action Noise vs. Delusional Ideation
source: https://github.com/ComputationalPsychiatry/ActionModels.jl/blob/main/docs/julia_files/D_tutorials_examples/example_jget.jl
example: rw_jget
dataset: jget_sub
formula: "action_noise ~ 1 + pdi_total + session + (1 | ID)"
----

Population-level hierarchical regression of the Rescorla-Wagner action noise (β, response
standard deviation) on PDI score and session. `(1|ID)` captures subject-level variability.
β is unconstrained (exponential link); prior: Normal(0, 0.3) on intercept.
"""
function examples(::Val{:rw_jget_noise})
    data = load(Val(:jget_sub))
    return ("action_noise ~ 1 + pdi_total + session + (1 | ID)", data)
end

# ── PVL-Delta × Iowa Gambling Task ────────────────────────────────────────────

"""
name: PVL-Delta × IGT — Learning Rate vs. Clinical Group
source: https://github.com/ComputationalPsychiatry/ActionModels.jl/blob/main/docs/julia_files/C_premade_models/pvl_delta.jl
example: pvl_igt
dataset: ahn_igt
formula: "learning_rate ~ clinical_group + (1 | subjID)"
----

Population-level hierarchical regression of the PVL-Delta learning rate (α ∈ [0,1])
on clinical group (healthy controls, heroin, amphetamine) in the Iowa Gambling Task.
`(1|subjID)` models within-subject random intercepts across trials. Transformed via
logistic link; LogitNormal prior on regression coefficients.
"""
function examples(::Val{:pvl_igt_lr})
    data = load(Val(:ahn_igt))
    return ("learning_rate ~ clinical_group + (1 | subjID)", data)
end

"""
name: PVL-Delta × IGT — Reward Sensitivity vs. Clinical Group
source: https://github.com/ComputationalPsychiatry/ActionModels.jl/blob/main/docs/julia_files/C_premade_models/pvl_delta.jl
example: pvl_igt
dataset: ahn_igt
formula: "reward_sensitivity ~ clinical_group + (1 | subjID)"
----

Population-level hierarchical regression of prospect-theoretic reward sensitivity
(A ∈ [0,1], power parameter in utility transformation `u = |r|^A`) on clinical group
in the Iowa Gambling Task. LogitNormal prior; logistic link.
"""
function examples(::Val{:pvl_igt_reward})
    data = load(Val(:ahn_igt))
    return ("reward_sensitivity ~ clinical_group + (1 | subjID)", data)
end

"""
name: PVL-Delta × IGT — Loss Aversion vs. Clinical Group
source: https://github.com/ComputationalPsychiatry/ActionModels.jl/blob/main/docs/julia_files/C_premade_models/pvl_delta.jl
example: pvl_igt
dataset: ahn_igt
formula: "loss_aversion ~ clinical_group + (1 | subjID)"
----

Population-level hierarchical regression of loss aversion weight (w ∈ [0,∞], scales
negative prospect values as `-w·|loss|^A`) on clinical group in the Iowa Gambling Task.
Exponential link; LogNormal prior on regression coefficients.
"""
function examples(::Val{:pvl_igt_loss})
    data = load(Val(:ahn_igt))
    return ("loss_aversion ~ clinical_group + (1 | subjID)", data)
end

"""
name: PVL-Delta × IGT — Action Noise vs. Clinical Group
source: https://github.com/ComputationalPsychiatry/ActionModels.jl/blob/main/docs/julia_files/C_premade_models/pvl_delta.jl
example: pvl_igt
dataset: ahn_igt
formula: "action_noise ~ clinical_group + (1 | subjID)"
----

Population-level hierarchical regression of the softmax inverse temperature (β ∈ [0,∞];
higher → more deterministic deck choice) on clinical group in the Iowa Gambling Task.
Exponential link; LogNormal prior on regression coefficients.
"""
function examples(::Val{:pvl_igt_noise})
    data = load(Val(:ahn_igt))
    return ("action_noise ~ clinical_group + (1 | subjID)", data)
end
