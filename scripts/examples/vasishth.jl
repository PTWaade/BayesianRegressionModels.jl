using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: "An Introduction to Bayesian Data Analysis for Cognitive Science"
# Bruno Nicenboim, Daniel Schad, Shravan Vasishth
# Online book: https://bruno.nicenboim.me/bayescogsci/
# Source code: https://github.com/bnicenboim/bayescogsci
# Data package: https://github.com/bnicenboim/bcogsci (R package `bcogsci`)
#
# Experiments covered (chs. 4–7, 11):
#   Ch 4: Pupil dilation, Button-press RTs, Working memory recall
#   Ch 5: N400 EEG, Stroop task, Hierarchical pooling illustration
#   Ch 6: Contrast coding (single factor)
#   Ch 7: Contrast coding (2 × 2 design)
#   Ch 11: Meta-analysis, Measurement-error individual differences
#
# All datasets belong to the `bcogsci` R package. Public source files are on OSF;
# synthetic equivalents are generated below to preserve the model formulas.
# Original data:
#   Pupil:    https://osf.io/z43dz/?action=download
#   Recall:   https://osf.io/6r9ka/?action=download
#   N400 EEG: https://osf.io/q7dsk/?action=download
#   Stroop:   https://osf.io/n8xa7/?action=download
#   Meta-SBI: https://osf.io/du3qp/?action=download

##############################################################################
# Ch 4: Pupil dilation experiment
# Cognitive load effect on pupil dilation during a working memory task.
# Each participant memorised 1–6 letters while their pupil size was recorded.
#
# Columns: p_size (pupil diameter in pixels), c_load (centered set size 1–6).
#
# brms model formula:
#   "p_size ~ 1 + c_load"
#     (Gaussian; higher cognitive load → larger pupil dilation)
##############################################################################

function load(::Val{:pupil})
    rng    = MersenneTwister(41)
    n      = 150
    load   = rand(rng, 1:6, n)
    c_load = float.(load) .- mean(float.(load))
    p_size = 1000.0 .+ 80.0 .* c_load .+ 200.0 .* randn(rng, n)
    return DataFrame(; p_size, c_load)
end

function examples(::Val{:pupil})
    d = load(Val(:pupil))
    return [("p_size ~ 1 + c_load", d)]
end

##############################################################################
# Ch 4: Spacebar-pressing response times — practice effects
# Participants press a spacebar as fast as possible across many trials.
# Response times decrease with practice (log-normal distribution).
#
# Columns: t (response time in seconds), c_trial (centered trial number).
#
# brms model formula:
#   "t ~ 1 + c_trial"
#     (lognormal; practice effect: RT decreases linearly on log scale)
##############################################################################

function load(::Val{:spacebar})
    rng     = MersenneTwister(42)
    n       = 300
    trial   = 1:n
    c_trial = float.(trial) .- mean(float.(trial))
    t       = exp.(6.0 .- 0.005 .* c_trial .+ 0.3 .* randn(rng, n))
    return DataFrame(; t, c_trial)
end

function examples(::Val{:spacebar})
    d = load(Val(:spacebar))
    return [("t ~ 1 + c_trial", d)]
end

##############################################################################
# Ch 4: Working memory recall accuracy — set size effect
# Participants recalled letters from sets of size 2–7.
# Accuracy decreases as set size increases.
#
# Columns: correct (0/1), c_set_size (centered set size).
#
# brms model formula:
#   "correct ~ 1 + c_set_size"
#     (bernoulli(logit); set-size effect on accuracy)
##############################################################################

function load(::Val{:recall_wm})
    rng       = MersenneTwister(43)
    n         = 400
    set_size  = rand(rng, 2:7, n)
    c_set_size = float.(set_size) .- mean(float.(set_size))
    p_correct  = 1.0 ./ (1.0 .+ exp.(-(1.5 .- 0.5 .* c_set_size)))
    correct    = [rand(rng) < p ? 1 : 0 for p in p_correct]
    return DataFrame(; correct, c_set_size)
end

function examples(::Val{:recall_wm})
    d = load(Val(:recall_wm))
    return [("correct ~ 1 + c_set_size", d)]
end

##############################################################################
# Ch 5: N400 EEG — neural response to word predictability
# Participants read sentences; N400 amplitude measured at each word.
# N400 is more negative for low-cloze (unexpected) words.
# c_cloze = centered cloze probability (0 = most predictable context).
#
# Columns: n400 (µV, mean-amplitude in 300–500 ms window), c_cloze,
#          subj (participant ID), item (stimulus ID).
#
# brms model formulas:
#   "n400 ~ c_cloze + (c_cloze || subj)"
#     (ch5: varying intercepts and slopes, uncorrelated; || suppresses correlation)
#   "n400 ~ c_cloze + (c_cloze | subj)"
#     (ch5: varying intercepts and slopes, correlated)
#   "n400 ~ c_cloze + (c_cloze | subj) + (c_cloze | item)"
#     (ch5: crossed random effects for subjects and items)
#   "bf(n400 ~ c_cloze + (c_cloze | subj) + (c_cloze | item), sigma ~ 1 + (1 | subj))"
#     (ch5: distributional model; sigma varies by subject)
##############################################################################

function load(::Val{:n400})
    rng      = MersenneTwister(400)
    n_subj   = 40
    n_items  = 80
    u_subj   = randn(rng, n_subj) .* 50.0   # subject intercepts
    u_item   = randn(rng, n_items) .* 30.0   # item intercepts
    rows     = NamedTuple[]
    for s in 1:n_subj, i in 1:n_items
        cloze  = rand(rng) * 2 - 1           # uniform in [-1, 1]
        n400   = -80.0 + 60.0 * cloze + u_subj[s] + u_item[i] + 40.0 * randn(rng)
        push!(rows, (; n400, c_cloze = cloze, subj = string(s), item = string(i)))
    end
    return DataFrame(rows)
end

function examples(::Val{:n400})
    d = load(Val(:n400))
    return [
        ("n400 ~ c_cloze + (c_cloze || subj)", d),
        ("n400 ~ c_cloze + (c_cloze | subj)", d),
        ("n400 ~ c_cloze + (c_cloze | subj) + (c_cloze | item)", d),
        ("bf(n400 ~ c_cloze + (c_cloze | subj) + (c_cloze | item), sigma ~ 1 + (1 | subj))", d),
    ]
end

##############################################################################
# Ch 5: Stroop task — congruence effect on response time
# Participants name ink colors of congruent vs. incongruent color words.
# Incongruent trials are slower (Stroop effect).
# c_cond: +0.5 = incongruent, -0.5 = congruent (sum contrast coding).
#
# Columns: RT (response time in ms), c_cond, subj (participant ID).
#
# brms model formula:
#   "RT ~ c_cond + (c_cond | subj)"
#     (lognormal; varying intercepts and correlated slopes; Stroop effect per subject)
##############################################################################

function load(::Val{:stroop})
    rng    = MersenneTwister(99)
    n_subj = 50
    u_int  = randn(rng, n_subj) .* 0.15     # subject intercepts (log scale)
    u_slp  = randn(rng, n_subj) .* 0.08     # subject slopes
    rows   = NamedTuple[]
    for s in 1:n_subj, _ in 1:40
        c_cond = rand(rng) < 0.5 ? -0.5 : 0.5
        log_RT = 6.3 + 0.1 * c_cond + u_int[s] + u_slp[s] * c_cond + 0.1 * randn(rng)
        push!(rows, (; RT = exp(log_RT), c_cond, subj = string(s)))
    end
    return DataFrame(rows)
end

function examples(::Val{:stroop})
    d = load(Val(:stroop))
    return [("RT ~ c_cond + (c_cond | subj)", d)]
end

##############################################################################
# Ch 5: Hierarchical pooling illustration (toy data)
# Three minimal datasets used to contrast complete pooling, no pooling,
# and partial pooling (hierarchical) approaches.
#
# Columns: y (outcome), subj (participant ID).
#
# brms model formulas:
#   "y ~ 1 + (1 | subj)"
#     (hierarchical / partial pooling; adaptive shrinkage across subjects)
#   "y ~ 1"
#     (complete pooling; single intercept)
#   "y ~ 0 + factor(subj)"
#     (no pooling; one intercept per subject; cell-means parameterization)
##############################################################################

function load(::Val{:pooling})
    rng    = MersenneTwister(5)
    n_subj = 10
    n_obs  = 5
    u      = randn(rng, n_subj) .* 1.5
    rows   = NamedTuple[]
    for s in 1:n_subj, _ in 1:n_obs
        push!(rows, (; y = 500.0 + 50.0 * u[s] + 20.0 * randn(rng),
                       subj = string(s)))
    end
    return DataFrame(rows)
end

function examples(::Val{:pooling})
    d = load(Val(:pooling))
    return [
        ("y ~ 1 + (1 | subj)", d),
        ("y ~ 1", d),
        ("y ~ 0 + factor(subj)", d),
    ]
end

##############################################################################
# Ch 6: Contrast coding — single factor with multiple levels
# Synthetic data illustrating treatment, sum, Helmert, and other contrast schemes.
# Factor F has 4 levels; DV is a normally distributed outcome.
#
# brms model formulas:
#   "DV ~ F"
#     (factor as predictor; treatment contrast by default; intercept = grand mean)
#   "DV ~ -1 + F"
#     (cell-means parameterization; no global intercept)
#   "DV ~ 1 + mo(F)"
#     (monotonic effect; F treated as ordered categorical)
##############################################################################

function load(::Val{:contrasts1})
    rng    = MersenneTwister(6)
    n      = 120
    levels = ["A", "B", "C", "D"]
    means  = [50.0, 55.0, 60.0, 65.0]
    F      = rand(rng, levels, n)
    DV     = [means[findfirst(==(f), levels)] + 10.0 * randn(rng) for f in F]
    return DataFrame(; DV, F)
end

function examples(::Val{:contrasts1})
    d = load(Val(:contrasts1))
    return [
        ("DV ~ F", d),
        ("DV ~ -1 + F", d),
        ("DV ~ 1 + mo(F)", d),
    ]
end

##############################################################################
# Ch 7: Contrast coding — 2 × 2 factorial design
# Synthetic data with factors A (2 levels) and B (2 levels).
# Used to illustrate main effects, interaction, and nested contrasts.
#
# brms model formulas:
#   "DV ~ A * B"
#     (full factorial: main effects of A and B plus A × B interaction)
#   "DV ~ B / A"
#     (B nested in A; A effects estimated separately within each level of B)
#   "pDV ~ A * B"
#     (bernoulli(logit); logistic regression with 2 × 2 design)
##############################################################################

function load(::Val{:contrasts2x2})
    rng = MersenneTwister(7)
    n   = 200
    A   = rand(rng, ["a1", "a2"], n)
    B   = rand(rng, ["b1", "b2"], n)
    DV  = [50.0 + (a == "a2" ? 5.0 : 0.0) + (b == "b2" ? 3.0 : 0.0) +
           (a == "a2" && b == "b2" ? -4.0 : 0.0) + 10.0 * randn(rng)
           for (a, b) in zip(A, B)]
    p   = 1.0 ./ (1.0 .+ exp.(-(DV .- 50) ./ 20))
    pDV = [rand(rng) < pi ? 1 : 0 for pi in p]
    return DataFrame(; DV, A, B, pDV)
end

function examples(::Val{:contrasts2x2})
    d = load(Val(:contrasts2x2))
    return [
        ("DV ~ A * B", d),
        ("DV ~ B / A", d),
        ("pDV ~ A * B", d),
    ]
end

##############################################################################
# Ch 11: Meta-analysis — stroke-brain injury effect on language
# Bayesian random-effects meta-analysis; effect sizes with known standard errors.
# Each row is a study; effect = standardized mean difference; SE = its standard error.
#
# Data source: OSF https://osf.io/du3qp/?action=download
# Synthetic equivalent generated below.
#
# brms model formula:
#   "effect | resp_se(SE, sigma = FALSE) ~ 1 + (1 | study_id)"
#     (resp_se passes known measurement error; sigma=FALSE fixes residual SD to 0;
#      random study intercept captures between-study heterogeneity τ)
##############################################################################

function load(::Val{:meta_sbi})
    rng      = MersenneTwister(11)
    n_studies = 20
    true_mu  = 0.5         # pooled effect
    tau      = 0.3         # between-study SD
    theta    = true_mu .+ tau .* randn(rng, n_studies)    # true study effects
    n_i      = rand(rng, 10:80, n_studies)                # sample sizes
    SE       = 1.0 ./ sqrt.(float.(n_i))
    effect   = theta .+ SE .* randn(rng, n_studies)
    return DataFrame(;
        study_id = ["study_" * string(i) for i in 1:n_studies],
        effect,
        SE,
    )
end

function examples(::Val{:meta_sbi})
    d = load(Val(:meta_sbi))
    return [
        ("effect | resp_se(SE, sigma = FALSE) ~ 1 + (1 | study_id)", d),
    ]
end

##############################################################################
# Ch 11: Individual differences — reading speed and prior context use
# Participants completed a reading experiment and a test of prior context use (PCU).
# Both measurements have uncertainty (standard errors).
#
# Data source: OSF https://osf.io/du3qp/?action=download (part of same dataset)
# Synthetic equivalent generated below.
#
# Columns: mean_rspeed (mean reading speed), se_rspeed (SE of reading speed),
#          c_mean_pcu (centered mean PCU score), se_pcu (SE of PCU score).
#
# brms model formulas:
#   "mean_rspeed ~ c_mean_pcu"
#     (naive OLS ignoring measurement error in both variables)
#   "mean_rspeed | resp_se(se_rspeed, sigma = TRUE) ~ me(c_mean_pcu, se_pcu)"
#     (measurement error in both response and predictor;
#      me() propagates uncertainty in the predictor;
#      resp_se() propagates uncertainty in the response and adds residual sigma)
##############################################################################

function load(::Val{:indiv_diff})
    rng     = MersenneTwister(110)
    n       = 60
    pcu_true = randn(rng, n) .* 0.5
    rs_true  = 300.0 .+ 30.0 .* pcu_true .+ 40.0 .* randn(rng, n)
    se_pcu   = 0.05 .+ 0.05 .* rand(rng, n)
    se_rs    = 15.0 .+ 10.0 .* rand(rng, n)
    c_mean_pcu  = pcu_true .+ se_pcu .* randn(rng, n)
    mean_rspeed = rs_true  .+ se_rs  .* randn(rng, n)
    c_mean_pcu  = c_mean_pcu .- mean(c_mean_pcu)
    return DataFrame(; mean_rspeed, se_rspeed = se_rs, c_mean_pcu, se_pcu)
end

function examples(::Val{:indiv_diff})
    d = load(Val(:indiv_diff))
    return [
        ("mean_rspeed ~ c_mean_pcu", d),
        ("mean_rspeed | resp_se(se_rspeed, sigma = TRUE) ~ me(c_mean_pcu, se_pcu)", d),
    ]
end
