using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: "An Introduction to Bayesian Data Analysis for Cognitive Science"
# Bruno Nicenboim, Daniel Schad, Shravan Vasishth
# Online book: https://bruno.nicenboim.me/bayescogsci/
# Source code: https://github.com/bnicenboim/bayescogsci
# Data package: https://github.com/bnicenboim/bcogsci (R package `bcogsci`)

"""
name: pupil — Synthetic Pupil Dilation Experiment Data
source: https://osf.io/z43dz/?action=download
chapter: Ch 4
----

Cognitive load effect on pupil dilation during a working memory task. Each participant
memorised 1–6 letters while their pupil size was recorded. Synthetic data generated here
to match the structure. 150 obs.

Columns: `p_size` (pupil diameter in pixels), `c_load` (centered set size 1–6).
"""
function load(::Val{:pupil})
    rng    = MersenneTwister(41)
    n      = 150
    load   = rand(rng, 1:6, n)
    c_load = float.(load) .- mean(float.(load))
    p_size = 1000.0 .+ 80.0 .* c_load .+ 200.0 .* randn(rng, n)
    return DataFrame(; p_size, c_load)
end

"""
name: Pupil Dilation Experiment — Gaussian Regression
source: https://bruno.nicenboim.me/bayescogsci/
example: pupil
dataset: pupil
chapter: Ch 4
formula: "p_size ~ 1 + c_load"
----

Gaussian regression; higher cognitive load → larger pupil dilation.
"""
function examples(::Val{:pupil})
    d = load(Val(:pupil))
    return ("p_size ~ 1 + c_load", d)
end

"""
name: spacebar — Synthetic Spacebar-Pressing RT Data
source: synthetic
chapter: Ch 4
----

Participants press a spacebar as fast as possible across many trials. Response times
decrease with practice (log-normal distribution). Synthetic data generated here.
300 obs.

Columns: `t` (response time in seconds), `c_trial` (centered trial number).
"""
function load(::Val{:spacebar})
    rng     = MersenneTwister(42)
    n       = 300
    trial   = 1:n
    c_trial = float.(trial) .- mean(float.(trial))
    t       = exp.(6.0 .- 0.005 .* c_trial .+ 0.3 .* randn(rng, n))
    return DataFrame(; t, c_trial)
end

"""
name: Spacebar-Pressing Response Times — Lognormal Regression
source: https://bruno.nicenboim.me/bayescogsci/
example: spacebar
dataset: spacebar
chapter: Ch 4
formula: "t ~ 1 + c_trial"
----

Lognormal regression; practice effect: RT decreases linearly on log scale.
"""
function examples(::Val{:spacebar})
    d = load(Val(:spacebar))
    return ("t ~ 1 + c_trial", d)
end

"""
name: recall_wm — Synthetic Working Memory Recall Data
source: https://osf.io/6r9ka/?action=download
chapter: Ch 4
----

Participants recalled letters from sets of size 2–7. Accuracy decreases as set size
increases. Synthetic data generated here to match the structure. 400 obs.

Columns: `correct` (0/1), `c_set_size` (centered set size 2–7).
"""
function load(::Val{:recall_wm})
    rng       = MersenneTwister(43)
    n         = 400
    set_size  = rand(rng, 2:7, n)
    c_set_size = float.(set_size) .- mean(float.(set_size))
    p_correct  = 1.0 ./ (1.0 .+ exp.(-(1.5 .- 0.5 .* c_set_size)))
    correct    = [rand(rng) < p ? 1 : 0 for p in p_correct]
    return DataFrame(; correct, c_set_size)
end

"""
name: Working Memory Recall Accuracy — Logistic Regression
source: https://bruno.nicenboim.me/bayescogsci/
example: recall_wm
dataset: recall_wm
chapter: Ch 4
formula: "correct ~ 1 + c_set_size"
----

Bernoulli(logit) regression; set-size effect on recall accuracy.
"""
function examples(::Val{:recall_wm})
    d = load(Val(:recall_wm))
    return ("correct ~ 1 + c_set_size", d)
end

"""
name: n400 — Synthetic N400 EEG Data
source: https://osf.io/q7dsk/?action=download
chapter: Ch 5
----

Participants read sentences; N400 amplitude measured at each word. N400 is more negative
for low-cloze (unexpected) words. Synthetic crossed design: 40 subjects × 80 items.

Columns: `n400` (µV, mean amplitude in 300–500 ms window), `c_cloze` (centered cloze
probability), `subj` (participant ID), `item` (stimulus ID).
"""
function load(::Val{:n400})
    rng      = MersenneTwister(400)
    n_subj   = 40
    n_items  = 80
    u_subj   = randn(rng, n_subj) .* 50.0
    u_item   = randn(rng, n_items) .* 30.0
    rows     = NamedTuple[]
    for s in 1:n_subj, i in 1:n_items
        cloze  = rand(rng) * 2 - 1
        n400   = -80.0 + 60.0 * cloze + u_subj[s] + u_item[i] + 40.0 * randn(rng)
        push!(rows, (; n400, c_cloze = cloze, subj = string(s), item = string(i)))
    end
    return DataFrame(rows)
end

"""
name: N400 EEG — Uncorrelated Varying Slopes
source: https://bruno.nicenboim.me/bayescogsci/
example: n400
dataset: n400
chapter: Ch 5
formula: "n400 ~ c_cloze + (c_cloze || subj)"
----

Varying intercepts and slopes for subjects, uncorrelated; `||` suppresses the
correlation parameter between intercept and slope.
"""
function examples(::Val{:n400_uncorr})
    d = load(Val(:n400))
    return ("n400 ~ c_cloze + (c_cloze || subj)", d)
end

"""
name: N400 EEG — Correlated Varying Slopes
source: https://bruno.nicenboim.me/bayescogsci/
example: n400
dataset: n400
chapter: Ch 5
formula: "n400 ~ c_cloze + (c_cloze | subj)"
----

Varying intercepts and correlated slopes for subjects; `|` estimates the
intercept–slope correlation.
"""
function examples(::Val{:n400_corr})
    d = load(Val(:n400))
    return ("n400 ~ c_cloze + (c_cloze | subj)", d)
end

"""
name: N400 EEG — Crossed Random Effects
source: https://bruno.nicenboim.me/bayescogsci/
example: n400
dataset: n400
chapter: Ch 5
formula: "n400 ~ c_cloze + (c_cloze | subj) + (c_cloze | item)"
----

Crossed random effects for subjects and items; both have varying intercepts and slopes.
"""
function examples(::Val{:n400_crossed})
    d = load(Val(:n400))
    return ("n400 ~ c_cloze + (c_cloze | subj) + (c_cloze | item)", d)
end

"""
name: N400 EEG — Distributional Model for Sigma
source: https://bruno.nicenboim.me/bayescogsci/
example: n400
dataset: n400
chapter: Ch 5
formula: "bf(n400 ~ c_cloze + (c_cloze | subj) + (c_cloze | item), sigma ~ 1 + (1 | subj))"
----

Distributional model: residual sigma modeled with a subject-level random intercept,
allowing between-subject variability in residual spread.
"""
function examples(::Val{:n400_distr})
    d = load(Val(:n400))
    return ("bf(n400 ~ c_cloze + (c_cloze | subj) + (c_cloze | item), sigma ~ 1 + (1 | subj))", d)
end

"""
name: stroop — Synthetic Stroop Task Data
source: https://osf.io/n8xa7/?action=download
chapter: Ch 5
----

Participants name ink colors of congruent vs. incongruent color words. Incongruent trials
are slower (Stroop effect). Synthetic data generated here. 50 subjects × 40 trials.

Columns: `RT` (response time in ms), `c_cond` (+0.5 = incongruent, −0.5 = congruent,
sum contrast coding), `subj` (participant ID).
"""
function load(::Val{:stroop})
    rng    = MersenneTwister(99)
    n_subj = 50
    u_int  = randn(rng, n_subj) .* 0.15
    u_slp  = randn(rng, n_subj) .* 0.08
    rows   = NamedTuple[]
    for s in 1:n_subj, _ in 1:40
        c_cond = rand(rng) < 0.5 ? -0.5 : 0.5
        log_RT = 6.3 + 0.1 * c_cond + u_int[s] + u_slp[s] * c_cond + 0.1 * randn(rng)
        push!(rows, (; RT = exp(log_RT), c_cond, subj = string(s)))
    end
    return DataFrame(rows)
end

"""
name: Stroop Task — Lognormal Varying-Slopes Model
source: https://bruno.nicenboim.me/bayescogsci/
example: stroop
dataset: stroop
chapter: Ch 5
formula: "RT ~ c_cond + (c_cond | subj)"
----

Lognormal regression; varying intercepts and correlated slopes per subject; Stroop
congruence effect estimated per subject.
"""
function examples(::Val{:stroop})
    d = load(Val(:stroop))
    return ("RT ~ c_cond + (c_cond | subj)", d)
end

"""
name: pooling — Synthetic Hierarchical Pooling Illustration Data
source: synthetic
chapter: Ch 5
----

Toy dataset contrasting complete pooling, no pooling, and partial pooling (hierarchical).
10 subjects × 5 observations each.

Columns: `y` (outcome), `subj` (participant ID).
"""
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

"""
name: Hierarchical Pooling — Partial Pooling
source: https://bruno.nicenboim.me/bayescogsci/
example: pooling
dataset: pooling
chapter: Ch 5
formula: "y ~ 1 + (1 | subj)"
----

Hierarchical / partial pooling; adaptive shrinkage across subjects.
"""
function examples(::Val{:pooling_partial})
    d = load(Val(:pooling))
    return ("y ~ 1 + (1 | subj)", d)
end

"""
name: Hierarchical Pooling — Complete Pooling
source: https://bruno.nicenboim.me/bayescogsci/
example: pooling
dataset: pooling
chapter: Ch 5
formula: "y ~ 1"
----

Complete pooling; single intercept ignoring subject identity.
"""
function examples(::Val{:pooling_complete})
    d = load(Val(:pooling))
    return ("y ~ 1", d)
end

"""
name: Hierarchical Pooling — No Pooling
source: https://bruno.nicenboim.me/bayescogsci/
example: pooling
dataset: pooling
chapter: Ch 5
formula: "y ~ 0 + factor(subj)"
----

No pooling; one intercept per subject; cell-means parameterization.
"""
function examples(::Val{:pooling_none})
    d = load(Val(:pooling))
    return ("y ~ 0 + factor(subj)", d)
end

"""
name: contrasts1 — Synthetic Single-Factor Contrast Data
source: synthetic
chapter: Ch 6
----

Synthetic data illustrating treatment, sum, Helmert, and other contrast schemes.
Factor `F` has 4 levels (A–D); `DV` is a normally distributed outcome. 120 obs.

Columns: `DV` (continuous outcome), `F` (factor with levels A, B, C, D).
"""
function load(::Val{:contrasts1})
    rng    = MersenneTwister(6)
    n      = 120
    levels = ["A", "B", "C", "D"]
    means  = [50.0, 55.0, 60.0, 65.0]
    F      = rand(rng, levels, n)
    DV     = [means[findfirst(==(f), levels)] + 10.0 * randn(rng) for f in F]
    return DataFrame(; DV, F)
end

"""
name: Contrast Coding — Treatment Contrast
source: https://bruno.nicenboim.me/bayescogsci/
example: contrasts1
dataset: contrasts1
chapter: Ch 6
formula: "DV ~ F"
----

Factor as predictor; treatment contrast by default (reference level = "A").
"""
function examples(::Val{:contrasts1_treatment})
    d = load(Val(:contrasts1))
    return ("DV ~ F", d)
end

"""
name: Contrast Coding — Cell-Means Parameterization
source: https://bruno.nicenboim.me/bayescogsci/
example: contrasts1
dataset: contrasts1
chapter: Ch 6
formula: "DV ~ -1 + F"
----

Cell-means parameterization; no global intercept; each level gets its own mean.
"""
function examples(::Val{:contrasts1_cellmeans})
    d = load(Val(:contrasts1))
    return ("DV ~ -1 + F", d)
end

"""
name: Contrast Coding — Monotonic Effect
source: https://bruno.nicenboim.me/bayescogsci/
example: contrasts1
dataset: contrasts1
chapter: Ch 6
formula: "DV ~ 1 + mo(F)"
----

Monotonic effect; `F` treated as ordered categorical; `mo()` constrains the effect to
be monotonically increasing or decreasing across levels.
"""
function examples(::Val{:contrasts1_monotonic})
    d = load(Val(:contrasts1))
    return ("DV ~ 1 + mo(F)", d)
end

"""
name: contrasts2x2 — Synthetic 2×2 Factorial Design Data
source: synthetic
chapter: Ch 7
----

Synthetic data with factors `A` (2 levels: a1/a2) and `B` (2 levels: b1/b2). Includes
a main effect of `A`, main effect of `B`, and A×B interaction. 200 obs.

Columns: `DV` (continuous outcome), `A`, `B`, `pDV` (binary outcome).
"""
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

"""
name: 2×2 Factorial — Full Factorial Model
source: https://bruno.nicenboim.me/bayescogsci/
example: contrasts2x2
dataset: contrasts2x2
chapter: Ch 7
formula: "DV ~ A * B"
----

Full factorial Gaussian model: main effects of `A` and `B` plus A×B interaction.
"""
function examples(::Val{:contrasts2x2_factorial})
    d = load(Val(:contrasts2x2))
    return ("DV ~ A * B", d)
end

"""
name: 2×2 Factorial — Nested Model
source: https://bruno.nicenboim.me/bayescogsci/
example: contrasts2x2
dataset: contrasts2x2
chapter: Ch 7
formula: "DV ~ B / A"
----

`A` nested in `B`; `A` effects estimated separately within each level of `B`.
"""
function examples(::Val{:contrasts2x2_nested})
    d = load(Val(:contrasts2x2))
    return ("DV ~ B / A", d)
end

"""
name: 2×2 Factorial — Logistic Regression
source: https://bruno.nicenboim.me/bayescogsci/
example: contrasts2x2
dataset: contrasts2x2
chapter: Ch 7
formula: "pDV ~ A * B"
----

Bernoulli(logit) regression; 2×2 factorial design with binary outcome `pDV`.
"""
function examples(::Val{:contrasts2x2_logistic})
    d = load(Val(:contrasts2x2))
    return ("pDV ~ A * B", d)
end

"""
name: meta_sbi — Synthetic Meta-Analysis Data
source: https://osf.io/du3qp/?action=download
chapter: Ch 11
----

Bayesian random-effects meta-analysis; effect sizes with known standard errors.
Synthetic data generated here to match the structure. 20 studies.

Columns: `study_id`, `effect` (standardized mean difference), `SE` (standard error
of the effect).
"""
function load(::Val{:meta_sbi})
    rng      = MersenneTwister(11)
    n_studies = 20
    true_mu  = 0.5
    tau      = 0.3
    theta    = true_mu .+ tau .* randn(rng, n_studies)
    n_i      = rand(rng, 10:80, n_studies)
    SE       = 1.0 ./ sqrt.(float.(n_i))
    effect   = theta .+ SE .* randn(rng, n_studies)
    return DataFrame(;
        study_id = ["study_" * string(i) for i in 1:n_studies],
        effect,
        SE,
    )
end

"""
name: Meta-Analysis — Random Effects Model
source: https://bruno.nicenboim.me/bayescogsci/
example: meta_sbi
dataset: meta_sbi
chapter: Ch 11
formula: "effect | resp_se(SE, sigma = FALSE) ~ 1 + (1 | study_id)"
----

`resp_se()` passes known measurement error; `sigma = FALSE` fixes residual SD to 0;
random study intercept captures between-study heterogeneity τ.
"""
function examples(::Val{:meta_sbi})
    d = load(Val(:meta_sbi))
    return ("effect | resp_se(SE, sigma = FALSE) ~ 1 + (1 | study_id)", d)
end

"""
name: indiv_diff — Synthetic Individual Differences Data
source: https://osf.io/du3qp/?action=download
chapter: Ch 11
----

Participants completed a reading experiment and a test of prior context use (PCU). Both
measurements have uncertainty (standard errors). Synthetic data generated here. 60 obs.

Columns: `mean_rspeed` (mean reading speed), `se_rspeed` (SE of reading speed),
`c_mean_pcu` (centered mean PCU score), `se_pcu` (SE of PCU score).
"""
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

"""
name: Individual Differences — Naive OLS
source: https://bruno.nicenboim.me/bayescogsci/
example: indiv_diff
dataset: indiv_diff
chapter: Ch 11
formula: "mean_rspeed ~ c_mean_pcu"
----

Naive OLS ignoring measurement error in both predictor and response; biased estimates.
"""
function examples(::Val{:indiv_diff_naive})
    d = load(Val(:indiv_diff))
    return ("mean_rspeed ~ c_mean_pcu", d)
end

"""
name: Individual Differences — Measurement Error Model
source: https://bruno.nicenboim.me/bayescogsci/
example: indiv_diff
dataset: indiv_diff
chapter: Ch 11
formula: "mean_rspeed | resp_se(se_rspeed, sigma = TRUE) ~ me(c_mean_pcu, se_pcu)"
----

Measurement error in both response and predictor; `me()` propagates predictor
uncertainty; `resp_se()` propagates response uncertainty and adds residual sigma.
"""
function examples(::Val{:indiv_diff_me})
    d = load(Val(:indiv_diff))
    return ("mean_rspeed | resp_se(se_rspeed, sigma = TRUE) ~ me(c_mean_pcu, se_pcu)", d)
end
