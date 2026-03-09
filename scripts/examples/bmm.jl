using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: bmm — Bayesian Measurement Models
# Source: https://venpopov.github.io/bmm/
# Package: https://github.com/venpopov/bmm
#
# bmm uses brms as a Stan code generation backend. Models are specified via
# bmf() (bmmformula), which defines sub-model formulas for each latent parameter
# of the measurement model. The resulting Stan code is compiled and fit via brms.
#
# Formula notation here: semicolon-separated sub-models mirroring bmf() syntax,
# one sub-formula per latent parameter.

# ── Utility ───────────────────────────────────────────────────────────────────

_wrap(θ) = mod(θ + π, 2π) - π   # wrap angle to (−π, π]

# ── Bays 2009 synthetic ───────────────────────────────────────────────────────

"""
name: bays2009_synth — Synthetic Continuous Report Color Wheel Task
source: synthetic
----

Synthetic data mimicking Bays et al. (2009): participants (N=20) report the
recalled color of a probe item on a color wheel. Set sizes 1–6; 30 trials per
set size per participant.

Angular recall errors are generated from a mixture of a wrapped-normal
distribution (on-target memory; concentration decreases with set size) and a
uniform distribution (guessing; probability increases with set size). Five
non-target feature angles are also generated for each trial.

Columns: `id` (participant 1–20), `set_size` (String "1"–"6"), `error`
(angular recall error, radians, in (−π, π]), `non_target_1`–`non_target_5`
(non-target item feature angles, radians).
"""
function load(::Val{:bays2009_synth})
    rng = MersenneTwister(2009)
    rows = NamedTuple[]
    for id in 1:20
        for ss in 1:6
            p_rem = max(0.10, 0.90 - 0.13 * (ss - 1))
            σ     = max(0.30, 0.40 + 0.18 * (ss - 1))  # wrapped-normal ≈ von Mises
            for _ in 1:30
                err = rand(rng) < p_rem ? _wrap(σ * randn(rng)) : (rand(rng) - 0.5) * 2π
                nt  = [_wrap((rand(rng) - 0.5) * 2π) for _ in 1:5]
                push!(rows, (;
                    id,
                    set_size     = string(ss),
                    error        = err,
                    non_target_1 = nt[1],
                    non_target_2 = nt[2],
                    non_target_3 = nt[3],
                    non_target_4 = nt[4],
                    non_target_5 = nt[5],
                ))
            end
        end
    end
    return DataFrame(rows)
end

"""
name: Mixture2p — Two-Parameter Mixture Model (Set Size Effects)
source: https://venpopov.github.io/bmm/articles/bmm_mixture_models.html
example: color_wheel_mixture
dataset: bays2009_synth
formula: "thetat ~ 0 + set_size + (0 + set_size | id); kappa ~ 0 + set_size + (0 + set_size | id)"
----

Two-parameter mixture model for continuous report working memory data
(Bays et al. 2009). The response error is decomposed into:
- A von Mises (on-target memory) component with concentration `kappa`
- A uniform (guessing) component with probability `1 − inv_logit(thetat)`

Both `thetat` (log-odds of remembering) and `kappa` (von Mises concentration,
log scale) are regressed on set size with by-participant random slopes.
The `0 + set_size` intercept-free parameterisation gives one coefficient per
set-size level (treatment of set_size as factor).

bmm model spec: `mixture2p(resp_error = "error")`
"""
function examples(::Val{:mixture2p_setsize})
    data = load(Val(:bays2009_synth))
    return (
        "thetat ~ 0 + set_size + (0 + set_size | id); kappa ~ 0 + set_size + (0 + set_size | id)",
        data,
    )
end

"""
name: Mixture3p — Three-Parameter Mixture Model (Non-Target Swaps)
source: https://venpopov.github.io/bmm/articles/bmm_mixture_models.html
example: color_wheel_mixture
dataset: bays2009_synth
formula: "thetat ~ 0 + set_size + (0 + set_size | id); thetant ~ 0 + set_size + (0 + set_size | id); kappa ~ 0 + set_size + (0 + set_size | id)"
----

Three-parameter mixture model extending `mixture2p` with a non-target swap
component (Bays et al. 2009). The response error is a mixture of:
- Von Mises centred on the target (memory; `thetat`, `kappa`)
- Von Mises centred on a randomly selected non-target (swap; `thetant`, same `kappa`)
- Uniform (guessing)

`thetant` captures log-odds of swapping to a non-target. Requires non-target
feature angles (`non_target_1`–`non_target_5`) and set_size.

bmm model spec: `mixture3p(resp_error="error", nt_features=paste0("non_target_",1:5), set_size="set_size")`
"""
function examples(::Val{:mixture3p_setsize})
    data = load(Val(:bays2009_synth))
    return (
        "thetat ~ 0 + set_size + (0 + set_size | id); thetant ~ 0 + set_size + (0 + set_size | id); kappa ~ 0 + set_size + (0 + set_size | id)",
        data,
    )
end

# ── SDM synthetic data ────────────────────────────────────────────────────────

"""
name: sdm_synth — Synthetic Signal Discrimination Model Data
source: synthetic
----

Synthetic data for the Signal Discrimination Model (Oberauer 2023): 30
participants × 3 experimental conditions (A/B/C) × 50 trials. Conditions
differ in memory strength `c` and precision `kappa`.

Columns: `id` (participant), `cond` (String "A"/"B"/"C"), `y` (angular recall
error, radians, in (−π, π]).
"""
function load(::Val{:sdm_synth})
    rng  = MersenneTwister(2023)
    cond_params = [("A", 1.5, 0.35), ("B", 2.5, 0.25), ("C", 3.5, 0.18)]
    rows = NamedTuple[]
    for id in 1:30
        for (cond, _, σ) in cond_params
            p_rem = 0.75 + 0.05 * (cond == "C") - 0.05 * (cond == "A")
            for _ in 1:50
                y = rand(rng) < p_rem ? _wrap(σ * randn(rng)) : (rand(rng) - 0.5) * 2π
                push!(rows, (; id, cond, y))
            end
        end
    end
    return DataFrame(rows)
end

"""
name: SDM — Signal Discrimination Model (Condition Effects)
source: https://venpopov.github.io/bmm/articles/bmm_sdm_simple.html
example: sdm_condition
dataset: sdm_synth
formula: "c ~ 0 + cond; kappa ~ 0 + cond"
----

Signal Discrimination Model (Oberauer 2023) fit to a three-condition
continuous report experiment. The SDM represents memory as a signal on a
circular space; the response reflects discrimination between a memory trace
and noise.

`c` (memory strength, real-valued) and `kappa` (precision, log scale) each
receive one coefficient per condition via `0 + cond`. The model uses a custom
circular-normal likelihood implemented via brms `stanvar()`.

bmm model spec: `sdm(resp_error = "y")`
"""
function examples(::Val{:sdm_condition})
    data = load(Val(:sdm_synth))
    return ("c ~ 0 + cond; kappa ~ 0 + cond", data)
end

# ── IMM synthetic data ────────────────────────────────────────────────────────

"""
name: imm_synth — Synthetic Interference Measurement Model Data
source: synthetic
----

Synthetic data for the Interference Measurement Model (Oberauer & Lin 2017):
30 participants × 2 conditions (A/B) × 60 trials. Fixed set size = 4.
Non-target features and spatial distances are drawn uniformly.

Columns: `id` (participant), `cond` (String "A"/"B"), `resp_error` (angular
recall error, radians), `color_item2`–`color_item5` (non-target feature
angles), `dist_item2`–`dist_item5` (spatial distances, normalised 0–1),
`set_size` (= "4" throughout).
"""
function load(::Val{:imm_synth})
    rng  = MersenneTwister(2017)
    rows = NamedTuple[]
    for id in 1:30
        for cond in ["A", "B"]
            σ = cond == "A" ? 0.40 : 0.55
            for _ in 1:60
                err = rand(rng) < 0.80 ? _wrap(σ * randn(rng)) : (rand(rng) - 0.5) * 2π
                colors = [_wrap((rand(rng) - 0.5) * 2π) for _ in 1:4]
                dists  = rand(rng, 4)
                push!(rows, (;
                    id,
                    cond,
                    resp_error   = err,
                    color_item2  = colors[1],
                    color_item3  = colors[2],
                    color_item4  = colors[3],
                    color_item5  = colors[4],
                    dist_item2   = dists[1],
                    dist_item3   = dists[2],
                    dist_item4   = dists[3],
                    dist_item5   = dists[4],
                    set_size     = "4",
                ))
            end
        end
    end
    return DataFrame(rows)
end

"""
name: IMM Full — Interference Measurement Model (All Four Parameters)
source: https://venpopov.github.io/bmm/articles/bmm_imm.html
example: imm_condition
dataset: imm_synth
formula: "c ~ 0 + cond; a ~ 0 + cond; s ~ 0 + cond; kappa ~ 0 + cond"
----

Full Interference Measurement Model (Oberauer & Lin 2017) with all four
parameters varying across two conditions:
- `c` — baseline memory strength (item activation)
- `a` — associative activation (binding strength)
- `s` — spatial similarity gradient (spread of interference)
- `kappa` — von Mises precision (log scale)

Each parameter gets one coefficient per condition via `0 + cond`. Non-target
feature angles (`color_item2`–`color_item5`) and spatial distances
(`dist_item2`–`dist_item5`) enter the likelihood directly (not as predictors
in the formula).

bmm model spec: `imm(resp_error="resp_error", nt_features=paste0("color_item",2:5), set_size="set_size", nt_distances=paste0("dist_item",2:5), version="full")`
"""
function examples(::Val{:imm_full_condition})
    data = load(Val(:imm_synth))
    return ("c ~ 0 + cond; a ~ 0 + cond; s ~ 0 + cond; kappa ~ 0 + cond", data)
end
