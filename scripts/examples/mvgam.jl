using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: mvgam — Multivariate Dynamic Generalized Additive Models
# Source: https://nicholasjclark.github.io/mvgam/
# Package: https://github.com/nicholasjclark/mvgam
#
# mvgam fits Dynamic Bayesian GAMs for multivariate time series. It compiles
# Stan code incorporating brms, mgcv, and splines2. Models are specified via:
#   formula       — observation-level regression (smooth terms, predictors)
#   trend_formula — process/trend-level regression (latent dynamics)
#   trend_model   — latent state-space structure: AR(), VAR(), RW(), GP(), etc.
#
# Formula notation here: when both formulas are present they are combined as
#   "obs: <formula> | trend: <trend_formula> [<trend_model>]"
# The observation formula alone is shown when there is no trend_formula.

# ── Utility ───────────────────────────────────────────────────────────────────

_logistic(x) = 1 / (1 + exp(-x))

# ── Portal-like rodent count data ─────────────────────────────────────────────

# Minimal Poisson sampler (avoid Distributions.jl dependency)
function _rpois(rng, λ)
    λ <= 0 && return 0
    # Knuth algorithm for small λ; normal approximation for large
    if λ < 30
        L, k, p = exp(-λ), 0, 1.0
        while p > L; k += 1; p *= rand(rng); end
        return k - 1
    else
        return max(0, round(Int, λ + sqrt(λ) * randn(rng)))
    end
end

"""
name: portal_synth — Synthetic Rodent Count Time Series (Portal-Like)
source: synthetic
----

Synthetic multi-species count data mimicking the Portal Project long-term
rodent monitoring dataset (Arizona, USA). Five species observed monthly for
60 months (5 years). Each species has a latent AR(1) trend plus a seasonal
NDVI effect.

Columns: `count` (Poisson count), `series` (String "sp1"–"sp5"), `time`
(Int 1–60), `year_fac` (String factor "Y1"–"Y5"), `season` (Int 1–12,
calendar month), `ndvi_ma12` (synthetic 12-month moving-average NDVI, mean 0
std 1).

Reference dataset: Ernest et al. (2018) Portal Project Teaching Database.
"""
function load(::Val{:portal_synth})
    rng       = MersenneTwister(2018)
    n_sp      = 5
    n_time    = 60
    ndvi_raw  = [0.3 * sin(2π * t / 12) + 0.1 * randn(rng) for t in 1:n_time]
    ndvi_ma12 = [mean(ndvi_raw[max(1, t-11):t]) for t in 1:n_time]
    μ_ndvi, σ_ndvi = mean(ndvi_ma12), std(ndvi_ma12)
    ndvi_ma12 = (ndvi_ma12 .- μ_ndvi) ./ σ_ndvi
    rows = NamedTuple[]
    for sp in 1:n_sp
        baseline = 1.0 + 0.5 * randn(rng)
        ndvi_eff = 0.3 + 0.2 * randn(rng)
        ar_coef  = 0.6 + 0.1 * randn(rng)
        η = 0.0
        for t in 1:n_time
            η = ar_coef * η + 0.3 * randn(rng)
            λ = exp(baseline + ndvi_eff * ndvi_ma12[t] + η)
            push!(rows, (;
                count     = _rpois(rng, λ),
                series    = "sp$sp",
                time      = t,
                year_fac  = "Y$(div(t - 1, 12) + 1)",
                season    = mod1(t, 12),
                ndvi_ma12 = ndvi_ma12[t],
            ))
        end
    end
    return DataFrame(rows)
end

"""
name: Poisson GLM — Hierarchical Year Random Effects
source: https://nicholasjclark.github.io/mvgam/articles/mvgam_overview.html
example: portal_counts
dataset: portal_synth
formula: 'count ~ s(year_fac, bs = "re") - 1'
----

Baseline Poisson GLM capturing inter-annual variation in rodent counts via
hierarchical (random-effects) smooths on year. `s(year_fac, bs = "re") - 1`
is the mgcv notation for a random intercept per year level with no overall
intercept. No explicit latent trend process.

Fit with: `mvgam(count ~ s(year_fac, bs="re") - 1, family=poisson(), data=portal_train, trend_model="None")`
"""
function examples(::Val{:portal_glm_re})
    data = load(Val(:portal_synth))
    return ("count ~ s(year_fac, bs = \"re\") - 1", data)
end

"""
name: Poisson GAM — NDVI Smooth + AR(1) Latent Trend
source: https://nicholasjclark.github.io/mvgam/articles/mvgam_overview.html
example: portal_counts
dataset: portal_synth
formula: "obs: count ~ 1 | trend: ~ s(ndvi_ma12, k = 6) [AR()]"
----

State-space Poisson model separating a smooth observation-level NDVI effect
from a latent AR(1) trend process. The trend captures residual temporal
autocorrelation not explained by the environmental covariate.

- `obs` formula: intercept-only observation model
- `trend` formula: smooth NDVI effect at the latent process level (`k = 6`
  basis functions)
- `trend_model = AR()`: AR(1) latent trend

Fit with: `mvgam(count ~ 1, trend_formula = ~ s(ndvi_ma12, k=6), trend_model=AR(), family=poisson(), data=portal_train)`
"""
function examples(::Val{:portal_ar_ndvi})
    data = load(Val(:portal_synth))
    return ("obs: count ~ 1 | trend: ~ s(ndvi_ma12, k = 6) [AR()]", data)
end

# ── Plankton-like multivariate Gaussian time series ───────────────────────────

"""
name: plankton_synth — Synthetic Multivariate Gaussian Plankton-Like Time Series
source: synthetic
----

Synthetic Gaussian time series for 4 "species" over 100 time points, mimicking
the Lake Washington phytoplankton dataset (Hampton et al. 2006). Observations
depend on a tensor-product interaction of temperature and month (seasonal
environmental driver).

Columns: `y` (Gaussian response, log-scale abundance), `series` (String
"sp1"–"sp4"), `time` (Int 1–100), `temp` (temperature, approximately
standardised), `month` (Int 1–12, cyclic seasonal index).

Reference: Hampton SE et al. (2006) Sixty years of environmental change in the
world's sixth largest lake — Lake Washington, Seattle, WA. *Global Change
Biology*.
"""
function load(::Val{:plankton_synth})
    rng    = MersenneTwister(2006)
    n_sp   = 4
    n_time = 100
    month  = [mod1(t, 12) for t in 1:n_time]
    # Temperature: seasonal cycle + slow trend + noise
    temp_base = [1.5 * sin(2π * (m - 7) / 12) for m in month]
    temp      = temp_base .+ 0.02 .* (1:n_time) .+ 0.3 .* randn(rng, n_time)
    temp      = (temp .- mean(temp)) ./ std(temp)
    # VAR(1) latent process
    A   = [0.6 -0.1 0.05 0.0;
           0.1  0.7 0.0  0.0;
           0.0  0.05 0.65 -0.1;
           0.0  0.0  0.1  0.7]
    η   = zeros(n_sp)
    rows = NamedTuple[]
    temp_eff = [0.5, -0.3, 0.4, -0.2]   # species-specific temperature effects
    for t in 1:n_time
        η = A * η .+ 0.3 .* randn(rng, n_sp)
        for sp in 1:n_sp
            μ = temp_eff[sp] * temp[t] + η[sp]
            push!(rows, (;
                y      = μ + 0.2 * randn(rng),
                series = "sp$sp",
                time   = t,
                temp   = temp[t],
                month  = month[t],
            ))
        end
    end
    return DataFrame(rows)
end

"""
name: Gaussian VAR — Tensor-Product Environmental Smooths
source: https://nicholasjclark.github.io/mvgam/articles/trend_formulas.html
example: plankton_var
dataset: plankton_synth
formula: "obs: y ~ -1 | trend: ~ te(temp, month, k = c(4, 4)) + te(temp, month, k = c(4, 4), by = trend) - 1 [VAR(cor = TRUE)]"
----

Vector Autoregressive (VAR) model for multivariate Gaussian plankton time
series with correlated process errors. Environmental forcing (temperature ×
month tensor product) acts at the latent process level.

- `obs` formula: no observation-level predictors
- `trend` formula: shared tensor-product smooth `te(temp, month, k=c(4,4))`
  plus species-specific deviations (`by = trend`); `-1` removes intercept
- `trend_model = VAR(cor = TRUE)`: VAR with correlated cross-series innovations

The `by = trend` idiom in mvgam indexes each latent trend series, allowing
species-specific environmental response curves.

Fit with: `mvgam(y~-1, trend_formula=~te(temp,month,k=c(4,4))+te(temp,month,k=c(4,4),by=trend)-1, trend_model=VAR(cor=TRUE), family=gaussian(), data=plankton_train)`
"""
function examples(::Val{:plankton_var})
    data = load(Val(:plankton_synth))
    return (
        "obs: y ~ -1 | trend: ~ te(temp, month, k = c(4, 4)) + te(temp, month, k = c(4, 4), by = trend) - 1 [VAR(cor = TRUE)]",
        data,
    )
end

# ── Salmon survival (Beta regression with time-varying effect) ─────────────────

"""
name: salmon_synth — Synthetic Salmon Survival Data (Beta Regression)
source: synthetic
----

Synthetic data mimicking annual Pacific salmon survival proportions from the
Columbia River basin (30 years). Survival is modelled as a function of spring
upwelling index (`CUI.apr`) whose effect on survival varies over time.

Columns: `survival` (proportion, ∈ (0, 1)), `time` (Int 1–30), `CUI.apr`
(April upwelling index, approximately standardised).

Reference: Cox SP & Hinch SG (1997) Jackson Lake chinook sockeye return data
(stylised).
"""
function load(::Val{:salmon_synth})
    rng   = MersenneTwister(1997)
    n     = 30
    cui   = randn(rng, n)   # upwelling index
    # Time-varying coefficient: RW on log scale
    β = cumsum(0.15 * randn(rng, n))
    ar_state = 0.0
    rows = NamedTuple[]
    for t in 1:n
        ar_state = 0.7 * ar_state + 0.3 * randn(rng)
        logit_s  = -0.5 + β[t] * cui[t] + ar_state
        s        = _logistic(logit_s)
        # Beta noise: clamp to avoid exact 0/1
        s        = clamp(s + 0.05 * randn(rng), 0.01, 0.99)
        push!(rows, (; survival = s, time = t, CUI_apr = cui[t]))
    end
    df = DataFrame(rows)
    rename!(df, :CUI_apr => Symbol("CUI.apr"))
    return df
end

"""
name: Beta AR(1) — Time-Varying Upwelling Effect on Salmon Survival
source: https://nicholasjclark.github.io/mvgam/articles/time_varying_effects.html
example: salmon_survival
dataset: salmon_synth
formula: "obs: survival ~ 1 | trend: ~ dynamic(CUI.apr, k = 25, scale = FALSE) - 1 [AR()]"
----

Beta-distributed salmon survival proportions with an AR(1) latent trend and
a time-varying effect of spring upwelling index (`CUI.apr`).

- `obs` formula: intercept-only observation model (Beta family)
- `trend` formula: `dynamic(CUI.apr, k=25, scale=FALSE)` makes the regression
  coefficient on upwelling a smooth function of time (random-walk basis with
  25 knots)
- `trend_model = AR()`: AR(1) latent process in addition to the TVP

The `dynamic()` function in mvgam fits a time-varying parameter (TVP) model
where the coefficient evolves as a random walk or Hilbert-space GP over time.

Fit with: `mvgam(survival~1, trend_formula=~dynamic(CUI.apr,k=25,scale=FALSE)-1, trend_model=AR(), family=betar(), data=salmon_train)`
"""
function examples(::Val{:salmon_beta_ar})
    data = load(Val(:salmon_synth))
    return (
        "obs: survival ~ 1 | trend: ~ dynamic(CUI.apr, k = 25, scale = FALSE) - 1 [AR()]",
        data,
    )
end

# ── N-mixture data ─────────────────────────────────────────────────────────────

"""
name: nmix_synth — Synthetic N-Mixture Detection/Abundance Data
source: synthetic
----

Synthetic N-mixture data: 100 sites × 3 replicates. True abundance at each
site depends on an abundance covariate (`abund_cov`) and a categorical factor
(`abund_fac`). Detection probability depends on two detection covariates.

Columns: `y` (observed count, ≤ true N), `site` (Int 1–100), `rep` (Int 1–3),
`det_cov` (detection covariate 1, N(0,1)), `det_cov2` (detection covariate 2,
N(0,1)), `abund_cov` (abundance covariate, N(0,1)), `abund_fac` (String factor
"low"/"mid"/"high", 3 levels).
"""
function load(::Val{:nmix_synth})
    rng          = MersenneTwister(3131)
    n_sites      = 100
    n_reps       = 3
    abund_fac    = rand(rng, ["low", "mid", "high"], n_sites)
    fac_effect   = Dict("low" => -0.5, "mid" => 0.0, "high" => 0.5)
    abund_cov    = randn(rng, n_sites)
    true_N       = [_rpois(rng, exp(1.5 + 0.6 * abund_cov[i] + fac_effect[abund_fac[i]])) for i in 1:n_sites]
    rows = NamedTuple[]
    for site in 1:n_sites
        for rep in 1:n_reps
            dc1 = randn(rng)
            dc2 = randn(rng)
            p   = _logistic(-0.3 + 0.5 * dc1 - 0.4 * dc2)
            y   = sum(rand(rng) < p for _ in 1:true_N[site])
            push!(rows, (;
                y         = y,
                site      = site,
                rep       = rep,
                det_cov   = dc1,
                det_cov2  = dc2,
                abund_cov = abund_cov[site],
                abund_fac = abund_fac[site],
            ))
        end
    end
    return DataFrame(rows)
end

"""
name: N-Mixture — Smooth Detection and Abundance Sub-Models
source: https://nicholasjclark.github.io/mvgam/articles/nmixtures.html
example: nmix_detection
dataset: nmix_synth
formula: 'obs: y ~ s(det_cov, k = 4) + s(det_cov2, k = 4) | trend: ~ s(abund_cov, k = 4) + s(abund_fac, bs = "re") [nmix()]'
----

N-mixture model for imperfect detection with non-parametric smooth effects.
The N-mixture likelihood jointly models the imperfect-detection observation
process and the true abundance process.

- `obs` formula: detection probability sub-model; smooth effects of two
  detection covariates
- `trend` formula: latent abundance sub-model; smooth abundance covariate +
  hierarchical random effect for the categorical habitat factor
- `family = nmix()`: N-mixture likelihood (Royle 2004)

Fit with: `mvgam(y~s(det_cov,k=4)+s(det_cov2,k=4), trend_formula=~s(abund_cov,k=4)+s(abund_fac,bs="re"), family=nmix(), data=nmix_data)`
"""
function examples(::Val{:nmix_detection})
    data = load(Val(:nmix_synth))
    return (
        "obs: y ~ s(det_cov, k = 4) + s(det_cov2, k = 4) | trend: ~ s(abund_cov, k = 4) + s(abund_fac, bs = \"re\") [nmix()]",
        data,
    )
end

"""
name: Shared Latent Trend — Multi-Series Seasonal AR(1)
source: https://nicholasjclark.github.io/mvgam/articles/shared_states.html
example: portal_counts
dataset: portal_synth
formula: 'obs: count ~ series - 1 | trend: ~ s(season, bs = "cc", k = 8) [AR()]'
----

Multi-series Poisson model where all five species share a single latent AR(1)
trend, with series-specific observation offsets and a cyclic seasonal spline
at the trend level.

- `obs` formula: `series - 1` gives a species-specific intercept (fixed offset)
- `trend` formula: cyclic cubic regression spline for seasonality (`bs = "cc"`)
- `trend_model = AR()` with `trend_map` forcing all series to share one trend

The shared-trend structure is specified via a `trend_map` data.frame passed
separately to `mvgam()` mapping each series to the same latent trend index.

Fit with: `mvgam(count~series-1, trend_formula=~s(season,bs="cc",k=8), trend_model=AR(), family=poisson(), data=portal_train, trend_map=trend_map)`
"""
function examples(::Val{:portal_shared_trend})
    data = load(Val(:portal_synth))
    return (
        "obs: count ~ series - 1 | trend: ~ s(season, bs = \"cc\", k = 8) [AR()]",
        data,
    )
end
