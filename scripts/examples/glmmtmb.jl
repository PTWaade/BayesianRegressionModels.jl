using CSV, DataFrames, Downloads, Random, Statistics, LinearAlgebra

# Formulas from: glmmTMB (Generalised Linear Mixed Models via Template Model Builder)
# Sources: https://cran.r-project.org/web/packages/glmmTMB/vignettes/glmmTMB.html
#          https://cran.r-project.org/web/packages/glmmTMB/vignettes/zerinfl.html
#          https://cran.r-project.org/web/packages/glmmTMB/vignettes/covstruct.html
#          https://glmmtmb.github.io/glmmTMB/
# Package: https://cran.r-project.org/package=glmmTMB
#
# glmmTMB extends lme4-style formulas with additional sub-model arguments:
#   ziformula  — zero-inflation probability model (default ~0, i.e. no ZI)
#   dispformula — dispersion/scale model (default ~1, i.e. constant)
#
# Formula notation here appends non-default zi/disp as labelled parts:
#   "y ~ x + (1|g); zi: ~1; disp: ~x"
# Parts that are default (~0 for zi, ~1 for disp) are omitted.
#
# Selected families (beyond Gaussian/binomial/Poisson):
#   nbinom1     — NB, variance = μ·φ          (linear mean-variance)
#   nbinom2     — NB, variance = μ + μ²/k     (quadratic mean-variance)
#   genpois     — Generalised Poisson (over- or under-dispersion)
#   compois     — Conway-Maxwell-Poisson
#   tweedie     — compound Poisson-Gamma
#   beta_family — proportions (0,1)
#   ordbeta     — ordered beta (0, (0,1), 1)
#   truncated_* — zero-truncated variants (used in hurdle models via ziformula)
#
# Covariance structures for random effects (covstruct vignette):
#   ar1(time + 0 | group)      — AR(1) temporal
#   us(time + 0 | group)       — unstructured
#   toep(time + 0 | group)     — Toeplitz
#   exp(pos + 0 | group)       — spatial exponential (pos = numFactor of coords)
#   rr(fac + 0 | id, d = k)    — rank-k reduced-rank latent variable model

const OWLS_URL       = "https://vincentarelbundock.github.io/Rdatasets/csv/glmmTMB/Owls.csv"
const SALAMAN_URL    = "https://vincentarelbundock.github.io/Rdatasets/csv/glmmTMB/Salamanders.csv"

# ── Datasets ──────────────────────────────────────────────────────────────────

"""
name: Owls — Barn Owl Sibling Negotiation (glmmTMB)
source: https://vincentarelbundock.github.io/Rdatasets/csv/glmmTMB/Owls.csv
----

599 barn owl nest-nights from Zuur et al. (2009) via glmmTMB.
Columns: `Nest` (27 nests), `FoodTreatment` (Deprived/Satiated), `SexParent`
(Female/Male), `ArrivalTime` (decimal hour), `SiblingNegotiation` (call count,
response), `BroodSize` (number of nestlings), `NegPerChick` = SiblingNegotiation /
BroodSize. Column aliases used in the vignette: `NCalls` = SiblingNegotiation;
`FT` = FoodTreatment. `log_brood` = log(BroodSize) added by `load()`.
"""
function load(::Val{:owls})
    data = CSV.read(Downloads.download(OWLS_URL), DataFrame)
    data.NCalls    = data.SiblingNegotiation
    data.FT        = data.FoodTreatment
    data.log_brood = log.(data.BroodSize)
    return data
end

"""
name: Salamanders — Eastern Salamander Species Counts (glmmTMB)
source: https://vincentarelbundock.github.io/Rdatasets/csv/glmmTMB/Salamanders.csv
----

644 observations of salamander counts from Price et al. (2016) via glmmTMB.
Columns: `site` (23 sites), `mined` (yes/no mining disturbance), `cover` (% leaf
litter cover), `sample` (visit number), `DOP` (days since October 1), `Wtemp`
(water temperature), `DOY` (day of year), `spp` (9 species), `count` (response).
"""
load(::Val{:salamanders}) = CSV.read(Downloads.download(SALAMAN_URL), DataFrame)

"""
name: ar1_sim — Simulated AR(1) Grouped Time Series
source: synthetic
----

Synthetic panel data: 1,000 groups each with 6 ordered time points, generated
from a multivariate normal with AR(1) covariance (ρ = 0.7, σ² = 1) using
`MersenneTwister(42)`.
Columns: `y` (Gaussian response), `group` (factor), `times` (ordered factor 1–6).
"""
function load(::Val{:ar1_sim})
    rng    = MersenneTwister(42)
    n_grp  = 1000
    n_time = 6
    ρ      = 0.7
    # AR(1) covariance matrix
    Σ = [ρ^abs(i-j) for i in 1:n_time, j in 1:n_time]
    L = cholesky(Σ).L
    rows = NamedTuple[]
    for g in 1:n_grp
        z = L * randn(rng, n_time)
        for t in 1:n_time
            push!(rows, (; y=z[t], group=string(g), times=string(t)))
        end
    end
    return DataFrame(rows)
end

"""
name: beta_sim — Simulated Beta-Distributed Proportions
source: synthetic
----

1,000 synthetic beta-distributed observations with mean and dispersion both
depending on `x` (Uniform[0,1]).  Generated with `MersenneTwister(2023)`.
Columns: `y` (proportion in (0,1)), `x` (continuous predictor).
"""
function load(::Val{:beta_sim})
    rng = MersenneTwister(2023)
    n   = 1000
    x   = rand(rng, n)
    μ   = 1 ./ (1 .+ exp.(-(-1 .+ 3 .* x)))    # logistic
    φ   = exp.(1 .+ 2 .* x)                     # precision increases with x
    a   = μ .* φ
    b   = (1 .- μ) .* φ
    # Beta(a, b) via inverse CDF from uniform
    y   = [rand(rng) < 0.5 ? min(0.99, max(0.01, a[i]/(a[i]+b[i]) + 0.1*randn(rng))) :
                              min(0.99, max(0.01, a[i]/(a[i]+b[i]) - 0.1*randn(rng)))
           for i in 1:n]
    return DataFrame(; y, x)
end

"""
name: volcano_spatial — Noisy Volcanic Topography (base R)
source: synthetic
----

100 subsampled grid cells from R's built-in `volcano` dataset (87×61 matrix of
Māngere Mountain elevation, metres), corrupted with Gaussian noise (σ=3).
Used to demonstrate spatial exponential covariance in glmmTMB.
Columns: `z` (noisy elevation), `x` (column index 1–61), `y` (row index 1–87),
`group` (factor "1" — all observations share one spatial covariance block).
"""
function load(::Val{:volcano_spatial})
    # Hard-coded 5×20 = 100 representative elevation values from volcano[1:5, 1:20]
    rng = MersenneTwister(1955)
    # Representative elevation slice: rows 40-44, cols 25-44 of volcano
    # (central region, moderate variance)
    elev = Float64[
        100,100,101,101,101,101,100,100,100,99,99,99,98,98,97,96,95,95,95,96,
        101,102,103,104,104,103,103,102,101,100,99,98,97,96,96,95,95,95,95,95,
        102,103,104,106,107,106,105,104,102,101,100,99,98,97,96,95,95,95,96,96,
        103,104,106,108,110,109,108,106,104,103,101,100,99,98,97,96,95,95,96,97,
        104,105,107,110,112,111,110,108,106,104,103,101,100,99,98,97,96,96,97,98,
    ]
    n = length(elev)
    row_idx = repeat(1:5, inner=20)
    col_idx = repeat(1:20, 5)
    z = elev .+ 3 .* randn(rng, n)
    return DataFrame(; z, x=col_idx, y=row_idx, group=fill("1", n))
end

# ── Owls: Zero-Inflation Variants ─────────────────────────────────────────────

"""
name: Barn Owls — Zero-Inflated Poisson GLMM
source: https://cran.r-project.org/web/packages/glmmTMB/vignettes/glmmTMB.html
example: owls_zi
dataset: owls
formula: "NCalls ~ (FT + ArrivalTime)*SexParent + offset(log_brood) + (1|Nest); zi: ~1"
----

`glmmTMB` with `family = poisson` and `ziformula = ~1` (constant zero-inflation
probability).  Nest random intercept handles clustering; offset accounts for
brood size.  Baseline model for owl sibling negotiation calls.
"""
function examples(::Val{:owls_zip})
    return (
        "NCalls ~ (FT + ArrivalTime)*SexParent + offset(log_brood) + (1|Nest); zi: ~1",
        load(Val(:owls)),
    )
end

"""
name: Barn Owls — Zero-Inflated Negative Binomial (NB2) GLMM
source: https://cran.r-project.org/web/packages/glmmTMB/vignettes/glmmTMB.html
example: owls_zi
dataset: owls
formula: "NCalls ~ (FT + ArrivalTime)*SexParent + offset(log_brood) + (1|Nest); zi: ~1"
----

Same formula as `:owls_zip` but with `family = nbinom2` (variance = μ + μ²/k).
Handles both overdispersion and excess zeros.  AIC typically preferred over
the Poisson version.
"""
function examples(::Val{:owls_zinb2})
    return (
        "NCalls ~ (FT + ArrivalTime)*SexParent + offset(log_brood) + (1|Nest); zi: ~1",
        load(Val(:owls)),
    )
end

"""
name: Barn Owls — Zero-Inflated Negative Binomial (NB1) GLMM
source: https://cran.r-project.org/web/packages/glmmTMB/vignettes/glmmTMB.html
example: owls_zi
dataset: owls
formula: "NCalls ~ (FT + ArrivalTime)*SexParent + BroodSize + (1|Nest); zi: ~1"
----

`family = nbinom1` (variance = φμ, linear mean-variance relationship); brood size
as a fixed covariate rather than offset.  Contrasts NB1 vs NB2 parameterisation.
"""
function examples(::Val{:owls_zinb1})
    return (
        "NCalls ~ (FT + ArrivalTime)*SexParent + BroodSize + (1|Nest); zi: ~1",
        load(Val(:owls)),
    )
end

"""
name: Barn Owls — Hurdle NB1 GLMM
source: https://cran.r-project.org/web/packages/glmmTMB/vignettes/glmmTMB.html
example: owls_zi
dataset: owls
formula: "NCalls ~ (FT + ArrivalTime)*SexParent + BroodSize + (1|Nest); zi: ~."
----

Hurdle model: `ziformula = ~.` mirrors the conditional formula in the zero part,
combined with `family = truncated_nbinom1` for the positive count part.  Treats
zero-generation and positive-count processes as completely separate.
"""
function examples(::Val{:owls_hurdle})
    return (
        "NCalls ~ (FT + ArrivalTime)*SexParent + BroodSize + (1|Nest); zi: ~.",
        load(Val(:owls)),
    )
end

# ── Salamanders: Species-Level ZI ─────────────────────────────────────────────

"""
name: Salamanders — ZI NB2 with Species-Varying Zero-Inflation
source: https://cran.r-project.org/web/packages/glmmTMB/vignettes/glmmTMB.html
example: salamanders_zi
dataset: salamanders
formula: "count ~ spp + (1|site); zi: ~spp"
----

Zero-inflation probability varies by species (`ziformula = ~spp`).  The
conditional mean count also varies by species with a site random intercept.
`family = nbinom2`.
"""
function examples(::Val{:salamanders_zinb2})
    return ("count ~ spp + (1|site); zi: ~spp", load(Val(:salamanders)))
end

"""
name: Salamanders — Generalised Poisson with ZI
source: https://cran.r-project.org/web/packages/glmmTMB/vignettes/glmmTMB.html
example: salamanders_zi
dataset: salamanders
formula: "count ~ spp + mined + (1|site); zi: ~spp + mined"
----

`family = genpois` (Generalised Poisson) handles both over- and under-dispersion;
zero-inflation depends on both species and mining status.
"""
function examples(::Val{:salamanders_genpois})
    return ("count ~ spp + mined + (1|site); zi: ~spp + mined", load(Val(:salamanders)))
end

# ── Distributional Models ─────────────────────────────────────────────────────

"""
name: Beta Regression — Dispersion as Function of Predictor
source: https://cran.r-project.org/web/packages/glmmTMB/vignettes/glmmTMB.html
example: beta_disp
dataset: beta_sim
formula: "y ~ x; disp: ~x"
----

`family = beta_family()` with `dispformula = ~x` models both the mean proportion
and the precision (φ) as functions of `x`.  Demonstrates that ignoring
heterogeneous dispersion (using `dispformula = ~1`) leads to worse fit.
"""
function examples(::Val{:beta_disp})
    return ("y ~ x; disp: ~x", load(Val(:beta_sim)))
end

# ── Covariance Structures ──────────────────────────────────────────────────────

"""
name: Grouped Time Series — AR(1) Random-Effect Covariance
source: https://cran.r-project.org/web/packages/glmmTMB/vignettes/covstruct.html
example: ar1_covar
dataset: ar1_sim
formula: "y ~ ar1(times + 0 | group)"
----

`ar1(times + 0 | group)` specifies AR(1) autocorrelation across ordered `times`
within each `group` as a random-effect covariance structure.  Gaussian family.
Distinguishes glmmTMB from lme4, which does not support non-diagonal random-effect
covariance natively.
"""
function examples(::Val{:ar1_covar})
    return ("y ~ ar1(times + 0 | group)", load(Val(:ar1_sim)))
end

"""
name: Volcanic Topography — Spatial Exponential Covariance
source: https://cran.r-project.org/web/packages/glmmTMB/vignettes/covstruct.html
example: volcano_spatial
dataset: volcano_spatial
formula: "z ~ 1 + exp(pos + 0 | group)"
----

`exp(pos + 0 | group)` fits a spatial random field with exponential correlation
decay.  `pos` must be a `numFactor` encoding 2D coordinates (x, y).  Gaussian
family.  All observations share one group so the full spatial covariance is
estimated as a single 100×100 block.
Note: `pos = numFactor(x, y)` must be computed before calling `glmmTMB`.
"""
function examples(::Val{:volcano_spatial})
    return ("z ~ 1 + exp(pos + 0 | group)", load(Val(:volcano_spatial)))
end
