using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: R-INLA (Integrated Nested Laplace Approximation)
# Sources: https://becarioprecario.bitbucket.io/inla-gitbook/ (Gómez-Rubio)
#          https://becarioprecario.bitbucket.io/spde-gitbook/ (Krainski et al.)
# Package: https://www.r-inla.org
#
# R-INLA uses a standard R fixed-effect formula extended by f() random-effect
# terms rather than lme4-style (1|group) notation:
#   f(group, model = "iid")          — exchangeable random intercept
#   f(time,  model = "rw1")          — first-order random walk
#   f(time,  model = "ar1")          — AR(1) latent process
#   f(time,  model = "seasonal", season.length = 12) — seasonal component
#   f(id,    model = "besag", graph = W) — ICAR spatial (requires adjacency W)
#   f(id,    model = "bym",   graph = W) — BYM = ICAR + IID
#   f(i,     model = spde)            — Matérn SPDE spatial field
#
# Formula strings in this catalog use single quotes for nested R strings where
# needed (e.g. model = 'iid').  f() terms that require non-tabular objects
# (graph, SPDE mesh) are noted in the docstring.
#
# Response helpers:
#   inla.surv(time, status)  — right-censored survival time

const CEMENT_URL      = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/cement.csv"
const PENICILLIN_URL  = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Penicillin.csv"
const SLEEPSTUDY_URL  = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv"
const VETERAN_URL     = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/veteran.csv"
const LIDAR_URL       = "https://vincentarelbundock.github.io/Rdatasets/csv/SemiPar/lidar.csv"
const BOSTON_URL      = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Boston.csv"

# ── Datasets ──────────────────────────────────────────────────────────────────

"""
name: cement — Heat of Hardening in Cement (MASS)
source: https://vincentarelbundock.github.io/Rdatasets/csv/MASS/cement.csv
----

13 observations from Woods, Steinour & Starke (1932) via MASS.
Columns: `y` (heat evolved, cal/g), `x1`–`x4` (percentages of four chemical
compounds: tricalcium aluminate, tricalcium silicate, tetracalcium alumino
ferrite, dicalcium silicate).
"""
load(::Val{:cement}) = CSV.read(Downloads.download(CEMENT_URL), DataFrame)

"""
name: Penicillin — Zone of Inhibition Assay (lme4)
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Penicillin.csv
----

144 observations (24 plates × 6 penicillin samples) from Davies (1954) via lme4.
Columns: `diameter` (zone of inhibition, mm), `plate` (assay plate A–X),
`sample` (penicillin sample A–F). `plate_id` (1–24) added by `load()`.
"""
function load(::Val{:penicillin})
    data = CSV.read(Downloads.download(PENICILLIN_URL), DataFrame)
    plates = sort(unique(data.plate))
    id_map = Dict(p => i for (i, p) in enumerate(plates))
    data.plate_id = [id_map[p] for p in data.plate]
    return data
end

"""
name: sleepstudy — Sleep Deprivation Reaction Times (lme4)
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv
----

180 observations (18 subjects × 10 days) from Belenky et al. (2003) via lme4.
Columns: `Reaction` (ms), `Days` (0–9), `Subject` (factor label).
`subject_id` (1–18) added by `load()` for INLA's numeric group index requirement.
"""
function load(::Val{:sleepstudy_inla})
    data = CSV.read(Downloads.download(SLEEPSTUDY_URL), DataFrame)
    subjs = sort(unique(data.Subject))
    id_map = Dict(s => i for (i, s) in enumerate(subjs))
    data.subject_id = [id_map[s] for s in data.Subject]
    return data
end

"""
name: airpassengers — Monthly Airline Passengers 1949–1960 (Box & Jenkins)
source: synthetic
----

144 monthly international airline passenger counts (thousands) from Box &
Jenkins (1976) — the classic `AirPassengers` time series in base R.
Columns: `y` (passengers), `t` (index 1–144), `month` (1–12), `year` (1949–1960),
`log_y` = log(y).
"""
function load(::Val{:airpassengers})
    y = Int[
        112,118,132,129,121,135,148,148,136,119,104,118,
        115,126,141,135,125,149,170,170,158,133,114,140,
        145,150,178,163,172,178,199,199,184,162,146,166,
        171,180,193,181,183,218,230,242,209,191,172,194,
        196,196,236,235,229,243,264,272,237,211,180,201,
        204,188,235,227,234,264,302,293,259,229,203,229,
        242,233,267,269,270,315,364,347,312,274,237,278,
        284,277,317,313,318,374,413,405,355,306,271,306,
        315,301,356,348,355,422,465,467,404,347,305,336,
        340,318,362,348,363,435,491,505,404,359,310,337,
        360,342,406,396,420,472,548,559,463,407,362,405,
        417,391,419,461,472,535,622,606,508,461,390,432,
    ]
    t     = 1:144
    month = repeat(1:12, 12)
    year  = repeat(1949:1960, inner=12)
    return DataFrame(; y, log_y=log.(y), t=collect(t), month, year)
end

"""
name: veteran — VA Lung Cancer Trial (survival)
source: https://vincentarelbundock.github.io/Rdatasets/csv/survival/veteran.csv
----

137 male lung cancer patients in a Veterans Administration randomised trial
(Kalbfleisch & Prentice 1980) via the survival package.
Columns: `trt` (1 = standard, 2 = test), `celltype` (squamous/smallcell/adeno/large),
`time` (survival/censoring time, days), `status` (1 = dead), `karno` (Karnofsky
score), `diagtime` (months from diagnosis), `age`, `prior` (prior therapy 0/10).
"""
load(::Val{:veteran}) = CSV.read(Downloads.download(VETERAN_URL), DataFrame)

"""
name: lidar — LIDAR Atmospheric Mercury Measurements (SemiPar)
source: https://vincentarelbundock.github.io/Rdatasets/csv/SemiPar/lidar.csv
----

221 observations from the LIDAR dataset of Sigrist et al. (1994) via SemiPar.
Columns: `range` (distance travelled, metres) and `logratio` (log-ratio of
returned laser pulse; a measure of atmospheric mercury concentration).
Classic 1D nonparametric regression benchmark.
"""
load(::Val{:lidar}) = CSV.read(Downloads.download(LIDAR_URL), DataFrame)

"""
name: boston_housing — Boston Housing (MASS)
source: https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Boston.csv
----

506 census tracts in the Boston Standard Metropolitan Statistical Area (Harrison &
Rubinfeld 1978) via MASS. Columns: `medv` (median home value, USD 1000s), `crim`,
`zn`, `indus`, `chas`, `nox`, `rm`, `age`, `dis`, `rad`, `tax`, `ptratio`,
`black`, `lstat`. `id` (1–506) added for INLA indexing.
Note: the `besag` model requires a spatial adjacency graph `W` passed separately
via `f(..., graph = W)`.
"""
function load(::Val{:boston_housing})
    data = CSV.read(Downloads.download(BOSTON_URL), DataFrame)
    data.id = 1:nrow(data)
    return data
end

"""
name: surg — Surgical Mortality Benchmarking (synthetic)
source: synthetic
----

12-hospital synthetic dataset inspired by the WinBUGS `Surg` example (Spiegelhalter
et al. 1996) used in Gómez-Rubio §2.4.3. Columns: `hospital` (1–12),
`n` (number of operations), `r` (30-day deaths).
Synthetic values mimic the original marginal rates (≈1–12 % mortality) with
`MersenneTwister(1996)`.
"""
function load(::Val{:surg})
    rng = MersenneTwister(1996)
    n   = [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360]
    p   = [0.04, 0.12, 0.07, 0.06, 0.04, 0.07, 0.06, 0.14, 0.07, 0.08, 0.11, 0.07]
    r   = [rand(rng, 0:round(Int, ni)) < round(Int, pi_*ni) ? round(Int, pi_*ni) : rand(rng, 0:round(Int, pi_*ni*1.5)) for (ni, pi_) in zip(n, p)]
    r   = round.(Int, p .* n)   # use expected values for reproducibility
    return DataFrame(; hospital=1:12, n, r)
end

# ── Basic Gaussian LM ─────────────────────────────────────────────────────────

"""
name: Cement — Gaussian Linear Model (no random effects)
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-INLA.html
example: cement
dataset: cement
formula: "y ~ x1 + x2 + x3 + x4"
----

`inla()` with `family = "gaussian"` — the simplest INLA use case, equivalent to
`lm()` but with Bayesian inference via INLA. Demonstrates default weakly-informative
priors on regression coefficients.
"""
function examples(::Val{:cement_lm})
    return ("y ~ x1 + x2 + x3 + x4", load(Val(:cement)))
end

# ── Random Intercept Models ───────────────────────────────────────────────────

"""
name: Penicillin — Gaussian GLMM with IID Random Intercept
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-multilevel.html
example: penicillin
dataset: penicillin
formula: "diameter ~ 1 + sample + f(plate_id, model = 'iid')"
----

`f(plate_id, model = "iid")` specifies exchangeable random intercepts per assay
plate — the INLA equivalent of `(1|plate)` in lme4.  Gaussian family.
"""
function examples(::Val{:penicillin_iid})
    return ("diameter ~ 1 + sample + f(plate_id, model = 'iid')", load(Val(:penicillin)))
end

"""
name: Sleep Deprivation — Gaussian GLMM with IID Random Intercept
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-multilevel.html
example: sleepstudy
dataset: sleepstudy_inla
formula: "Reaction ~ 1 + Days + f(subject_id, model = 'iid')"
----

Random intercept per subject; `subject_id` is the numeric version of `Subject`.
Gaussian family.  Equivalent to `lmer(Reaction ~ Days + (1|Subject))`.
"""
function examples(::Val{:sleepstudy_iid})
    return ("Reaction ~ 1 + Days + f(subject_id, model = 'iid')", load(Val(:sleepstudy_inla)))
end

"""
name: Sleep Deprivation — Gaussian GLMM with IID Random Slopes
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-multilevel.html
example: sleepstudy
dataset: sleepstudy_inla
formula: "Reaction ~ 1 + f(subject_id, Days, model = 'iid')"
----

Random slopes model: `f(subject_id, Days, ...)` specifies subject-specific slopes
for `Days` — the INLA way of writing `(0 + Days | Subject)`.  Gaussian family.
"""
function examples(::Val{:sleepstudy_slopes})
    return ("Reaction ~ 1 + f(subject_id, Days, model = 'iid')", load(Val(:sleepstudy_inla)))
end

# ── Temporal Models ───────────────────────────────────────────────────────────

"""
name: Airline Passengers — Random Walk (RW1) Temporal Smoothing
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-temporal.html
example: airpassengers_temporal
dataset: airpassengers
formula: "log_y ~ 0 + f(t, model = 'rw1')"
----

First-order random walk latent field on the time index `t`; the intercept is
absorbed into the random walk (hence `0 +`).  Gaussian family.  Equivalent to a
Bayesian smoothing spline.
"""
function examples(::Val{:airp_rw1})
    return ("log_y ~ 0 + f(t, model = 'rw1')", load(Val(:airpassengers)))
end

"""
name: Airline Passengers — Random Walk (RW2) Temporal Smoothing
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-temporal.html
example: airpassengers_temporal
dataset: airpassengers
formula: "log_y ~ 0 + f(t, model = 'rw2')"
----

Second-order random walk; smoother than RW1, penalises second differences.
Equivalent to cubic spline smoothing.  Gaussian family.
"""
function examples(::Val{:airp_rw2})
    return ("log_y ~ 0 + f(t, model = 'rw2')", load(Val(:airpassengers)))
end

"""
name: Airline Passengers — AR(1) Temporal Process
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-temporal.html
example: airpassengers_temporal
dataset: airpassengers
formula: "log_y ~ 0 + f(t, model = 'ar1')"
----

Latent AR(1) process over time.  Estimates autocorrelation parameter ρ jointly
with the observation variance.  Gaussian family.
"""
function examples(::Val{:airp_ar1})
    return ("log_y ~ 0 + f(t, model = 'ar1')", load(Val(:airpassengers)))
end

"""
name: Airline Passengers — Seasonal Latent Field
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-temporal.html
example: airpassengers_temporal
dataset: airpassengers
formula: "log_y ~ 0 + f(t, model = 'seasonal', season.length = 12)"
----

Seasonal latent field with period 12 months (sum-to-zero constraint within each
year).  Captures multiplicative seasonality on the log scale.  Gaussian family.
"""
function examples(::Val{:airp_seasonal})
    return ("log_y ~ 0 + f(t, model = 'seasonal', season.length = 12)", load(Val(:airpassengers)))
end

# ── Survival ──────────────────────────────────────────────────────────────────

"""
name: VA Lung Cancer — Weibull Survival Model
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-survival.html
example: veteran_survival
dataset: veteran
formula: "inla.surv(time, status) ~ trt + celltype + karno + diagtime + age + prior"
----

`inla.surv()` wraps right-censored event times for INLA's survival families.
Weibull proportional hazards model (`family = "weibullsurv"`); the shape parameter
is estimated as a hyperparameter.
"""
function examples(::Val{:veteran_weibull})
    return (
        "inla.surv(time, status) ~ trt + celltype + karno + diagtime + age + prior",
        load(Val(:veteran)),
    )
end

# ── Nonparametric Smoothing ───────────────────────────────────────────────────

"""
name: LIDAR — Nonparametric Regression via RW2
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-smoothing.html
example: lidar_smooth
dataset: lidar
formula: "logratio ~ -1 + f(range, model = 'rw2', constr = FALSE)"
----

1D nonparametric regression using a second-order random walk latent field over
the continuous predictor `range`.  `constr = FALSE` drops the sum-to-zero
constraint; `-1` removes the fixed intercept (absorbed by the random walk).
Gaussian family.  Compare with spline-based approaches.
"""
function examples(::Val{:lidar_rw2})
    return ("logratio ~ -1 + f(range, model = 'rw2', constr = FALSE)", load(Val(:lidar)))
end

# ── Areal Spatial ──────────────────────────────────────────────────────────────

"""
name: Hospital Mortality — Binomial IID Overdispersion
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-INLA.html
example: surg_mortality
dataset: surg
formula: "r ~ f(hospital, model = 'iid')"
----

12-hospital mortality benchmark.  `f(hospital, model = "iid")` with
`family = "binomial"` (trials = `n`) shrinks small-hospital rates toward the
population mean.  Demonstrates PC priors on the precision hyperparameter.
"""
function examples(::Val{:surg_binomial})
    return ("r ~ f(hospital, model = 'iid')", load(Val(:surg)))
end

"""
name: Boston Housing — Gaussian Spatial (ICAR / Besag)
source: https://becarioprecario.bitbucket.io/inla-gitbook/ch-spatial.html
example: boston_spatial
dataset: boston_housing
formula: "log(medv) ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat + f(id, model = 'besag', graph = W)"
----

Intrinsic CAR (Besag) spatial random effect for 506 Boston census tracts.
`f(id, model = "besag", graph = W)` requires an adjacency matrix `W` (spdep
object) passed separately — it is not contained in the tabular data.
Gaussian family.  Also fitted with `"bym"` (BYM = ICAR + IID) for comparison.
"""
function examples(::Val{:boston_besag})
    return (
        "log(medv) ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat + f(id, model = 'besag', graph = W)",
        load(Val(:boston_housing)),
    )
end
