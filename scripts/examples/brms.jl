using CSV, DataFrames, Downloads, Random, Statistics

const CBPP_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/cbpp.csv"

"""
name: cbpp — Contagious Bovine Pleuropneumonia
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/cbpp.csv
----

**cbpp** from the lme4 R package — 56 observations of 15 herds across 4 periods.
Columns: `incidence` (count of affected animals), `size` (herd size), `period` (1–4),
`herd` (herd ID). `period` is recoded to String in `examples()`.
"""
load(::Val{:cbpp}) = CSV.read(Downloads.download(CBPP_URL), DataFrame)

"""
name: Custom Families — Standard Binomial GLMM
source: https://paulbuerkner.com/brms/articles/brms_customfamilies.html
example: cbpp
dataset: cbpp
formula: "incidence | trials(size) ~ period + (1|herd)"
----

Standard binomial GLMM for contagious bovine pleuropneumonia incidence across 4 periods.
"""
function examples(::Val{:cbpp_binomial})
    data = load(Val(:cbpp))
    data.period = string.(data.period)
    return ("incidence | trials(size) ~ period + (1|herd)", data)
end

"""
name: Custom Families — Beta-Binomial GLMM
source: https://paulbuerkner.com/brms/articles/brms_customfamilies.html
example: cbpp
dataset: cbpp
formula: "incidence | vint(size) ~ period + (1|herd)"
----

Custom `beta_binomial2` family; `vint` passes integer auxiliary data (herd size) to the
custom family. Accounts for extra-binomial overdispersion.
"""
function examples(::Val{:cbpp_beta_binomial})
    data = load(Val(:cbpp))
    data.period = string.(data.period)
    return ("incidence | vint(size) ~ period + (1|herd)", data)
end

const FISH_URL = "http://paulbuerkner.com/data/fish.csv"

"""
name: distreg_dat1 — Synthetic Location-Scale Normal Data
source: synthetic
----

Two-group synthetic data where both mean and SD differ across groups. 100 obs.
Columns: `symptom_post` (continuous outcome), `group` (String "0"/"1"). The "hi"
group has a larger mean and larger variance.
"""
function load(::Val{:distreg_dat1})
    rng = MersenneTwister(1234)
    n = 100
    group = rand(rng, [0, 1], n)
    symptom_post = 0.5 .+ 0.5 .* group .+ exp.(0.4 .* group) .* randn(rng, n)
    return DataFrame(; symptom_post, group = string.(group))
end

"""
name: distreg_fish — Recreational Fishing Trip Data
source: http://paulbuerkner.com/data/fish.csv
----

UCLA IDRE fishing dataset; 250 fishing-trip records. Also covered in
`burkner_papers.jl` (`:fish_rj`). Columns: `nofish`, `livebait`, `camper`,
`persons`, `child`, `xb`, `zg`, `count`.
"""
load(::Val{:distreg_fish}) = CSV.read(Downloads.download(FISH_URL), DataFrame)

"""
name: distreg_gam — Synthetic Smooth Distributional Data
source: synthetic
----

Synthetic equivalent of `mgcv::gamSim(eg=1, n=200, scale=0.5)` with grouping factor.
200 obs. Columns: `y` (response), `x0` (sigma predictor), `x1`, `x2` (mean predictors),
`fac` (categorical 1–4). `sigma` varies with `x0`.
"""
function load(::Val{:distreg_gam})
    rng = MersenneTwister(42)
    n = 200
    x0 = rand(rng, n)
    x1 = rand(rng, n)
    x2 = rand(rng, n)
    fac = string.(rand(rng, 1:4, n))
    f1 = 2.0 .* sin.(π .* x1)
    f2 = exp.(2.0 .* x2)
    sigma = exp.(1.5 .* x0 .- 0.8)
    y = f1 .+ f2 .+ sigma .* randn(rng, n)
    return DataFrame(; y, x0, x1, x2, fac)
end

"""
name: Distributional Regression — Location-Scale Normal
source: https://paulbuerkner.com/brms/articles/brms_distreg.html
example: distreg
dataset: distreg_dat1
formula: "bf(symptom_post ~ group, sigma ~ group)"
----

Distributional model for both mean and log(sigma) as functions of group; demonstrates
that the two groups differ in both location and scale.
"""
function examples(::Val{:distreg_normal})
    dat1 = load(Val(:distreg_dat1))
    return ("bf(symptom_post ~ group, sigma ~ group)", dat1)
end

"""
name: Distributional Regression — Negative Binomial Counts
source: https://paulbuerkner.com/brms/articles/brms_distreg.html
example: distreg
dataset: distreg_fish
formula: "count ~ persons + child + camper"
----

Negative binomial model for fishing trip catch counts; no zero-inflation component.
"""
function examples(::Val{:distreg_nb})
    fish = load(Val(:distreg_fish))
    return ("count ~ persons + child + camper", fish)
end

"""
name: Distributional Regression — Zero-Inflated Poisson
source: https://paulbuerkner.com/brms/articles/brms_distreg.html
example: distreg
dataset: distreg_fish
formula: "bf(count ~ persons + child + camper, zi ~ child)"
----

Zero-inflated Poisson; `zi` sub-model predicts structural-zero probability from `child`.
"""
function examples(::Val{:distreg_zip})
    fish = load(Val(:distreg_fish))
    return ("bf(count ~ persons + child + camper, zi ~ child)", fish)
end

"""
name: Distributional Regression — Smooth GAM
source: https://paulbuerkner.com/brms/articles/brms_distreg.html
example: distreg
dataset: distreg_gam
formula: "bf(y ~ s(x1) + s(x2) + (1|fac), sigma ~ s(x0) + (1|fac))"
----

Both mean and log(sigma) modeled as smooth splines; shared random group effects.
"""
function examples(::Val{:distreg_gam})
    gam = load(Val(:distreg_gam))
    return ("bf(y ~ s(x1) + s(x2) + (1|fac), sigma ~ s(x0) + (1|fac))", gam)
end

const NHANES_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/mice/nhanes.csv"

"""
name: nhanes — NHANES Missing Data Subset
source: https://vincentarelbundock.github.io/Rdatasets/csv/mice/nhanes.csv
----

**nhanes** from the mice R package — 25 obs, 4 variables with planned missingness (MCAR):
`age` (group 1–3), `bmi` (partially missing), `hyp` (hypertension 1/2, partially missing),
`chl` (cholesterol, partially missing).
"""
load(::Val{:nhanes}) = CSV.read(Downloads.download(NHANES_URL), DataFrame)

"""
name: Handle Missing Values — Multiple Imputation
source: https://paulbuerkner.com/brms/articles/brms_missings.html
example: nhanes
dataset: nhanes
formula: "bmi ~ age * chl"
----

Fit to multiply-imputed datasets via `brm_multiple`; standard regression template on
complete cases from each imputed dataset.
"""
function examples(::Val{:nhanes_imputed})
    data = load(Val(:nhanes))
    return ("bmi ~ age * chl", data)
end

"""
name: Handle Missing Values — Joint Imputation Model
source: https://paulbuerkner.com/brms/articles/brms_missings.html
example: nhanes
dataset: nhanes
formula: "bf(bmi | mi() ~ age * mi(chl)) + bf(chl | mi() ~ age) + set_rescor(FALSE)"
----

Joint model: simultaneously imputes `chl` while modelling `bmi`; `mi()` marks imputed
variables; `set_rescor(FALSE)` removes residual correlations between equations.
"""
function examples(::Val{:nhanes_joint})
    data = load(Val(:nhanes))
    return ("bf(bmi | mi() ~ age * mi(chl)) + bf(chl | mi() ~ age) + set_rescor(FALSE)", data)
end

"""
name: Handle Missing Values — Measurement Error Variant
source: https://paulbuerkner.com/brms/articles/brms_missings.html
example: nhanes
dataset: nhanes
formula: "bf(bmi | mi() ~ age * mi(chl)) + bf(chl | mi(se = se_chl) ~ age) + set_rescor(FALSE)"
----

Variant: `chl` measured with known standard error `se_chl`; propagates measurement
uncertainty into the joint imputation model.
"""
function examples(::Val{:nhanes_error})
    data = load(Val(:nhanes))
    return ("bf(bmi | mi() ~ age * mi(chl)) + bf(chl | mi(se = se_chl) ~ age) + set_rescor(FALSE)", data)
end

"""
name: income — Synthetic Life Satisfaction Survey Data
source: synthetic
----

Synthetic individual-level dataset. 200 obs. Columns: `ls` (life satisfaction, numeric),
`income` (ordered factor: `below_20` < `20_to_40` < `40_to_100` < `greater_100`),
`income_num` (numeric coding 1–4), `age` (group 1–3, String), `city` (1–5, String).
"""
function load(::Val{:income})
    rng = MersenneTwister(2023)
    n = 200
    income_levels = ["below_20", "20_to_40", "40_to_100", "greater_100"]
    income_idx = rand(rng, 1:4, n)
    income = income_levels[income_idx]
    income_num = float.(income_idx)
    age = rand(rng, 1:3, n)
    city = rand(rng, 1:5, n)
    ls = income_num .+ 0.4 .* age .+ randn(rng, n)
    return DataFrame(;
        ls,
        income,
        income_num,
        age  = string.(age),
        city = string.(city),
    )
end

"""
name: Monotonic Effects — Main Effect
source: https://paulbuerkner.com/brms/articles/brms_monotonic.html
example: income
dataset: income
formula: "ls ~ mo(income)"
----

Monotonic main effect of ordered income; `mo()` constrains the response to be
monotonically increasing or decreasing across ordered categories.
"""
function examples(::Val{:income_mo})
    data = load(Val(:income))
    return ("ls ~ mo(income)", data)
end

"""
name: Monotonic Effects — Numeric Comparison
source: https://paulbuerkner.com/brms/articles/brms_monotonic.html
example: income
dataset: income
formula: "ls ~ income_num"
----

Comparison: standard numeric predictor, ignoring the ordinal structure of income.
"""
function examples(::Val{:income_num})
    data = load(Val(:income))
    return ("ls ~ income_num", data)
end

"""
name: Monotonic Effects — Nominal Factor Comparison
source: https://paulbuerkner.com/brms/articles/brms_monotonic.html
example: income
dataset: income
formula: "ls ~ income"
----

Comparison: unordered factor coding, ignores both ordinal structure and monotonicity.
"""
function examples(::Val{:income_nominal})
    data = load(Val(:income))
    return ("ls ~ income", data)
end

"""
name: Monotonic Effects — Interaction with Age
source: https://paulbuerkner.com/brms/articles/brms_monotonic.html
example: income
dataset: income
formula: "ls ~ mo(income) * age"
----

Interaction of monotonic income effect with age group.
"""
function examples(::Val{:income_mo_age})
    data = load(Val(:income))
    return ("ls ~ mo(income) * age", data)
end

"""
name: Monotonic Effects — City-Level Random Slopes
source: https://paulbuerkner.com/brms/articles/brms_monotonic.html
example: income
dataset: income
formula: "ls ~ mo(income) * age + (mo(income) | city)"
----

Adds city-level random slopes for the monotonic income effect.
"""
function examples(::Val{:income_mo_city})
    data = load(Val(:income))
    return ("ls ~ mo(income) * age + (mo(income) | city)", data)
end

const BTDATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv"

"""
name: btdata — Blue Tit Morphology Data
source: https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv
----

**BTdata** from the MCMCglmm R package — 828 observations of blue tits. Columns:
`tarsus` (tarsus length), `back` (back coloration), `sex`, `hatchdate`, `fosternest`
(foster nest ID), `dam` (mother ID), `animal` (individual ID).
"""
load(::Val{:btdata}) = CSV.read(Downloads.download(BTDATA_URL), DataFrame)

"""
name: Multivariate Models — Compact mvbind Syntax
source: https://paulbuerkner.com/brms/articles/brms_multivariate.html
example: btdata
dataset: btdata
formula: "bf(mvbind(tarsus, back) ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)"
----

Compact syntax: one formula for both responses; correlated random effects via shared
labels `p` and `q`; `set_rescor(TRUE)` adds residual correlations.
"""
function examples(::Val{:btdata_compact})
    data = load(Val(:btdata))
    return ("bf(mvbind(tarsus, back) ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)", data)
end

"""
name: Multivariate Models — Explicit Separate Formulas
source: https://paulbuerkner.com/brms/articles/brms_multivariate.html
example: btdata
dataset: btdata
formula: "bf(tarsus ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + bf(back ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)"
----

Equivalent explicit syntax with separate `bf()` per response; same model as
`:btdata_compact` but with individually specified sub-formulas.
"""
function examples(::Val{:btdata_explicit})
    data = load(Val(:btdata))
    return ("bf(tarsus ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + bf(back ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)", data)
end

"""
name: Multivariate Models — Skew-Normal with Splines
source: https://paulbuerkner.com/brms/articles/brms_multivariate.html
example: btdata
dataset: btdata
formula: "bf(tarsus ~ sex + s(hatchdate, k=5) + (1|p|fosternest) + (1|q|dam), family = skew_normal()) + bf(back ~ sex + s(hatchdate, k=5) + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)"
----

Extended model: skew-normal family for `tarsus`, smooth spline for `hatchdate` in
both sub-models.
"""
function examples(::Val{:btdata_spline})
    data = load(Val(:btdata))
    return ("bf(tarsus ~ sex + s(hatchdate, k=5) + (1|p|fosternest) + (1|q|dam), family = skew_normal()) + bf(back ~ sex + s(hatchdate, k=5) + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)", data)
end

"""
name: nonlinear_dat — Synthetic Exponential Decay Data
source: synthetic
----

100 obs; `y ≈ 2·exp(−x) + noise`, `x` ∈ [0, 2]. Columns: `x`, `y`.
Generated with `MersenneTwister(123)`.
"""
function load(::Val{:nonlinear_dat})
    rng = MersenneTwister(123)
    n = 100
    x = sort(rand(rng, n) .* 2)
    y = 2.0 .* exp.(-1.0 .* x) .+ 0.1 .* randn(rng, n)
    return DataFrame(; x, y)
end

"""
name: loss — Actuarial Loss Development Triangle
source: synthetic
----

Synthetic equivalent of `brms::loss` — actuarial chain-ladder data. The original
dataset is internal to the brms R package; to export:
`data("loss", package = "brms"); write.csv(loss, "loss.csv")`. Columns: `AY`
(accident year, 1981–1990), `dev` (development period), `cum` (cumulative paid
losses), `premium`; 55 rows forming a loss triangle.
"""
function load(::Val{:loss})
    rng = MersenneTwister(2024)
    rows = NamedTuple[]
    for ay in 1981:1990
        max_dev = 1990 - ay + 1
        premium = 10000.0 + 1000.0 * randn(rng)
        ult     = 0.65 * premium + 500.0 * randn(rng)
        omega   = 1.2 + 0.1 * randn(rng)
        theta   = 3.0 + 0.2 * randn(rng)
        for dev in 1:max_dev
            frac = 1.0 - exp(-((dev / theta)^omega))
            cum  = ult * frac * (1.0 + 0.02 * randn(rng))
            push!(rows, (; AY = string(ay), dev = float(dev), cum = max(0.0, cum), premium))
        end
    end
    return DataFrame(rows)
end

"""
name: Nonlinear Models — Exponential Decay
source: https://paulbuerkner.com/brms/articles/brms_nonlinear.html
example: nonlinear
dataset: nonlinear_dat
formula: "bf(y ~ b1 * exp(b2 * x), b1 + b2 ~ 1, nl = TRUE)"
----

Nonlinear exponential model; `nl=TRUE` enables custom nonlinear predictors;
`b1 + b2 ~ 1` declares both as intercept-only submodels.
"""
function examples(::Val{:nonlinear_exp})
    dat = load(Val(:nonlinear_dat))
    return ("bf(y ~ b1 * exp(b2 * x), b1 + b2 ~ 1, nl = TRUE)", dat)
end

"""
name: Nonlinear Models — Linear Comparison
source: https://paulbuerkner.com/brms/articles/brms_nonlinear.html
example: nonlinear
dataset: nonlinear_dat
formula: "y ~ x"
----

Standard linear model on the same exponential-decay data; baseline comparison.
"""
function examples(::Val{:nonlinear_linear})
    dat = load(Val(:nonlinear_dat))
    return ("y ~ x", dat)
end

"""
name: Nonlinear Models — Weibull Loss Development
source: https://paulbuerkner.com/brms/articles/brms_nonlinear.html
example: nonlinear
dataset: loss
formula: "bf(cum ~ ult * (1 - exp(-(dev/theta)^omega)), ult ~ 1 + (1|AY), omega ~ 1, theta ~ 1, nl = TRUE)"
----

Nonlinear Weibull loss development curve; `ult` (ultimate loss) varies by accident year;
`omega` = shape, `theta` = scale.
"""
function examples(::Val{:nonlinear_loss})
    loss = load(Val(:loss))
    return ("bf(cum ~ ult * (1 - exp(-(dev/theta)^omega)), ult ~ 1 + (1|AY), omega ~ 1, theta ~ 1, nl = TRUE)", loss)
end

const PHYLO_SIMPLE_URL = "http://paulbuerkner.com/data/data_simple.txt"
const PHYLO_REPEAT_URL = "http://paulbuerkner.com/data/data_repeat.txt"
const PHYLO_EFFECT_URL = "http://paulbuerkner.com/data/data_effect.txt"
const PHYLO_POIS_URL   = "http://paulbuerkner.com/data/data_pois.txt"

function _load_phylo_txt(url)
    CSV.read(Downloads.download(url), DataFrame; delim = ' ', ignorerepeated = true)
end

"""
name: phylo_simple — Phylogenetic Data, One Obs per Species
source: http://paulbuerkner.com/data/data_simple.txt
----

200 species; one observation each. Columns: `phen` (phenotype, continuous), `cofactor`
(continuous), `phylo` (species ID). Used with a phylogenetic covariance matrix `A`
derived from a Nexus tree via `ape::vcv.phylo()` in R (tree at
`http://paulbuerkner.com/data/phylo.nex`; parse with e.g. PhyloNetworks.jl).
"""
load(::Val{:phylo_simple}) = _load_phylo_txt(PHYLO_SIMPLE_URL)

"""
name: phylo_repeat — Phylogenetic Data, Repeated Observations
source: http://paulbuerkner.com/data/data_repeat.txt
----

200 species × 5 repeated observations each (1000 rows total). Columns: `phen`,
`cofactor`, `species` (repeated ID), `phylo` (species ID).
"""
load(::Val{:phylo_repeat}) = _load_phylo_txt(PHYLO_REPEAT_URL)

"""
name: phylo_effect — Phylogenetic Meta-Analysis Data
source: http://paulbuerkner.com/data/data_effect.txt
----

200 species; one effect size per species. Columns: `Zr` (Fisher z-transformed
correlation), `N` (sample size), `phylo` (species ID).
"""
load(::Val{:phylo_effect}) = _load_phylo_txt(PHYLO_EFFECT_URL)

"""
name: phylo_pois — Phylogenetic Poisson Count Data
source: http://paulbuerkner.com/data/data_pois.txt
----

200 species; one count observation per species. Columns: `phen_pois` (count response),
`cofactor` (continuous), `phylo` (species ID).
"""
load(::Val{:phylo_pois}) = _load_phylo_txt(PHYLO_POIS_URL)

"""
name: Phylogenetic Models — Simple Random Effect
source: https://paulbuerkner.com/brms/articles/brms_phylogenetics.html
example: phylogenetics
dataset: phylo_simple
formula: "phen ~ cofactor + (1|gr(phylo, cov = A))"
----

Phylogenetic random effect only; `A` is the phylogenetic covariance matrix.
"""
function examples(::Val{:phylo_simple_re})
    dat = load(Val(:phylo_simple))
    return ("phen ~ cofactor + (1|gr(phylo, cov = A))", dat)
end

"""
name: Phylogenetic Models — Additional Residual Effect
source: https://paulbuerkner.com/brms/articles/brms_phylogenetics.html
example: phylogenetics
dataset: phylo_simple
formula: "phen ~ cofactor + (1|gr(phylo, cov = A)) + (1|phylo)"
----

Adds a species-level residual random effect on top of the phylogenetic effect;
`(1|phylo)` captures species-specific deviations not explained by phylogeny.
"""
function examples(::Val{:phylo_simple_resid})
    dat = load(Val(:phylo_simple))
    return ("phen ~ cofactor + (1|gr(phylo, cov = A)) + (1|phylo)", dat)
end

"""
name: Phylogenetic Models — Repeated Observations
source: https://paulbuerkner.com/brms/articles/brms_phylogenetics.html
example: phylogenetics
dataset: phylo_repeat
formula: "phen ~ cofactor + (1|gr(phylo, cov = A)) + (1|species)"
----

Phylogenetic effect plus within-species random effect for 5 repeated observations
per species.
"""
function examples(::Val{:phylo_repeat_re})
    dat = load(Val(:phylo_repeat))
    return ("phen ~ cofactor + (1|gr(phylo, cov = A)) + (1|species)", dat)
end

"""
name: Phylogenetic Models — Meta-Analysis
source: https://paulbuerkner.com/brms/articles/brms_phylogenetics.html
example: phylogenetics
dataset: phylo_effect
formula: "Zr | se(sqrt(1/(N-3))) ~ 1 + (1|gr(phylo, cov = A))"
----

Meta-analytic model; known SE derived from `N`; `se()` passes known measurement error;
random study intercept captures between-species heterogeneity.
"""
function examples(::Val{:phylo_effect})
    dat = load(Val(:phylo_effect))
    return ("Zr | se(sqrt(1/(N-3))) ~ 1 + (1|gr(phylo, cov = A))", dat)
end

"""
name: Phylogenetic Models — Poisson Family
source: https://paulbuerkner.com/brms/articles/brms_phylogenetics.html
example: phylogenetics
dataset: phylo_pois
formula: "phen_pois ~ cofactor + (1|gr(phylo, cov = A))"
----

Poisson GLMM with phylogenetic random effect; models count phenotype data.
"""
function examples(::Val{:phylo_pois})
    dat = load(Val(:phylo_pois))
    return ("phen_pois ~ cofactor + (1|gr(phylo, cov = A))", dat)
end
