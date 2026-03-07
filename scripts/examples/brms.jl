using CSV, DataFrames, Downloads, Random, Statistics

##############################################################################
# brms example: Custom Families
# Source: https://paulbuerkner.com/brms/articles/brms_customfamilies.html
#
# Dataset: cbpp (contagious bovine pleuropneumonia) from lme4
#   Incidence of disease in herds of cattle across 4 periods.
#   56 observations: incidence count, size (herd size), period, herd ID.
#
# brms model formulas:
#   "incidence | trials(size) ~ period + (1|herd)"
#     (standard binomial GLMM)
#   "incidence | vint(size) ~ period + (1|herd)"
#     (custom beta_binomial2 family; vint passes integer auxiliary data)
##############################################################################

const CBPP_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/cbpp.csv"

load(::Val{:cbpp}) = CSV.read(Downloads.download(CBPP_URL), DataFrame)

function examples(::Val{:cbpp})
    data = load(Val(:cbpp))
    data.period = string.(data.period)  # period is an ordered factor in R
    return [
        ("incidence | trials(size) ~ period + (1|herd)", data),
        ("incidence | vint(size) ~ period + (1|herd)", data),
    ]
end

##############################################################################
# brms example: Distributional Regression Models
# Source: https://paulbuerkner.com/brms/articles/brms_distreg.html
#
# Three sub-datasets:
#
# (1) Synthetic location-scale normal data (dat1 in vignette)
#     Two groups; both mean and SD differ across groups.
#     brms model formula:
#       "bf(symptom_post ~ group, sigma ~ group)"
#
# (2) fish.csv — recreational fishing trip data (zero-inflated counts)
#     Columns: nofish, livebait, camper, persons, child, xb, zg, count
#     brms model formulas:
#       "count ~ persons + child + camper"
#         (negative binomial)
#       "bf(count ~ persons + child + camper, zi ~ child)"
#         (zero-inflated Poisson; zi = probability of structural zero)
#
# (3) Synthetic gamSim-equivalent data (smooth distributional regression)
#     Mimics mgcv::gamSim(eg=1, ...) with a group factor.
#     brms model formula:
#       "bf(y ~ s(x1) + s(x2) + (1|fac), sigma ~ s(x0) + (1|fac))"
##############################################################################

const FISH_URL = "http://paulbuerkner.com/data/fish.csv"

function load(::Val{:distreg_dat1})
    rng = MersenneTwister(1234)
    n = 100
    group = rand(rng, [0, 1], n)
    # mean shifts with group; sigma also differs between groups
    symptom_post = 0.5 .+ 0.5 .* group .+ exp.(0.4 .* group) .* randn(rng, n)
    return DataFrame(; symptom_post, group = string.(group))
end

load(::Val{:distreg_fish}) = CSV.read(Downloads.download(FISH_URL), DataFrame)

function load(::Val{:distreg_gam})
    # Synthetic equivalent of mgcv::gamSim(eg=1, n=200, scale=0.5)
    rng = MersenneTwister(42)
    n = 200
    x0 = rand(rng, n)
    x1 = rand(rng, n)
    x2 = rand(rng, n)
    fac = string.(rand(rng, 1:4, n))
    f1 = 2.0 .* sin.(π .* x1)
    f2 = exp.(2.0 .* x2)
    sigma = exp.(1.5 .* x0 .- 0.8)   # sigma varies with x0
    y = f1 .+ f2 .+ sigma .* randn(rng, n)
    return DataFrame(; y, x0, x1, x2, fac)
end

function examples(::Val{:distreg})
    dat1 = load(Val(:distreg_dat1))
    fish = load(Val(:distreg_fish))
    gam  = load(Val(:distreg_gam))
    return [
        ("bf(symptom_post ~ group, sigma ~ group)", dat1),
        ("count ~ persons + child + camper", fish),
        ("bf(count ~ persons + child + camper, zi ~ child)", fish),
        ("bf(y ~ s(x1) + s(x2) + (1|fac), sigma ~ s(x0) + (1|fac))", gam),
    ]
end

##############################################################################
# brms example: Handle Missing Values
# Source: https://paulbuerkner.com/brms/articles/brms_missings.html
#
# Dataset: nhanes from the mice R package
#   25 observations, 4 variables with planned missingness (MCAR):
#     age: age group (1–3)
#     bmi: body mass index (continuous, partially missing)
#     hyp: hypertension status (1/2, partially missing)
#     chl: cholesterol (continuous, partially missing)
#
# brms model formulas:
#   "bmi ~ age * chl"
#     (model fit to each of several multiply-imputed datasets via brm_multiple)
#   "bf(bmi | mi() ~ age * mi(chl)) + bf(chl | mi() ~ age) + set_rescor(FALSE)"
#     (joint model: simultaneously impute chl while modelling bmi)
#   "bf(bmi | mi() ~ age * mi(chl)) + bf(chl | mi(se = se_chl) ~ age) + set_rescor(FALSE)"
#     (variant: chl measured with known error se_chl; se_chl must be added to data)
##############################################################################

const NHANES_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/mice/nhanes.csv"

load(::Val{:nhanes}) = CSV.read(Downloads.download(NHANES_URL), DataFrame)

function examples(::Val{:nhanes})
    data = load(Val(:nhanes))
    return [
        ("bmi ~ age * chl", data),
        ("bf(bmi | mi() ~ age * mi(chl)) + bf(chl | mi() ~ age) + set_rescor(FALSE)", data),
        ("bf(bmi | mi() ~ age * mi(chl)) + bf(chl | mi(se = se_chl) ~ age) + set_rescor(FALSE)", data),
    ]
end

##############################################################################
# brms example: Monotonic Effects
# Source: https://paulbuerkner.com/brms/articles/brms_monotonic.html
#
# Dataset: Synthetic life satisfaction survey data
#   Simulates an individual-level dataset with:
#     ls:          life satisfaction (numeric outcome, higher = more satisfied)
#     income:      ordered factor (below_20 < 20_to_40 < 40_to_100 < greater_100)
#     income_num:  numeric coding of income level (1–4)
#     age:         age group (1–3, treated as ordinal or categorical)
#     city:        city of residence (1–5, random grouping factor)
#
# brms model formulas:
#   "ls ~ mo(income)"
#     (monotonic main effect of ordered income; mo() constrains the response
#      to be monotonically increasing or decreasing across ordered categories)
#   "ls ~ income_num"
#     (comparison: standard numeric predictor)
#   "ls ~ income"
#     (comparison: unordered factor — ignores ordinal structure)
#   "ls ~ mo(income) * age"
#     (interaction of monotonic income effect with age group)
#   "ls ~ mo(income) * age + (mo(income) | city)"
#     (with city-level random slopes for the monotonic income effect)
##############################################################################

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

function examples(::Val{:income})
    data = load(Val(:income))
    return [
        ("ls ~ mo(income)", data),
        ("ls ~ income_num", data),
        ("ls ~ income", data),
        ("ls ~ mo(income) * age", data),
        ("ls ~ mo(income) * age + (mo(income) | city)", data),
    ]
end

##############################################################################
# brms example: Multivariate Models
# Source: https://paulbuerkner.com/brms/articles/brms_multivariate.html
#
# Dataset: BTdata from the MCMCglmm R package
#   828 observations of blue tits; bivariate outcome (tarsus + back color).
#   Columns: tarsus (length), back (coloration), sex, hatchdate,
#            fosternest (foster nest ID), dam (mother ID), animal (individual ID).
#
# brms model formulas:
#   "bf(mvbind(tarsus, back) ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)"
#     (compact syntax: one formula for both responses, correlated random effects
#      via shared labels p and q; set_rescor adds residual correlations)
#   "bf(tarsus ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + bf(back ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)"
#     (equivalent explicit syntax with separate bf() per response)
#   "bf(tarsus ~ sex + s(hatchdate, k=5) + (1|p|fosternest) + (1|q|dam), family = skew_normal()) + bf(back ~ sex + s(hatchdate, k=5) + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)"
#     (extended model: skew-normal family for tarsus, smooth spline for hatchdate)
##############################################################################

const BTDATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv"

load(::Val{:btdata}) = CSV.read(Downloads.download(BTDATA_URL), DataFrame)

function examples(::Val{:btdata})
    data = load(Val(:btdata))
    return [
        ("bf(mvbind(tarsus, back) ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)", data),
        ("bf(tarsus ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + bf(back ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)", data),
        ("bf(tarsus ~ sex + s(hatchdate, k=5) + (1|p|fosternest) + (1|q|dam), family = skew_normal()) + bf(back ~ sex + s(hatchdate, k=5) + (1|p|fosternest) + (1|q|dam)) + set_rescor(TRUE)", data),
    ]
end

##############################################################################
# brms example: Nonlinear Models
# Source: https://paulbuerkner.com/brms/articles/brms_nonlinear.html
#
# Two sub-datasets:
#
# (1) Synthetic exponential-growth data
#     y = b1 * exp(b2 * x) + noise, x in [0, 2]
#     brms model formulas:
#       "bf(y ~ b1 * exp(b2 * x), b1 + b2 ~ 1, nl = TRUE)"
#         (nonlinear model; nl=TRUE enables custom nonlinear predictors;
#          b1 + b2 ~ 1 declares both as intercept-only submodels)
#       "y ~ x"
#         (comparison: standard linear model)
#
# (2) brms::loss — actuarial chain-ladder / loss development data
#     The original dataset is internal to the brms R package (not publicly hosted).
#     To export it:  data("loss", package = "brms"); write.csv(loss, "loss.csv")
#     Structure: AY (accident year), dev (development period), cum (cumulative paid
#     losses), premium (net premium). Observations form a loss triangle (55 rows).
#     Synthetic data with the same structure is generated below.
#     brms model formula:
#       "bf(cum ~ ult * (1 - exp(-(dev/theta)^omega)), ult ~ 1 + (1|AY), omega ~ 1, theta ~ 1, nl = TRUE)"
#         (nonlinear Weibull loss development model; ult is the ultimate loss,
#          allowed to vary by accident year; omega and theta are shape parameters)
##############################################################################

function load(::Val{:nonlinear_dat})
    rng = MersenneTwister(123)
    n = 100
    x = sort(rand(rng, n) .* 2)
    y = 2.0 .* exp.(-1.0 .* x) .+ 0.1 .* randn(rng, n)
    return DataFrame(; x, y)
end

function load(::Val{:loss})
    # Synthetic equivalent of brms::loss (actuarial loss triangle)
    rng = MersenneTwister(2024)
    rows = NamedTuple[]
    for ay in 1981:1990
        max_dev = 1990 - ay + 1
        premium = 10000.0 + 1000.0 * randn(rng)
        ult     = 0.65 * premium + 500.0 * randn(rng)   # ultimate expected loss
        omega   = 1.2 + 0.1 * randn(rng)
        theta   = 3.0 + 0.2 * randn(rng)
        for dev in 1:max_dev
            # Weibull cumulative development fraction with observation noise
            frac = 1.0 - exp(-((dev / theta)^omega))
            cum  = ult * frac * (1.0 + 0.02 * randn(rng))
            push!(rows, (; AY = string(ay), dev = float(dev), cum = max(0.0, cum), premium))
        end
    end
    return DataFrame(rows)
end

function examples(::Val{:nonlinear})
    dat  = load(Val(:nonlinear_dat))
    loss = load(Val(:loss))
    return [
        ("bf(y ~ b1 * exp(b2 * x), b1 + b2 ~ 1, nl = TRUE)", dat),
        ("y ~ x", dat),
        ("bf(cum ~ ult * (1 - exp(-(dev/theta)^omega)), ult ~ 1 + (1|AY), omega ~ 1, theta ~ 1, nl = TRUE)", loss),
    ]
end

##############################################################################
# brms example: Phylogenetic Models
# Source: https://paulbuerkner.com/brms/articles/brms_phylogenetics.html
#
# Four datasets (200 species each) and a shared phylogenetic tree (phylo.nex).
# All models include a phylogenetic random effect via gr(phylo, cov = A), where
# A is a covariance matrix derived from the tree using ape::vcv.phylo() in R.
# The tree file (Nexus format) is at: http://paulbuerkner.com/data/phylo.nex
# Parsing Nexus files in Julia requires e.g. PhyloNetworks.jl.
#
# (1) data_simple.txt — one observation per species
#     Columns: phen (phenotype, continuous), cofactor (continuous), phylo (species ID)
#     brms model formulas:
#       "phen ~ cofactor + (1|gr(phylo, cov = A))"
#         (phylogenetic random effect only; A is the phylogenetic cov matrix)
#       "phen ~ cofactor + (1|gr(phylo, cov = A)) + (1|phylo)"
#         (adds species-level residual random effect on top of the phylogenetic one)
#
# (2) data_repeat.txt — 5 repeated observations per species (1 000 rows total)
#     Columns: phen, cofactor, species (repeated ID), phylo (species ID)
#     brms model formula:
#       "phen ~ cofactor + (1|gr(phylo, cov = A)) + (1|species)"
#         (phylogenetic effect + within-species random effect for repeats)
#
# (3) data_effect.txt — phylogenetic meta-analysis (one effect size per species)
#     Columns: Zr (Fisher z-transformed correlation), N (sample size), phylo
#     brms model formula:
#       "Zr | se(sqrt(1/(N-3))) ~ 1 + (1|gr(phylo, cov = A))"
#         (meta-analytic model: known SE from N; intercept-only fixed effect)
#
# (4) data_pois.txt — Poisson phenotype (one count observation per species)
#     Columns: phen_pois (count response), cofactor (continuous), phylo (species ID)
#     brms model formula:
#       "phen_pois ~ cofactor + (1|gr(phylo, cov = A))"
#         (Poisson GLMM with phylogenetic random effect)
##############################################################################

const PHYLO_SIMPLE_URL = "http://paulbuerkner.com/data/data_simple.txt"
const PHYLO_REPEAT_URL = "http://paulbuerkner.com/data/data_repeat.txt"
const PHYLO_EFFECT_URL = "http://paulbuerkner.com/data/data_effect.txt"
const PHYLO_POIS_URL   = "http://paulbuerkner.com/data/data_pois.txt"

function _load_phylo_txt(url)
    CSV.read(Downloads.download(url), DataFrame; delim = ' ', ignorerepeated = true)
end

load(::Val{:phylo_simple}) = _load_phylo_txt(PHYLO_SIMPLE_URL)
load(::Val{:phylo_repeat}) = _load_phylo_txt(PHYLO_REPEAT_URL)
load(::Val{:phylo_effect}) = _load_phylo_txt(PHYLO_EFFECT_URL)
load(::Val{:phylo_pois})   = _load_phylo_txt(PHYLO_POIS_URL)

function examples(::Val{:phylogenetics})
    dat_simple = load(Val(:phylo_simple))
    dat_repeat = load(Val(:phylo_repeat))
    dat_effect = load(Val(:phylo_effect))
    dat_pois   = load(Val(:phylo_pois))
    return [
        ("phen ~ cofactor + (1|gr(phylo, cov = A))", dat_simple),
        ("phen ~ cofactor + (1|gr(phylo, cov = A)) + (1|phylo)", dat_simple),
        ("phen ~ cofactor + (1|gr(phylo, cov = A)) + (1|species)", dat_repeat),
        ("Zr | se(sqrt(1/(N-3))) ~ 1 + (1|gr(phylo, cov = A))", dat_effect),
        ("phen_pois ~ cofactor + (1|gr(phylo, cov = A))", dat_pois),
    ]
end
