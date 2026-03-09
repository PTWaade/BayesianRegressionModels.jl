using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: rstanarm (Stan-based Bayesian regression)
# Sources: https://mc-stan.org/rstanarm/articles/
# Package: https://github.com/stan-dev/rstanarm
#
# rstanarm provides Stan-based Bayesian equivalents of standard R modelling
# functions (lm, glm, lmer, glmer, polr, betareg, etc.) with weakly-informative
# default priors and optional autoscaling.  Formulas follow standard R/lme4 syntax.

const MTCARS_URL     = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv"
const WOMENSROLE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/womensrole.csv"
const CLOUDS_URL     = "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/clouds.csv"
const KIDIQ_URL      = "https://vincentarelbundock.github.io/Rdatasets/csv/rstanarm/kidiq.csv"
const WELLS_URL      = "https://vincentarelbundock.github.io/Rdatasets/csv/rstanarm/wells.csv"
const ROACHES_URL    = "https://vincentarelbundock.github.io/Rdatasets/csv/rstanarm/roaches.csv"
const WEIGHTGAIN_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/weightgain.csv"
const ESOPH_URL      = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/esoph.csv"
const GASOLINE_URL   = "https://vincentarelbundock.github.io/Rdatasets/csv/betareg/GasolineYield.csv"
const CBPP_RS_URL    = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/cbpp.csv"
const BBALL1970_URL  = "https://vincentarelbundock.github.io/Rdatasets/csv/rstanarm/bball1970.csv"

# ── Datasets ──────────────────────────────────────────────────────────────────

"""
name: mtcars — Motor Trend Car Road Tests
source: https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv
----

Classic 1974 Motor Trend dataset; 32 automobiles, 11 variables.
Columns used: `mpg` (miles per gallon), `wt` (weight in 1000 lb),
`am` (transmission: 0 = automatic, 1 = manual).
"""
load(::Val{:mtcars}) = CSV.read(Downloads.download(MTCARS_URL), DataFrame)

"""
name: womensrole — HSAUR Women's Role Survey
source: https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/womensrole.csv
----

1974–1975 British survey of attitudes toward women in employment; 42 rows
(education × sex combinations). Columns: `education` (years, 0–7 grouped),
`sex` (Male/Female), `agree` (count who agreed), `disagree` (count who disagreed).
"""
load(::Val{:womensrole}) = CSV.read(Downloads.download(WOMENSROLE_URL), DataFrame)

"""
name: clouds — Cloud Seeding Experiment (HSAUR)
source: https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/clouds.csv
----

Florida cloud-seeding experiment; 24 observations.
Columns: `rainfall` (acre-feet), `seeding` (yes/no), `time` (time since seeding),
`cloudcover`, `prewetness`, `echomotion`, `sne` (suitability criterion).
"""
load(::Val{:clouds}) = CSV.read(Downloads.download(CLOUDS_URL), DataFrame)

"""
name: kidiq — Child IQ and Maternal Variables (rstanarm)
source: https://vincentarelbundock.github.io/Rdatasets/csv/rstanarm/kidiq.csv
----

434 children from the NLSY child supplement (Gelman & Hill 2007).
Columns: `kid_score` (child cognitive test score), `mom_hs` (mother completed
high school: 0/1), `mom_iq` (mother IQ score), `mom_work` (mother work status),
`mom_age`.
"""
load(::Val{:kidiq}) = CSV.read(Downloads.download(KIDIQ_URL), DataFrame)

"""
name: wells — Bangladesh Arsenic Well Switching (rstanarm)
source: https://vincentarelbundock.github.io/Rdatasets/csv/rstanarm/wells.csv
----

Survey of 3,020 Bangladeshi households near unsafe arsenic wells (Gelman & Hill 2007).
Columns: `switch` (1 = switched to a safe well), `dist` (distance to nearest safe well,
metres), `arsenic` (arsenic level of current well, hundreds of micrograms/litre),
`assoc` (community association membership), `educ` (years of education).
A `dist100 = dist/100` column is added by `load()` for scale-compatible modelling.
"""
function load(::Val{:wells})
    data = CSV.read(Downloads.download(WELLS_URL), DataFrame)
    data.dist100 = data.dist ./ 100
    return data
end

"""
name: roaches — Urban Pest Management (rstanarm)
source: https://vincentarelbundock.github.io/Rdatasets/csv/rstanarm/roaches.csv
----

262 urban apartments in a pest management study (Gelman & Hill 2007).
Columns: `y` (post-treatment roach count in traps), `roach1` (pre-treatment count),
`treatment` (1 = received pest management), `senior` (1 = senior housing),
`exposure2` (number of days traps were exposed; used as offset `log(exposure2)`).
"""
load(::Val{:roaches}) = CSV.read(Downloads.download(ROACHES_URL), DataFrame)

"""
name: weightgain — Rat Weight Gain Factorial Experiment (HSAUR)
source: https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/weightgain.csv
----

40 rats in a 2×2 factorial design (HSAUR3).
Columns: `source` (protein source: Beef/Cereal), `type` (protein level: High/Low),
`weightgain` (weight gain in grams).
"""
load(::Val{:weightgain}) = CSV.read(Downloads.download(WEIGHTGAIN_URL), DataFrame)

"""
name: esoph — Esophageal Cancer Case-Control Study
source: https://vincentarelbundock.github.io/Rdatasets/csv/datasets/esoph.csv
----

88-group case-control study of esophageal cancer (Breslow & Day 1980).
Columns: `agegp` (age group, ordered), `alcgp` (alcohol consumption group, ordered),
`tobgp` (tobacco consumption group, ordered), `ncases`, `ncontrols`.
"""
load(::Val{:esoph}) = CSV.read(Downloads.download(ESOPH_URL), DataFrame)

"""
name: GasolineYield — Crude Oil Distillation to Gasoline
source: https://vincentarelbundock.github.io/Rdatasets/csv/betareg/GasolineYield.csv
----

32 batches of crude oil distilled to gasoline (Prater 1956 via betareg).
Columns: `yield` (proportion of crude converted, 0–1), `gravity` (API gravity),
`pressure` (vapor pressure, psi), `temp10` through `temp35` (temperatures at
which 10–35 % has vaporized), `batch` (factor, 1–10).
"""
load(::Val{:gasoline_yield}) = CSV.read(Downloads.download(GASOLINE_URL), DataFrame)

"""
name: cbpp_rs — CBPP Herd Disease Incidence (lme4)
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/cbpp.csv
----

Contagious bovine pleuropneumonia in 15 herds across 4 periods; 56 rows.
Columns: `incidence` (affected animals), `size` (herd size), `period` (1–4),
`herd`. `period` is recoded to String in `examples()`.
"""
load(::Val{:cbpp_rs}) = CSV.read(Downloads.download(CBPP_RS_URL), DataFrame)

"""
name: clotting — Blood Clotting Time (Dobson 2002)
source: synthetic
----

18-observation Gamma regression example from Dobson (2002) §9.2, as used in the
rstanarm *continuous* vignette. Two thromboplastin lots (`lot_id` 1 and 2),
each tested at 9 plasma dilution levels.
Columns: `log_plasma` (log plasma concentration), `clot_time` (clotting time),
`lot_id` (factor: "1" / "2").
"""
function load(::Val{:clotting})
    log_plasma = repeat([5, 10, 15, 20, 30, 40, 60, 80, 100], 2)
    clot_time  = [118, 58, 42, 35, 27, 25, 21, 19, 18, 69, 35, 26, 21, 18, 16, 13, 12, 12]
    lot_id     = vcat(fill("1", 9), fill("2", 9))
    return DataFrame(; log_plasma, clot_time, lot_id)
end

"""
name: bball1970 — 1970 MLB Season Batting (rstanarm)
source: https://vincentarelbundock.github.io/Rdatasets/csv/rstanarm/bball1970.csv
----

18 major-league players' batting records from the first 45 at-bats of the 1970
season (Efron & Morris 1975 via rstanarm *pooling* vignette).
Columns: `Player` (player name), `Hits` (hits in first 45 AB), `AB` (= 45),
`RemainingHits`, `RemainingAB`, `BattingAverage`, `RemainingAverage`.
"""
load(::Val{:bball1970}) = CSV.read(Downloads.download(BBALL1970_URL), DataFrame)

# ── Priors / Getting-Started: mtcars ──────────────────────────────────────────

"""
name: Motor Trend Cars — Gaussian Linear Model
source: https://mc-stan.org/rstanarm/articles/priors.html
example: mtcars
dataset: mtcars
formula: "mpg ~ wt + am"
----

`stan_glm` with default weakly-informative priors (autoscaled Normal).
Demonstrates how rstanarm's prior autoscaling adapts to predictor scale.
Gaussian family, identity link.
"""
function examples(::Val{:mtcars_lm})
    return ("mpg ~ wt + am", load(Val(:mtcars)))
end

# ── Regularized Linear Model: clouds ──────────────────────────────────────────

"""
name: Cloud Seeding — Regularized Linear Model (R² Prior)
source: https://mc-stan.org/rstanarm/articles/lm.html
example: clouds
dataset: clouds
formula: "rainfall ~ seeding * (sne + cloudcover + prewetness + echomotion) + time"
----

`stan_lm` with an R² prior on explained variance.  Full interaction model of
cloud-seeding conditions.  Gaussian family, identity link.
"""
function examples(::Val{:clouds_lm})
    return (
        "rainfall ~ seeding * (sne + cloudcover + prewetness + echomotion) + time",
        load(Val(:clouds)),
    )
end

# ── Women's Role: logistic binomial ───────────────────────────────────────────

"""
name: Women's Role Survey — Logistic Regression
source: https://mc-stan.org/rstanarm/articles/rstanarm.html
example: womensrole
dataset: womensrole
formula: "cbind(agree, disagree) ~ education + sex"
----

`stan_glm` binomial model of survey agreement with gender-role statements as a
function of education and respondent sex.  Grouped binomial with logit link.
"""
function examples(::Val{:womensrole_logit})
    return ("cbind(agree, disagree) ~ education + sex", load(Val(:womensrole)))
end

"""
name: Women's Role Survey — Logistic with Quadratic Education
source: https://mc-stan.org/rstanarm/articles/rstanarm.html
example: womensrole
dataset: womensrole
formula: "cbind(agree, disagree) ~ education + I(education^2) + sex"
----

Extends the linear education model with a quadratic term to capture diminishing
effects at higher education levels.  Binomial / logit.
"""
function examples(::Val{:womensrole_quad})
    return ("cbind(agree, disagree) ~ education + I(education^2) + sex", load(Val(:womensrole)))
end

# ── Continuous / kidiq ────────────────────────────────────────────────────────

"""
name: Child IQ — Maternal High-School Status
source: https://mc-stan.org/rstanarm/articles/continuous.html
example: kidiq
dataset: kidiq
formula: "kid_score ~ mom_hs"
----

Baseline Gaussian model regressing child cognitive score on whether the mother
completed high school.  Demonstrates default weakly-informative priors.
"""
function examples(::Val{:kidiq_hs})
    return ("kid_score ~ mom_hs", load(Val(:kidiq)))
end

"""
name: Child IQ — Maternal IQ
source: https://mc-stan.org/rstanarm/articles/continuous.html
example: kidiq
dataset: kidiq
formula: "kid_score ~ mom_iq"
----

Regresses child test score on continuous maternal IQ; single continuous predictor.
Gaussian / identity.
"""
function examples(::Val{:kidiq_iq})
    return ("kid_score ~ mom_iq", load(Val(:kidiq)))
end

"""
name: Child IQ — Maternal High-School Status and IQ
source: https://mc-stan.org/rstanarm/articles/continuous.html
example: kidiq
dataset: kidiq
formula: "kid_score ~ mom_hs + mom_iq"
----

Additive model combining both maternal predictors.  Gaussian / identity.
"""
function examples(::Val{:kidiq_both})
    return ("kid_score ~ mom_hs + mom_iq", load(Val(:kidiq)))
end

"""
name: Child IQ — Maternal High-School × IQ Interaction
source: https://mc-stan.org/rstanarm/articles/continuous.html
example: kidiq
dataset: kidiq
formula: "kid_score ~ mom_hs * mom_iq"
----

Full interaction; the slope of maternal IQ on child score is allowed to differ
by high-school completion status.  Gaussian / identity.
"""
function examples(::Val{:kidiq_interaction})
    return ("kid_score ~ mom_hs * mom_iq", load(Val(:kidiq)))
end

# ── Continuous / clotting: Gamma regression ───────────────────────────────────

"""
name: Blood Clotting Time — Gamma GLM
source: https://mc-stan.org/rstanarm/articles/continuous.html
example: clotting
dataset: clotting
formula: "clot_time ~ log_plasma * lot_id"
----

`stan_glm` with Gamma family and inverse link, modelling blood clotting time as
a function of log plasma concentration, thromboplastin lot, and their interaction.
Classic Dobson (2002) §9.2 example.
"""
function examples(::Val{:clotting_gamma})
    return ("clot_time ~ log_plasma * lot_id", load(Val(:clotting)))
end

# ── Binomial / wells ──────────────────────────────────────────────────────────

"""
name: Bangladesh Wells — Logistic (Distance Only)
source: https://mc-stan.org/rstanarm/articles/binomial.html
example: wells
dataset: wells
formula: "switch ~ dist100"
----

`stan_glm` logistic regression of household well-switching on distance to the
nearest safe well (in units of 100 metres).  Binomial / logit.
"""
function examples(::Val{:wells_dist})
    return ("switch ~ dist100", load(Val(:wells)))
end

"""
name: Bangladesh Wells — Logistic (Distance + Arsenic)
source: https://mc-stan.org/rstanarm/articles/binomial.html
example: wells
dataset: wells
formula: "switch ~ dist100 + arsenic"
----

Adds arsenic level of the current well as a second predictor for switching.
Binomial / logit.
"""
function examples(::Val{:wells_both})
    return ("switch ~ dist100 + arsenic", load(Val(:wells)))
end

# ── Count / roaches ───────────────────────────────────────────────────────────

"""
name: Urban Roaches — Poisson Count Model
source: https://mc-stan.org/rstanarm/articles/count.html
example: roaches
dataset: roaches
formula: "y ~ roach1 + treatment + senior + offset(log(exposure2))"
----

`stan_glm` Poisson regression for post-treatment roach trap counts, controlling
for pre-treatment baseline and senior-housing status.  Trap-exposure days enter
as a log offset.  Poisson / log.
"""
function examples(::Val{:roaches_poisson})
    return ("y ~ roach1 + treatment + senior + offset(log(exposure2))", load(Val(:roaches)))
end

"""
name: Urban Roaches — Negative Binomial Count Model
source: https://mc-stan.org/rstanarm/articles/count.html
example: roaches
dataset: roaches
formula: "y ~ roach1 + treatment + senior + offset(log(exposure2))"
----

Same predictors as `:roaches_poisson` but with the negative binomial family to
accommodate overdispersion and excess zeros.  `stan_glm.nb` / log.
"""
function examples(::Val{:roaches_nb})
    return ("y ~ roach1 + treatment + senior + offset(log(exposure2))", load(Val(:roaches)))
end

# ── ANOVA / weightgain ────────────────────────────────────────────────────────

"""
name: Rat Weight Gain — Bayesian ANOVA (R² Prior)
source: https://mc-stan.org/rstanarm/articles/aov.html
example: weightgain
dataset: weightgain
formula: "weightgain ~ source * type"
----

`stan_aov` two-way factorial model with R² prior on variance explained by the
protein source × protein level interaction.  Gaussian / identity.
"""
function examples(::Val{:weightgain_anova})
    return ("weightgain ~ source * type", load(Val(:weightgain)))
end

"""
name: Rat Weight Gain — Factors as Random Effects
source: https://mc-stan.org/rstanarm/articles/aov.html
example: weightgain
dataset: weightgain
formula: "weightgain ~ 1 + (1|source) + (1|type) + (1|source:type)"
----

`stan_lmer` refit treating the factorial structure as nested random intercepts,
illustrating partial pooling across factor levels.  Gaussian / identity.
"""
function examples(::Val{:weightgain_re})
    return ("weightgain ~ 1 + (1|source) + (1|type) + (1|source:type)", load(Val(:weightgain)))
end

# ── Ordinal regression / esoph ────────────────────────────────────────────────

"""
name: Esophageal Cancer — Ordinal Logistic Regression
source: https://mc-stan.org/rstanarm/articles/polr.html
example: esoph
dataset: esoph
formula: "tobgp ~ agegp + alcgp"
----

`stan_polr` proportional-odds model of tobacco consumption group as a function
of age and alcohol group.  Ordered logistic.
"""
function examples(::Val{:esoph_polr})
    return ("tobgp ~ agegp + alcgp", load(Val(:esoph)))
end

# ── Beta regression / GasolineYield ──────────────────────────────────────────

"""
name: Gasoline Yield — Beta Regression (Mean Only)
source: https://mc-stan.org/rstanarm/articles/betareg.html
example: gasoline_yield
dataset: gasoline_yield
formula: "yield ~ gravity + pressure + temp10 + batch"
----

`stan_betareg` for proportion outcome (0–1): fraction of crude oil converted to
gasoline as a function of API gravity, vapor pressure, vaporization temperature,
and batch factor.  Beta / logit (mean), constant precision.
"""
function examples(::Val{:gasoline_beta})
    return ("yield ~ gravity + pressure + temp10 + batch", load(Val(:gasoline_yield)))
end

"""
name: Gasoline Yield — Distributional Beta Regression
source: https://mc-stan.org/rstanarm/articles/betareg.html
example: gasoline_yield
dataset: gasoline_yield
formula: "yield ~ gravity + pressure + temp10 + batch | gravity + pressure"
----

Extends `:gasoline_beta` by modelling precision (`phi`) as a function of gravity
and pressure.  The `|` separator follows the betareg/rstanarm distributional
formula interface.  Beta / logit (mean) + log (precision).
"""
function examples(::Val{:gasoline_beta_distr})
    return (
        "yield ~ gravity + pressure + temp10 + batch | gravity + pressure",
        load(Val(:gasoline_yield)),
    )
end

# ── GLMM / cbpp ───────────────────────────────────────────────────────────────

"""
name: Bovine Pleuropneumonia — Binomial GLMM
source: https://mc-stan.org/rstanarm/articles/glmer.html
example: cbpp_glmm
dataset: cbpp_rs
formula: "cbind(incidence, size - incidence) ~ size + period + (1|herd)"
----

Canonical `stan_glmer` example from the rstanarm *mixed effects* vignette.
Herd-level random intercept for disease incidence across 4 time periods.
Binomial / logit.
"""
function examples(::Val{:cbpp_glmm})
    data = load(Val(:cbpp_rs))
    data.period = string.(data.period)
    return ("cbind(incidence, size - incidence) ~ size + period + (1|herd)", data)
end

# ── Partial Pooling / batting ─────────────────────────────────────────────────

"""
name: MLB Batting 1970 — Complete Pooling
source: https://mc-stan.org/rstanarm/articles/pooling.html
example: bball_pooling
dataset: bball1970
formula: "cbind(Hits, AB - Hits) ~ 1"
----

Complete-pooling estimate: single shared batting-average parameter for all 18
players.  Binomial / logit.  Compare with `:bball_nopooling` and `:bball_partial`.
"""
function examples(::Val{:bball_pooled})
    return ("cbind(Hits, AB - Hits) ~ 1", load(Val(:bball1970)))
end

"""
name: MLB Batting 1970 — No Pooling
source: https://mc-stan.org/rstanarm/articles/pooling.html
example: bball_pooling
dataset: bball1970
formula: "cbind(Hits, AB - Hits) ~ 0 + Player"
----

No-pooling model: independent intercept per player, no shrinkage.
Binomial / logit.
"""
function examples(::Val{:bball_nopooling})
    return ("cbind(Hits, AB - Hits) ~ 0 + Player", load(Val(:bball1970)))
end

"""
name: MLB Batting 1970 — Partial Pooling (Hierarchical)
source: https://mc-stan.org/rstanarm/articles/pooling.html
example: bball_pooling
dataset: bball1970
formula: "cbind(Hits, AB - Hits) ~ (1|Player)"
----

Hierarchical shrinkage model: player-level random intercepts partially pool
estimates toward the population mean.  `stan_glmer` / Binomial / logit.
The rstanarm *pooling* vignette shows this recovers end-of-season averages better
than either complete or no pooling.
"""
function examples(::Val{:bball_partial})
    return ("cbind(Hits, AB - Hits) ~ (1|Player)", load(Val(:bball1970)))
end
