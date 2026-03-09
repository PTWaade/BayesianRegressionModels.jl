using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: MCMCglmm (MCMC Generalised Linear Mixed Models)
# Sources: https://cran.r-project.org/web/packages/MCMCglmm/vignettes/Overview.pdf
#          https://jarrodhadfield.github.io/MCMCglmm/course-notes/
# Paper:   Hadfield (2010) Journal of Statistical Software 33(2)
# Package: https://cran.r-project.org/package=MCMCglmm
#
# MCMCglmm uses separate fixed, random, and rcov formula arguments.
# Formula notation here concatenates them as semicolon-labelled sub-formulas:
#   "fixed: y ~ x; random: ~group; rcov: ~units"
# The rcov part is omitted when it is the default ~units (diagonal residual).
#
# Random effect covariance structures use MCMCglmm notation:
#   ~us(1 + x):group  — unstructured (intercept + slope) per group
#   ~idh(trait):group — diagonal, heterogeneous variances per trait
#   ~us(trait):units  — unstructured residual across traits (multi-response)
#
# Multi-response models use cbind(y1, y2) ~ trait - 1 on the fixed side;
# trait is a reserved variable in MCMCglmm naming the response column.
#
# Animal models pass a pedigree (kinship) matrix via ginverse = list(animal = Ainv);
# the tabular data only needs an `animal` column with individual IDs.

const BTDATA_URL     = "https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv"
const BTPED_URL      = "https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTped.csv"
const PLODIARO_URL   = "https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/PlodiaPO.csv"
const CHICKWT_URL    = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ChickWeight.csv"
const PBCSEQ_URL     = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv"
const SSHORNS_URL    = "https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/SShorns.csv"

# ── Datasets ──────────────────────────────────────────────────────────────────

"""
name: BTdata — Blue Tit Morphology (MCMCglmm)
source: https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv
----

828 blue tit chicks from Hadfield et al. (2007 JEB) via MCMCglmm.
Columns: `tarsus` (tarsus length, mm), `back` (back colour brightness),
`animal` (individual ID, links to BTped pedigree), `dam` (mother ID),
`fosternest` (rearing nest ID), `hatchdate` (Julian day), `sex`
(Fem/Male/UNK).
"""
load(::Val{:btdata}) = CSV.read(Downloads.download(BTDATA_URL), DataFrame)

"""
name: BTped — Blue Tit Pedigree (MCMCglmm)
source: https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTped.csv
----

Pedigree for the BTdata blue tit population; 1,040 rows.
Columns: `animal` (individual ID), `dam` (mother ID, NA = unknown),
`sire` (father ID, NA = unknown).
Used via `ginverse = list(animal = Ainv)` where `Ainv` is the inverse
numerator relationship matrix from `MCMCglmm::inverseA(BTped)`.
"""
load(::Val{:btped}) = CSV.read(Downloads.download(BTPED_URL), DataFrame)

"""
name: PlodiaPO — Phenoloxidase in Indian Meal Moth (MCMCglmm)
source: https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/PlodiaPO.csv
----

511 *Plodia interpunctella* caterpillars assayed for phenoloxidase (an immune
enzyme).  Columns: `PO` (phenoloxidase activity, continuous), `FSfamily`
(full-sib family ID), `plate` (assay plate ID).
"""
load(::Val{:plodiaPO}) = CSV.read(Downloads.download(PLODIARO_URL), DataFrame)

"""
name: ChickWeight — Chick Growth Under Four Diets (base R)
source: https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ChickWeight.csv
----

578 observations of 50 chicks weighed every 2 days from hatching to day 21.
Columns: `weight` (g), `Time` (day 0–21), `Chick` (chick ID 1–50),
`Diet` (factor 1–4). `chick_id` (1–50) added by `load()` for numeric indexing.
"""
function load(::Val{:chickweight})
    data = CSV.read(Downloads.download(CHICKWT_URL), DataFrame)
    chicks = sort(unique(data.Chick))
    id_map = Dict(c => i for (i, c) in enumerate(chicks))
    data.chick_id = [id_map[c] for c in data.Chick]
    data.Diet = string.(data.Diet)
    return data
end

"""
name: pbcseq — Primary Biliary Cirrhosis Longitudinal Data (survival)
source: https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv
----

1,945 repeated-measures observations on 312 PBC patients from the Mayo Clinic.
Columns: `id`, `futime` (follow-up days), `status` (0=alive/transplant, 1=dead),
`trt` (1=D-penicillamine, 2=placebo), `age`, `sex` (f/m), `bili` (bilirubin,
mg/dL), `albumin`, `alk.phos`, `ast`, `platelet`, `protime`, `ascites`, `hepato`,
`spiders`, `edema`, `stage`. `log_bili` = log(bili) added by `load()`.
"""
function load(::Val{:pbcseq})
    data = CSV.read(Downloads.download(PBCSEQ_URL), DataFrame)
    data.log_bili = log.(data.bili)
    return data
end

"""
name: SShorns — Soay Sheep Horn Morphology (MCMCglmm)
source: https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/SShorns.csv
----

666 Soay sheep with horn morphology records.
Columns: `id` (individual ID), `horn` (ordered factor: polled < scurred < normal),
`sex` (F/M).  Used for ordinal (threshold) regression of horn type.
"""
load(::Val{:sshorns}) = CSV.read(Downloads.download(SSHORNS_URL), DataFrame)

# ── Univariate Gaussian GLMM ──────────────────────────────────────────────────

"""
name: Phenoloxidase — Gaussian GLMM with Full-Sib Family Random Effect
source: https://cran.r-project.org/web/packages/MCMCglmm/vignettes/Overview.pdf
example: plodiaPO_iid
dataset: plodiaPO
formula: "fixed: PO ~ 1; random: ~FSfamily"
----

Intercept-only fixed effects; `~FSfamily` random intercept partitions phenoloxidase
variance into between-family and within-family components.  Gaussian family.
Textbook introductory example from the MCMCglmm reference manual.
"""
function examples(::Val{:plodia_iid})
    return ("fixed: PO ~ 1; random: ~FSfamily", load(Val(:plodiaPO)))
end

# ── Blue Tit Mixed Models ─────────────────────────────────────────────────────

"""
name: Blue Tit Tarsus — Gaussian GLMM with Foster-Nest Random Intercept
source: https://jarrodhadfield.github.io/MCMCglmm/course-notes/
example: btdata_iid
dataset: btdata
formula: "fixed: tarsus ~ sex + hatchdate; random: ~fosternest"
----

Simple random intercept for rearing nest; estimates shared nest environment
contribution to tarsus length.  Gaussian family.
"""
function examples(::Val{:btdata_iid})
    return ("fixed: tarsus ~ sex + hatchdate; random: ~fosternest", load(Val(:btdata)))
end

"""
name: Blue Tit Tarsus — Animal Model (Additive Genetic Variance)
source: https://jarrodhadfield.github.io/MCMCglmm/course-notes/
example: btdata_animal
dataset: btdata
formula: "fixed: tarsus ~ sex + hatchdate; random: ~fosternest + animal"
----

The canonical MCMCglmm animal model: `animal` random effect with the inverse
numerator relationship matrix `Ainv` (from `BTped` via `inverseA()`) separates
additive genetic variance from shared rearing-environment variance.
`ginverse = list(animal = Ainv)` must be passed at fitting time.  Gaussian family.
"""
function examples(::Val{:btdata_animal})
    return ("fixed: tarsus ~ sex + hatchdate; random: ~fosternest + animal", load(Val(:btdata)))
end

# ── Random Regression ─────────────────────────────────────────────────────────

"""
name: Chick Growth — Random Regression (Unstructured G-matrix)
source: https://jarrodhadfield.github.io/MCMCglmm/course-notes/
example: chickweight_rslope
dataset: chickweight
formula: "fixed: weight ~ Time + I(Time^2) + Diet:Time; random: ~us(1 + Time):Chick"
----

Individual chicks vary in both baseline weight and growth rate.
`~us(1 + Time):Chick` fits a 2×2 unstructured G-matrix per chick capturing the
(co)variance of intercepts and slopes.  Gaussian family.
"""
function examples(::Val{:chickweight_rslope})
    return (
        "fixed: weight ~ Time + I(Time^2) + Diet:Time; random: ~us(1 + Time):Chick",
        load(Val(:chickweight)),
    )
end

# ── Multi-Response Models ──────────────────────────────────────────────────────

"""
name: Blue Tit Bivariate — Multivariate Gaussian Animal Model
source: https://jarrodhadfield.github.io/MCMCglmm/course-notes/
example: btdata_bivariate
dataset: btdata
formula: "fixed: cbind(tarsus, back) ~ trait - 1 + trait:sex + trait:hatchdate; random: ~us(trait):fosternest + us(trait):animal; rcov: ~us(trait):units"
----

Joint model for tarsus length and back colour.  `trait - 1` gives one intercept
per response; `us(trait):animal` estimates a 2×2 genetic covariance matrix across
traits.  Allows estimating genetic correlations between morphological traits.
`ginverse = list(animal = Ainv)` required.  Gaussian/Gaussian family.
"""
function examples(::Val{:btdata_bivariate})
    return (
        "fixed: cbind(tarsus, back) ~ trait - 1 + trait:sex + trait:hatchdate; random: ~us(trait):fosternest + us(trait):animal; rcov: ~us(trait):units",
        load(Val(:btdata)),
    )
end

"""
name: PBC — Bivariate Gaussian + Threshold Joint Model
source: https://jarrodhadfield.github.io/MCMCglmm/course-notes/
example: pbcseq_bivariate
dataset: pbcseq
formula: "fixed: cbind(log_bili, ascites) ~ trait - 1 + trait:(age + sex); random: ~us(trait):id; rcov: ~us(trait):units"
----

Joint longitudinal model for continuous log-bilirubin (Gaussian) and binary
ascites (threshold/probit).  `family = c("gaussian", "threshold")`.
The residual covariance `~us(trait):units` accounts for within-visit correlation
between the two outcomes after conditioning on the shared subject random effect.
"""
function examples(::Val{:pbcseq_bivariate})
    return (
        "fixed: cbind(log_bili, ascites) ~ trait - 1 + trait:(age + sex); random: ~us(trait):id; rcov: ~us(trait):units",
        load(Val(:pbcseq)),
    )
end

# ── Ordinal Regression ─────────────────────────────────────────────────────────

"""
name: Soay Sheep Horn Type — Ordinal Threshold Model
source: https://jarrodhadfield.github.io/MCMCglmm/course-notes/
example: sshorns_ordinal
dataset: sshorns
formula: "fixed: horn ~ sex; random: ~id"
----

Ordered threshold (probit) model for horn morphology: polled < scurred < normal.
`family = "threshold"`; residual variance is fixed at 1 for identifiability.
`~id` captures repeated-individual variation (some animals appear multiple times).
"""
function examples(::Val{:sshorns_ordinal})
    return ("fixed: horn ~ sex; random: ~id", load(Val(:sshorns)))
end
