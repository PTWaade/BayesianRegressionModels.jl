# scripts/examples/lme4.jl
#
# Formulas from the lme4 R package (Bates, Maechler, Bolker & Walker, JSS 2015).
# https://cran.r-project.org/package=lme4
# Interface: lmer() for LMMs, glmer() for GLMMs.
#
# Key formula features:
#   (1 | g)         — random intercept per group g
#   (x | g)         — correlated random slope + intercept
#   (x || g)        — uncorrelated random slope + intercept
#   (1 | g1/g2)     — nested: expands to (1|g1) + (1|g1:g2)
#   (1|g1) + (1|g2) — fully crossed random intercepts

using CSV, DataFrames, Downloads

const _L4_SLEEPSTUDY_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv"
const _L4_CBPP_URL       = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/cbpp.csv"
const _L4_DYESTUFF_URL   = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Dyestuff.csv"
const _L4_PENICILLIN_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Penicillin.csv"
const _L4_PASTES_URL     = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Pastes.csv"
const _L4_VERBAGG_URL    = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/VerbAgg.csv"
const _L4_CONTRA_URL     = "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Contraception.csv"

# ── Data loaders ─────────────────────────────────────────────────────────────

"""
name: sleepstudy — Sleep Deprivation Reaction Times
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv
----

Belenky et al. (2003) — 18 subjects, 10 days of restricted sleep (3 h/night).
Columns: `Reaction` (average response time, ms), `Days` (0–9), `Subject`.
"""
load(::Val{:sleepstudy}) = CSV.read(Downloads.download(_L4_SLEEPSTUDY_URL), DataFrame)

"""
name: lme4_cbpp — Contagious Bovine Pleuropneumonia
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/cbpp.csv
----

56 observations of 15 herds across 4 periods. Columns: `incidence`, `size`,
`period` (integer 1–4), `herd`. Same data as `brms:cbpp` but period kept numeric.
"""
load(::Val{:lme4_cbpp}) = CSV.read(Downloads.download(_L4_CBPP_URL), DataFrame)

"""
name: Dyestuff — Batch Yield of Dyestuff
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Dyestuff.csv
----

30 observations: 6 batches × 5 replicate yields of a chemical intermediate.
Classic random-intercept-only example from Bates et al. (JSS 2015).
Columns: `Batch` (A–F), `Yield` (grams).
"""
load(::Val{:dyestuff}) = CSV.read(Downloads.download(_L4_DYESTUFF_URL), DataFrame)

"""
name: Penicillin — Zone of Inhibition
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Penicillin.csv
----

144 observations from a fully balanced, crossed two-way random-effects experiment.
6 penicillin samples × 24 assay plates. Columns: `diameter` (zone of inhibition, mm),
`plate` (a–x), `sample` (A–F).
"""
load(::Val{:penicillin}) = CSV.read(Downloads.download(_L4_PENICILLIN_URL), DataFrame)

"""
name: Pastes — Calcium Carbonate Paste Strength
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Pastes.csv
----

60 obs: 10 batches × 3 casks × 2 samples each. Classic nested random-effects example.
Columns: `strength` (response), `batch`, `cask`, `sample`.
"""
load(::Val{:pastes}) = CSV.read(Downloads.download(_L4_PASTES_URL), DataFrame)

"""
name: VerbAgg — Verbal Aggression Ratings
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/VerbAgg.csv
----

316 subjects × 24 items = 7,584 obs. Subjects rate tendency to respond
aggressively in frustrating situations. Columns: `Anger` (trait anger score),
`Gender`, `id` (subject), `item`, `btype` (want/do), `situ` (curse/scold/shout),
`mode` (self/other), `r2` (binary response, Y/N).
"""
load(::Val{:verbagg}) = CSV.read(Downloads.download(_L4_VERBAGG_URL), DataFrame)

"""
name: Contraception — Bangladesh Contraception Use Survey
source: https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Contraception.csv
----

1,934 women in 60 districts, 1988 Bangladesh Fertility Survey (Rodriguez & Elo 2003).
Columns: `woman`, `district`, `use` (Y/N), `livch` (living children: 0/1/2/3+),
`age` (centred), `urban` (Y/N).
"""
load(::Val{:contraception}) = CSV.read(Downloads.download(_L4_CONTRA_URL), DataFrame)

# ── Examples ──────────────────────────────────────────────────────────────────

"""
name: Sleep Deprivation — Correlated Random Slopes
source: https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf
example: sleep_deprivation
dataset: sleepstudy
formula: "Reaction ~ Days + (Days | Subject)"
family: gaussian
----

LMM with correlated by-subject random slope and intercept for `Days`.
`(Days | Subject)` fits one 2×2 random-effects covariance matrix per subject.

```r
lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy)
```
"""
function examples(::Val{:sleepstudy_slope})
    return ("Reaction ~ Days + (Days | Subject)", load(Val(:sleepstudy)))
end

"""
name: Sleep Deprivation — Uncorrelated Random Slopes
source: https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf
example: sleep_deprivation
dataset: sleepstudy
formula: "Reaction ~ Days + (Days || Subject)"
family: gaussian
----

LMM with **uncorrelated** by-subject random slope and intercept.
`(Days || Subject)` constrains the off-diagonal covariance to zero
(diagonal random-effects covariance matrix).

```r
lmer(Reaction ~ Days + (Days || Subject), data = sleepstudy)
```
"""
function examples(::Val{:sleepstudy_uncorr})
    return ("Reaction ~ Days + (Days || Subject)", load(Val(:sleepstudy)))
end

"""
name: Dyestuff — Minimal Random-Intercept LMM
source: https://doi.org/10.18637/jss.v067.i01
example: dyestuff
dataset: dyestuff
formula: "Yield ~ 1 + (1 | Batch)"
family: gaussian
----

The simplest possible LMM: grand-mean intercept plus a random intercept for each
batch. Introductory example in Bates et al. (JSS 2015, §2).

```r
lmer(Yield ~ 1 + (1 | Batch), data = Dyestuff)
```
"""
function examples(::Val{:dyestuff_re})
    return ("Yield ~ 1 + (1 | Batch)", load(Val(:dyestuff)))
end

"""
name: CBPP — Binomial GLMM (lme4 syntax)
source: https://doi.org/10.18637/jss.v067.i01
example: cbpp
dataset: lme4_cbpp
formula: "cbind(incidence, size - incidence) ~ period + (1 | herd)"
family: binomial
----

Binomial GLMM for CBPP incidence using lme4's matrix-response `cbind` syntax.
Contrast with the brms equivalent: `incidence | trials(size) ~ period + (1|herd)`.

```r
glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
      family = binomial, data = cbpp)
```
"""
function examples(::Val{:lme4_cbpp_glmm})
    data = load(Val(:lme4_cbpp))
    data.period = string.(data.period)
    return ("cbind(incidence, size - incidence) ~ period + (1 | herd)", data)
end

"""
name: Penicillin — Crossed Random Intercepts
source: https://doi.org/10.18637/jss.v067.i01
example: penicillin
dataset: penicillin
formula: "diameter ~ 1 + (1 | plate) + (1 | sample)"
family: gaussian
----

LMM with two fully **crossed** random intercepts (plates and samples).
Neither grouping factor is nested within the other.

```r
lmer(diameter ~ 1 + (1 | plate) + (1 | sample), data = Penicillin)
```
"""
function examples(::Val{:penicillin_crossed})
    return ("diameter ~ 1 + (1 | plate) + (1 | sample)", load(Val(:penicillin)))
end

"""
name: Pastes — Nested Random Effects
source: https://doi.org/10.18637/jss.v067.i01
example: pastes
dataset: pastes
formula: "strength ~ (1 | batch/cask)"
family: gaussian
----

LMM with nested random effects. `batch/cask` expands to `(1|batch) + (1|batch:cask)`;
casks are identified locally within each batch, not globally.

```r
lmer(strength ~ (1 | batch/cask), data = Pastes)
```
"""
function examples(::Val{:pastes_nested})
    return ("strength ~ (1 | batch/cask)", load(Val(:pastes)))
end

"""
name: Verbal Aggression — Bernoulli GLMM, Crossed Random Effects
source: https://doi.org/10.18637/jss.v067.i01
example: verbal_aggression
dataset: verbagg
formula: "r2 ~ Anger + Gender + btype + situ + (1 | id) + (1 | item)"
family: binomial
----

Bernoulli GLMM with two fully crossed random intercepts (subjects × items).
`r2` is recoded to integer 0/1 from Y/N.

```r
glmer(r2 ~ Anger + Gender + btype + situ + (1|id) + (1|item),
      family = binomial, data = VerbAgg)
```
"""
function examples(::Val{:verbagg_crossed})
    data = load(Val(:verbagg))
    data.r2 = ifelse.(data.r2 .== "Y", 1, 0)
    return ("r2 ~ Anger + Gender + btype + situ + (1 | id) + (1 | item)", data)
end

"""
name: Contraception — Binomial GLMM with Random Slopes
source: https://cran.r-project.org/package=lme4
example: contraception
dataset: contraception
formula: "use ~ age + I(age^2) + livch + urban + (urban | district)"
family: binomial
----

Binomial GLMM for contraception use with a random slope for `urban` within
district — allowing the urban/rural gap to vary across districts. `use` is
recoded from Y/N to integer 0/1.

```r
glmer(use ~ age + I(age^2) + livch + urban + (urban | district),
      family = binomial, data = Contraception)
```
"""
function examples(::Val{:contraception_urban})
    data = load(Val(:contraception))
    data.use = ifelse.(data.use .== "Y", 1, 0)
    return ("use ~ age + I(age^2) + livch + urban + (urban | district)", data)
end
