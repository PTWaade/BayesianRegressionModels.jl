# scripts/examples/mixed_models_jl.jl
#
# Formulas from MixedModels.jl (JuliaStats).
# https://juliastats.org/MixedModels.jl/stable/
#
# Pure-Julia implementation of linear and generalised linear mixed-effects models,
# equivalent to R's lme4. Uses StatsModels.jl's @formula macro.
#
# Key formula differences from lme4:
#   - Explicit intercepts idiomatic:  1 + x  rather than just  x
#   - Uncorrelated slopes:  zerocorr(x | g)  rather than  (x || g)
#   - Grouping separator in @formula uses |  (same as lme4)
#
# Fitting:
#   LinearMixedModel(@formula(y ~ ...), data) |> fit!
#   GeneralizedLinearMixedModel(@formula(y ~ ...), data, Binomial())  |> fit!

using CSV, DataFrames, Downloads

const _MM_SLEEPSTUDY_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv"
const _MM_DYESTUFF_URL   = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Dyestuff.csv"
const _MM_PENICILLIN_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Penicillin.csv"
const _MM_PASTES_URL     = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Pastes.csv"
const _MM_VERBAGG_URL    = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/VerbAgg.csv"
const _MM_CONTRA_URL     = "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Contraception.csv"
const _MM_INSTEVAL_URL   = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/InstEval.csv"

# ── Data loaders ─────────────────────────────────────────────────────────────

"""
name: sleepstudy — Sleep Deprivation Reaction Times
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv
----

Belenky et al. (2003) — 18 subjects, 10 days of restricted sleep (3 h/night).
Columns: `Reaction` (ms), `Days` (0–9), `Subject`. Also see `lme4:sleepstudy`.
"""
load(::Val{:sleepstudy}) = CSV.read(Downloads.download(_MM_SLEEPSTUDY_URL), DataFrame)

"""
name: Dyestuff — Batch Yield
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Dyestuff.csv
----

30 obs: 6 batches × 5 replicate yields. Columns: `Batch` (A–F), `Yield` (g).
MixedModels.jl uses lowercase `yield`/`batch` in its bundled copy; the Rdatasets
CSV uses capitalised names.
"""
load(::Val{:dyestuff}) = CSV.read(Downloads.download(_MM_DYESTUFF_URL), DataFrame)

"""
name: Penicillin — Zone of Inhibition
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Penicillin.csv
----

144 obs: fully balanced 6-sample × 24-plate crossed design.
Columns: `diameter` (mm), `plate` (a–x), `sample` (A–F).
"""
load(::Val{:penicillin}) = CSV.read(Downloads.download(_MM_PENICILLIN_URL), DataFrame)

"""
name: Pastes — Paste Strength
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Pastes.csv
----

60 obs: 10 batches × 3 casks × 2 samples. Nested random-effects benchmark dataset.
Columns: `strength`, `batch`, `cask`, `sample`.
"""
load(::Val{:pastes}) = CSV.read(Downloads.download(_MM_PASTES_URL), DataFrame)

"""
name: VerbAgg — Verbal Aggression Ratings
source: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/VerbAgg.csv
----

7,584 obs: 316 subjects × 24 items. Binary outcome `r2` (Y/N) recoded to 0/1.
Columns: `Anger`, `Gender`, `id`, `item`, `btype`, `situ`, `mode`, `r2`.
"""
load(::Val{:verbagg}) = CSV.read(Downloads.download(_MM_VERBAGG_URL), DataFrame)

"""
name: Contraception — Bangladesh Contraception Survey
source: https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Contraception.csv
----

1,934 women in 60 districts. Binary outcome `use` (Y/N) recoded to 0/1.
Columns: `woman`, `district`, `use`, `livch`, `age`, `urban`.
"""
load(::Val{:contraception}) = CSV.read(Downloads.download(_MM_CONTRA_URL), DataFrame)

# ── Examples ──────────────────────────────────────────────────────────────────

"""
name: Dyestuff — Minimal Random-Intercept LMM
source: https://juliastats.org/MixedModels.jl/stable/constructors/
example: dyestuff
dataset: dyestuff
formula: "Yield ~ 1 + (1 | Batch)"
family: gaussian
----

The canonical introductory LMM in MixedModels.jl: one variance component for
batch-to-batch variability, no fixed covariates.

```julia
fit(LinearMixedModel, @formula(Yield ~ 1 + (1 | Batch)), dyestuff)
```

Note: MixedModels.jl's bundled copy uses lowercase `yield`/`batch`; the Rdatasets
CSV uses `Yield`/`Batch`.
"""
function examples(::Val{:dyestuff_re})
    return ("Yield ~ 1 + (1 | Batch)", load(Val(:dyestuff)))
end

"""
name: Sleep Deprivation — Correlated Random Slopes
source: https://juliastats.org/MixedModels.jl/stable/constructors/
example: sleep_deprivation
dataset: sleepstudy
formula: "Reaction ~ 1 + Days + (1 + Days | Subject)"
family: gaussian
----

LMM with correlated by-subject random slope and intercept for `Days`.
Idiomatic MixedModels.jl style writes explicit `1 +` intercepts.

```julia
fit(LinearMixedModel, @formula(Reaction ~ 1 + Days + (1 + Days | Subject)), sleepstudy)
```
"""
function examples(::Val{:sleepstudy_slope})
    return ("Reaction ~ 1 + Days + (1 + Days | Subject)", load(Val(:sleepstudy)))
end

"""
name: Sleep Deprivation — Uncorrelated Random Slopes (zerocorr)
source: https://juliastats.org/MixedModels.jl/stable/constructors/
example: sleep_deprivation
dataset: sleepstudy
formula: "Reaction ~ 1 + Days + zerocorr(1 + Days | Subject)"
family: gaussian
----

Same as the correlated model but with the off-diagonal covariance forced to zero.
MixedModels.jl uses `zerocorr()` instead of lme4's `(x || g)` syntax.

```julia
fit(LinearMixedModel,
    @formula(Reaction ~ 1 + Days + zerocorr(1 + Days | Subject)), sleepstudy)
```
"""
function examples(::Val{:sleepstudy_zerocorr})
    return ("Reaction ~ 1 + Days + zerocorr(1 + Days | Subject)", load(Val(:sleepstudy)))
end

"""
name: Penicillin — Crossed Random Intercepts
source: https://juliastats.org/MixedModels.jl/stable/constructors/
example: penicillin
dataset: penicillin
formula: "diameter ~ 1 + (1 | plate) + (1 | sample)"
family: gaussian
----

LMM with two fully crossed random intercepts. Neither `plate` nor `sample`
is nested within the other; both random intercepts appear additively.

```julia
fit(LinearMixedModel,
    @formula(diameter ~ 1 + (1 | plate) + (1 | sample)), penicillin)
```
"""
function examples(::Val{:penicillin_crossed})
    return ("diameter ~ 1 + (1 | plate) + (1 | sample)", load(Val(:penicillin)))
end

"""
name: Pastes — Nested Random Effects
source: https://juliastats.org/MixedModels.jl/stable/constructors/
example: pastes
dataset: pastes
formula: "strength ~ 1 + (1 | batch/cask)"
family: gaussian
----

LMM with nested random effects: `batch/cask` expands to `(1|batch) + (1|batch:cask)`.
Cask labels are local to each batch.

```julia
fit(LinearMixedModel, @formula(strength ~ 1 + (1 | batch/cask)), pastes)
```
"""
function examples(::Val{:pastes_nested})
    return ("strength ~ 1 + (1 | batch/cask)", load(Val(:pastes)))
end

"""
name: Verbal Aggression — Bernoulli GLMM, Crossed Random Effects
source: https://juliastats.org/MixedModels.jl/stable/constructors/
example: verbal_aggression
dataset: verbagg
formula: "r2 ~ 1 + Anger + Gender + btype + situ + (1 | id) + (1 | item)"
family: bernoulli
----

Bernoulli GLMM with two crossed random intercepts (subjects × items). `r2`
recoded from Y/N to integer 0/1.

```julia
fit(GeneralizedLinearMixedModel,
    @formula(r2 ~ 1 + Anger + Gender + btype + situ + (1|id) + (1|item)),
    verbagg, Bernoulli())
```
"""
function examples(::Val{:verbagg_crossed})
    data = load(Val(:verbagg))
    data.r2 = ifelse.(data.r2 .== "Y", 1, 0)
    return ("r2 ~ 1 + Anger + Gender + btype + situ + (1 | id) + (1 | item)", data)
end

"""
name: Contraception — Bernoulli GLMM with Polynomial Age
source: https://juliastats.org/MixedModels.jl/stable/constructors/
example: contraception
dataset: contraception
formula: "use ~ 1 + age + abs2(age) + livch + urban + (1 | district)"
family: bernoulli
----

Bernoulli GLMM for contraception use in Bangladesh. MixedModels.jl uses `abs2()`
(squaring function) inside `@formula` instead of R's `I(age^2)`. `use` recoded
from Y/N to integer 0/1.

```julia
fit(GeneralizedLinearMixedModel,
    @formula(use ~ 1 + age + abs2(age) + livch + urban + (1|district)),
    contraception, Bernoulli())
```
"""
function examples(::Val{:contraception_glmm})
    data = load(Val(:contraception))
    data.use = ifelse.(data.use .== "Y", 1, 0)
    return ("use ~ 1 + age + abs2(age) + livch + urban + (1 | district)", data)
end
