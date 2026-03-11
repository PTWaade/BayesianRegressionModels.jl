# scripts/examples/glm_jl.jl
#
# Formulas from GLM.jl (JuliaStats).
# https://juliastats.org/GLM.jl/stable/
#
# Pure-Julia GLM fitting via StatsModels.jl @formula.
# Supports Gaussian, Binomial, Poisson, NegativeBinomial, Gamma, InverseGaussian
# with the canonical link or a specified alternative.
#
# Fitting:
#   glm(@formula(y ~ x1 + x2), data, Poisson())
#   glm(@formula(y ~ x1 + x2), data, Binomial(), ProbitLink())
#   lm(@formula(y ~ x1 + x2), data)         # Gaussian identity (OLS)
#   negbin(@formula(y ~ x), data, LogLink()) # Negative binomial

using CSV, DataFrames, Downloads, Random

const _GLM_QUINE_URL  = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/quine.csv"
const _GLM_TREES_URL  = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/trees.csv"
const _GLM_PIMA_URL   = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Pima.tr.csv"
const _GLM_SAVING_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/LifeCycleSavings.csv"

# ── Data loaders ─────────────────────────────────────────────────────────────

"""
name: dobson_poisson — Dobson (1990) Poisson Example
source: synthetic
----

Classic textbook Poisson log-linear GLM from Dobson (1990, pp. 93–94), reproduced
in the GLM.jl documentation. 9 observations, 3 outcomes × 3 treatments.
Columns: `Counts`, `Outcome` (A/B/C), `Treatment` (I/II/III).
"""
function load(::Val{:dobson_poisson})
    return DataFrame(
        Counts    = [18, 17, 15, 20, 10, 20, 25, 13, 12],
        Outcome   = repeat(["A", "B", "C"], 3),
        Treatment = repeat(["I", "II", "III"], inner=3),
    )
end

"""
name: quine — School Absenteeism
source: https://vincentarelbundock.github.io/Rdatasets/csv/MASS/quine.csv
----

146 Australian school children (MASS package). Used in both GLM.jl and R's MASS.
Columns: `Eth` (A/N), `Sex` (F/M), `Age` (F0/F1/F2/F3), `Lrn` (AL/SL), `Days`
(absent days, count response).
"""
load(::Val{:quine}) = CSV.read(Downloads.download(_GLM_QUINE_URL), DataFrame)

"""
name: trees — Timber Volume of Black Cherry Trees
source: https://vincentarelbundock.github.io/Rdatasets/csv/datasets/trees.csv
----

31 black cherry trees (base R `datasets` package). Columns: `Girth` (diameter at
4.5 ft, inches), `Height` (ft), `Volume` (cubic ft).
"""
load(::Val{:trees}) = CSV.read(Downloads.download(_GLM_TREES_URL), DataFrame)

"""
name: Pima.tr — Pima Indians Diabetes (Training Set)
source: https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Pima.tr.csv
----

200 Pima Indian women; outcome is diabetes diagnosis (Yes/No). From MASS.
Columns: `npreg`, `glu`, `bp`, `skin`, `bmi`, `ped`, `age`, `type` (Yes/No).
"""
load(::Val{:pima_tr}) = CSV.read(Downloads.download(_GLM_PIMA_URL), DataFrame)

"""
name: LifeCycleSavings — International Savings Rate Data
source: https://vincentarelbundock.github.io/Rdatasets/csv/datasets/LifeCycleSavings.csv
----

50 countries, 1960–1970 (Belsley, Kuh & Welsch 1980). Columns: `SR` (savings rate,
response), `Pop15` (% under 15), `Pop75` (% over 75), `DPI` (real per-capita income),
`DDPI` (income growth rate).
"""
load(::Val{:lifecycle_savings}) = CSV.read(Downloads.download(_GLM_SAVING_URL), DataFrame)

# ── Examples ──────────────────────────────────────────────────────────────────

"""
name: Dobson Poisson — Log-Linear GLM
source: https://juliastats.org/GLM.jl/stable/examples/
example: dobson_poisson
dataset: dobson_poisson
formula: "Counts ~ Outcome + Treatment"
family: poisson
----

Canonical example from the GLM.jl documentation (and Dobson 1990). Poisson
log-linear model for counts in a two-way layout.

```julia
glm(@formula(Counts ~ Outcome + Treatment), dobson, Poisson())
```
"""
function examples(::Val{:dobson_poisson})
    return ("Counts ~ Outcome + Treatment", load(Val(:dobson_poisson)))
end

"""
name: Quine — Negative Binomial GLM
source: https://juliastats.org/GLM.jl/stable/examples/
example: quine
dataset: quine
formula: "Days ~ Eth + Sex + Age + Lrn"
family: negativebinomial
----

Negative binomial GLM for overdispersed count data (school absent days).
Example from the GLM.jl documentation; R equivalent is `MASS::glm.nb()`.

```julia
negbin(@formula(Days ~ Eth + Sex + Age + Lrn), quine, LogLink())
```
"""
function examples(::Val{:quine_nb})
    return ("Days ~ Eth + Sex + Age + Lrn", load(Val(:quine)))
end

"""
name: Trees — Gaussian GLM (OLS)
source: https://juliastats.org/GLM.jl/stable/
example: trees
dataset: trees
formula: "Volume ~ Height + Girth"
family: gaussian
----

Linear regression of timber volume on tree girth and height; standard OLS
via GLM.jl's `lm()` (equivalent to `glm(..., Normal(), IdentityLink())`).

```julia
lm(@formula(Volume ~ Height + Girth), trees)
```
"""
function examples(::Val{:trees_lm})
    return ("Volume ~ Height + Girth", load(Val(:trees)))
end

"""
name: Pima Diabetes — Logistic Regression
source: https://juliastats.org/GLM.jl/stable/
example: pima_diabetes
dataset: pima_tr
formula: "type ~ npreg + glu + bp + bmi + ped + age"
family: binomial
----

Binary logistic regression for diabetes diagnosis in Pima Indian women.
`type` is recoded from Yes/No to integer 0/1.

```julia
glm(@formula(type ~ npreg + glu + bp + bmi + ped + age), pima, Binomial())
```
"""
function examples(::Val{:pima_logistic})
    data = load(Val(:pima_tr))
    data.type = ifelse.(data.type .== "Yes", 1, 0)
    return ("type ~ npreg + glu + bp + bmi + ped + age", data)
end

"""
name: Life-Cycle Savings — Multiple Linear Regression
source: https://juliastats.org/GLM.jl/stable/
example: lifecycle_savings
dataset: lifecycle_savings
formula: "SR ~ Pop15 + Pop75 + DPI + DDPI"
family: gaussian
----

OLS regression of national savings rate on demographic and income predictors
(50 countries). Classic textbook dataset (Belsley et al. 1980).

```julia
lm(@formula(SR ~ Pop15 + Pop75 + DPI + DDPI), LifeCycleSavings)
```
"""
function examples(::Val{:savings_ols})
    return ("SR ~ Pop15 + Pop75 + DPI + DDPI", load(Val(:lifecycle_savings)))
end

"""
name: Quine — Poisson GLM (vs. Negative Binomial)
source: https://juliastats.org/GLM.jl/stable/examples/
example: quine
dataset: quine
formula: "Days ~ Eth + Sex + Age + Lrn"
family: poisson
----

Poisson GLM for school absenteeism counts. Overdispersion relative to Poisson
motivates the negative-binomial model (`:quine_nb`). Same formula, different
distributional assumption.

```julia
glm(@formula(Days ~ Eth + Sex + Age + Lrn), quine, Poisson())
```
"""
function examples(::Val{:quine_poisson})
    return ("Days ~ Eth + Sex + Age + Lrn", load(Val(:quine)))
end
