using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from two canonical brms papers by Paul-Christian Bürkner:
#
#   JSS 2017: "brms: An R Package for Bayesian Multilevel Models Using Stan"
#             Journal of Statistical Software, 80(1), 1–28.
#             https://doi.org/10.18637/jss.v080.i01
#
#   RJ 2018:  "Advanced Bayesian Multilevel Modeling with the R Package brms"
#             The R Journal, 10(1), 395–411.
#             https://doi.org/10.32614/RJ-2018-017
#             Replication code: https://github.com/paul-buerkner/brms-multilevel-paper

const EPIL_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/epil.csv"

"""
name: epilepsy — MASS::epil Seizure Counts
source: https://vincentarelbundock.github.io/Rdatasets/csv/MASS/epil.csv
----

**MASS::epil** (Thall & Vail 1990); available via Rdatasets. 236 observations on 59
epileptic patients across 4 biweekly visits. Original columns: `y` (seizure count),
`trt` (progabide/placebo), `base` (8-week baseline count), `age`, `subject`, `period`,
`lbase` (log base/4), `lage` (log age). Renamed for brms paper: `count`, `Trt` (0/1),
`zBase` (standardised log base rate), `zAge` (standardised log age), `patient`, `visit`.
"""
function load(::Val{:epilepsy})
    d = CSV.read(Downloads.download(EPIL_URL), DataFrame)
    rename!(d, :y => :count, :subject => :patient, :period => :visit)
    d.Trt   = ifelse.(d.trt .== "progabide", 1, 0)
    d.zBase = (d.lbase .- mean(d.lbase)) ./ std(d.lbase)
    d.zAge  = (d.lage  .- mean(d.lage))  ./ std(d.lage)
    d.patient = string.(d.patient)
    return d
end

"""
name: Epileptic Seizure Counts — Poisson GLMM
source: https://doi.org/10.18637/jss.v080.i01
example: epilepsy
dataset: epilepsy
chapter: JSS 2017
formula: "count ~ zBase * Trt + (1 | patient)"
----

Poisson GLMM; interaction of standardised baseline × treatment; random patient intercept.
"""
function examples(::Val{:epilepsy_base})
    d = load(Val(:epilepsy))
    return ("count ~ zBase * Trt + (1 | patient)", d)
end

"""
name: Epileptic Seizure Counts — Truncated Poisson
source: https://doi.org/10.18637/jss.v080.i01
example: epilepsy
dataset: epilepsy
chapter: JSS 2017
formula: "count | trunc(ub = 104) ~ zBase * Trt + (1 | patient)"
----

Same Poisson GLMM with upper truncation at 104 (the maximum possible count per visit).
"""
function examples(::Val{:epilepsy_trunc})
    d = load(Val(:epilepsy))
    return ("count | trunc(ub = 104) ~ zBase * Trt + (1 | patient)", d)
end

"""
name: Epileptic Seizure Counts — Simplified Main Effects
source: https://doi.org/10.18637/jss.v080.i01
example: epilepsy
dataset: epilepsy
chapter: JSS 2017
formula: "count ~ Trt + (1 | patient)"
----

Simplified main-effects Poisson; omits baseline covariate; comparison model.
"""
function examples(::Val{:epilepsy_simple})
    d = load(Val(:epilepsy))
    return ("count ~ Trt + (1 | patient)", d)
end

"""
name: inhaler — Synthetic Inhaler Ordinal Ratings
source: synthetic
----

**brms::inhaler** (Ezzet & Whitehead 1991) — crossover clinical trial on asthma inhalers.
Original dataset built into brms (not publicly hosted as CSV). Synthetic data generated
here to match the structure.

572 observations on 286 patients × 2 periods: `rating` (1–4 ordinal ease-of-use score),
`treat` (0=reference/1=active), `period` (1/2), `carry` (carryover: -1/0/1), `subject`.
"""
function load(::Val{:inhaler})
    rng = MersenneTwister(2017)
    n_subj = 143
    rows   = NamedTuple[]
    for subj in 1:n_subj
        treat_first = rand(rng) > 0.5
        for period in 1:2
            treat  = (period == 1) == treat_first ? 1 : 0
            carry  = period == 2 ? (treat_first ? 1 : -1) : 0
            logit_base = -0.8 + 0.4 * treat - 0.1 * carry
            p = [1 / (1 + exp(-logit_base + k)) for k in [-1.5, 0.0, 1.5]]
            probs = [p[1], p[2] - p[1], p[3] - p[2], 1 - p[3]]
            probs = max.(probs, 0.0); probs ./= sum(probs)
            rating = findfirst(cumsum(probs) .>= rand(rng))
            push!(rows, (; subject = string(subj), period, treat, carry, rating))
        end
    end
    return DataFrame(rows)
end

"""
name: Inhaler Ordinal Ratings — Sequential Ratio Model
source: https://doi.org/10.18637/jss.v080.i01
example: inhaler
dataset: inhaler
chapter: JSS 2017
formula: "rating ~ period + carry + cs(treat)"
----

Sequential ratio ordinal model; `cs()` = category-specific coefficient for `treat`;
`sratio("logit")` family estimates P(Y > k | Y ≥ k) at each threshold.
"""
function examples(::Val{:inhaler})
    d = load(Val(:inhaler))
    return ("rating ~ period + carry + cs(treat)", d)
end

"""
name: kidney — Synthetic Kidney Recurrence Times
source: synthetic
----

**brms::kidney** (McGilchrist & Aisbett 1991) — recurrence of infection in kidney
patients. Built into brms; not publicly hosted as CSV. To export from R:
`data("kidney", package = "brms"); write.csv(kidney, "kidney.csv")`. Synthetic data
generated here to match the structure.

114 observations on 57 patients (two kidneys each): `time` (days until recurrence),
`censored` (0=observed/1=right-censored), `age`, `sex` (male/female),
`disease` (GN/AN/PKD/other), `patient`.
"""
function load(::Val{:kidney})
    rng       = MersenneTwister(1991)
    n_patients = 57
    diseases   = ["GN", "AN", "PKD", "other"]
    sexes      = ["male", "female"]
    rows       = NamedTuple[]
    for patient in 1:n_patients
        age     = round(Int, 30 + 25 * rand(rng))
        sex     = rand(rng, sexes)
        disease = rand(rng, diseases)
        u_pat   = 0.5 * randn(rng)
        for _ in 1:2
            mu_log  = 3.5 - 0.01 * (age - 40) + (sex == "male" ? 0.3 : 0.0) + u_pat
            time    = round(Int, max(1, exp(mu_log + 0.8 * randn(rng))))
            censored = rand(rng) < 0.4 ? 1 : 0
            push!(rows, (; patient = string(patient), time, censored, age, sex, disease))
        end
    end
    return DataFrame(rows)
end

"""
name: Kidney Recurrence Times — Lognormal Survival
source: https://doi.org/10.18637/jss.v080.i01
example: kidney
dataset: kidney
chapter: JSS 2017
formula: "time | cens(censored) ~ age * sex + disease + (1 | patient)"
----

Lognormal survival model; `cens()` handles right-censored observations; random intercept
per patient accounts for within-patient correlation between two kidneys.
"""
function examples(::Val{:kidney})
    d = load(Val(:kidney))
    return ("time | cens(censored) ~ age * sex + disease + (1 | patient)", d)
end

"""
name: hetero_jss — Synthetic Heteroscedastic Data
source: synthetic
----

Simulated data from the JSS paper to demonstrate distributional regression. 100
observations: `y` (continuous outcome), `x` (group: `"lo"` or `"hi"`). The `"hi"`
group has a larger variance, making the data heteroscedastic.
"""
function load(::Val{:hetero_jss})
    rng = MersenneTwister(42)
    n   = 100
    x   = rand(rng, ["lo", "hi"], n)
    y   = [xi == "hi" ? 1.0 + 2.0 * randn(rng) : 0.0 + 0.5 * randn(rng) for xi in x]
    return DataFrame(; y, x)
end

"""
name: Heteroscedastic Model — Group-Specific Sigma
source: https://doi.org/10.18637/jss.v080.i01
example: hetero_jss
dataset: hetero_jss
chapter: JSS 2017
formula: "bf(y ~ x, sigma ~ 0 + x)"
----

Gaussian with group-specific `sigma`; the sub-formula `sigma ~ 0 + x` models
log(sigma) as a function of `x`.
"""
function examples(::Val{:hetero_jss_sigma})
    d = load(Val(:hetero_jss))
    return ("bf(y ~ x, sigma ~ 0 + x)", d)
end

"""
name: Quantile Regression — Asymmetric Laplace
source: https://doi.org/10.18637/jss.v080.i01
example: hetero_jss
dataset: hetero_jss
chapter: JSS 2017
formula: "bf(y ~ x, quantile = 0.25)"
----

Bayesian quantile regression at the 25th percentile using an asymmetric Laplace
distribution.
"""
function examples(::Val{:hetero_jss_quantile})
    d = load(Val(:hetero_jss))
    return ("bf(y ~ x, quantile = 0.25)", d)
end

const FISH_URL = "http://paulbuerkner.com/data/fish.csv"

"""
name: fish_rj — UCLA IDRE Fishing Dataset
source: http://paulbuerkner.com/data/fish.csv
----

UCLA IDRE fishing dataset; 250 fishing-trip records. Also covered in
`brms.jl` (`:distreg_fish`). Columns: `nofish`, `livebait`, `camper`,
`persons`, `child`, `xb`, `zg`, `count`.
"""
load(::Val{:fish_rj}) = CSV.read(Downloads.download(FISH_URL), DataFrame)

"""
name: Fish Counts — Poisson Baseline
source: https://doi.org/10.32614/RJ-2018-017
example: fish_rj
dataset: fish_rj
chapter: RJ 2018
formula: "count ~ persons + child + camper"
----

Poisson model for fishing trip catch counts; no zero-inflation component.
"""
function examples(::Val{:fish_rj_poisson})
    d = load(Val(:fish_rj))
    return ("count ~ persons + child + camper", d)
end

"""
name: Fish Counts — Zero-Inflated Poisson
source: https://doi.org/10.32614/RJ-2018-017
example: fish_rj
dataset: fish_rj
chapter: RJ 2018
formula: "bf(count ~ persons + child + camper, zi ~ child)"
----

Zero-inflated Poisson; `zi` sub-model predicts structural-zero probability from `child`.
"""
function examples(::Val{:fish_rj_zip})
    d = load(Val(:fish_rj))
    return ("bf(count ~ persons + child + camper, zi ~ child)", d)
end

"""
name: rent99 — Synthetic Munich Rent Data
source: synthetic
----

**gamlss.data::rent99** (Fahrmeir et al. 2013) — Munich rental apartments, 1999.
Not on Rdatasets; to obtain the original: `data("rent99", package = "gamlss.data")`.
Synthetic data generated here to match the structure.

~3082 observations: `rentsqm` (rent per m², EUR), `area` (floor area m²), `yearc`
(year of construction, 1918–1997), `district` (1–90, city district).
"""
function load(::Val{:rent99})
    rng = MersenneTwister(1999)
    n   = 3082
    area  = 20.0 .+ 120.0 .* rand(rng, n)
    yearc = round.(Int, 1918 .+ 79 .* rand(rng, n))
    dist  = rand(rng, 1:90, n)
    u_dist = randn(rng, 90) .* 0.4
    mu     = 6.0 .- 0.005 .* (area .- 60) .+ 0.008 .* (yearc .- 1960) .+ u_dist[dist]
    sigma  = exp.(-0.5 .+ 0.003 .* (yearc .- 1960))
    rentsqm = mu .+ sigma .* randn(rng, n)
    return DataFrame(; rentsqm, area, yearc, district = string.(dist))
end

"""
name: Munich Rent Data — Tensor-Product Spline
source: https://doi.org/10.32614/RJ-2018-017
example: rent99
dataset: rent99
chapter: RJ 2018
formula: "rentsqm ~ t2(area, yearc) + (1 | district)"
----

Tensor-product spline of area × construction year; random intercept per district.
"""
function examples(::Val{:rent99_spline})
    d = load(Val(:rent99))
    return ("rentsqm ~ t2(area, yearc) + (1 | district)", d)
end

"""
name: Munich Rent Data — Distributional Smooth Regression
source: https://doi.org/10.32614/RJ-2018-017
example: rent99
dataset: rent99
chapter: RJ 2018
formula: "bf(rentsqm ~ t2(area, yearc) + (1 | ID1 | district), sigma ~ t2(area, yearc) + (1 | ID1 | district))"
----

Distributional model: both mean and log(sigma) are smooth functions; `ID1` label allows
correlation between the two sets of district random effects.
"""
function examples(::Val{:rent99_distr})
    d = load(Val(:rent99))
    return ("bf(rentsqm ~ t2(area, yearc) + (1 | ID1 | district), sigma ~ t2(area, yearc) + (1 | ID1 | district))", d)
end

"""
name: loss_rj — Synthetic Actuarial Loss Triangle
source: synthetic
----

**brms::loss** — actuarial loss development triangle. Also covered in
`brms.jl` (`:loss`). Synthetic data generated here to match the structure.

Columns: `AY` (accident year, 1981–1990), `dev` (development year), `cum` (cumulative
losses), `premium`; 55 rows forming a loss triangle.
"""
function load(::Val{:loss_rj})
    rng = MersenneTwister(2018)
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
name: Nonlinear Loss Development — Weibull Curve
source: https://doi.org/10.32614/RJ-2018-017
example: loss_rj
dataset: loss_rj
chapter: RJ 2018
formula: "bf(cum ~ ult * (1 - exp(-(dev/theta)^omega)), ult ~ 1 + (1 | AY), omega ~ 1, theta ~ 1, nl = TRUE)"
----

Weibull loss development curve; `ult` (ultimate loss) varies by accident year;
`omega` = shape, `theta` = scale; `nl = TRUE` enables the nonlinear formula.
"""
function examples(::Val{:loss_rj})
    d = load(Val(:loss_rj))
    return ("bf(cum ~ ult * (1 - exp(-(dev/theta)^omega)), ult ~ 1 + (1 | AY), omega ~ 1, theta ~ 1, nl = TRUE)", d)
end

"""
name: multi_member — Synthetic Multiple Membership Data
source: synthetic
----

1000 students nested in 10 schools, each attending two schools with weights `w1`, `w2`
(summing to 1). Used to demonstrate brms's `mm()` syntax.

Columns: `y` (outcome), `s1`, `s2` (school IDs), `w1`, `w2` (membership weights).
"""
function load(::Val{:multi_member})
    rng      = MersenneTwister(2018)
    nschools = 10
    nstudents = 1000
    u_school = randn(rng, nschools) .* 0.4
    rows     = NamedTuple[]
    for _ in 1:nstudents
        s1 = rand(rng, 1:nschools)
        s2 = rand(rng, setdiff(1:nschools, [s1]))
        w1 = 0.3 + 0.4 * rand(rng)
        w2 = 1.0 - w1
        y  = w1 * u_school[s1] + w2 * u_school[s2] + 0.5 * randn(rng)
        push!(rows, (; y, s1 = string(s1), s2 = string(s2), w1, w2))
    end
    return DataFrame(rows)
end

"""
name: Multiple Membership Model — Equal Weights
source: https://doi.org/10.32614/RJ-2018-017
example: multi_member
dataset: multi_member
chapter: RJ 2018
formula: "y ~ 1 + (1 | mm(s1, s2))"
----

Equal-weight multiple membership; student outcome depends equally on two schools.
"""
function examples(::Val{:multi_member_equal})
    d = load(Val(:multi_member))
    return ("y ~ 1 + (1 | mm(s1, s2))", d)
end

"""
name: Multiple Membership Model — Weighted
source: https://doi.org/10.32614/RJ-2018-017
example: multi_member
dataset: multi_member
chapter: RJ 2018
formula: "y ~ 1 + (1 | mm(s1, s2, weights = cbind(w1, w2)))"
----

Weighted multiple membership; `w1 + w2 = 1` for each student; weights the school
random effects by the time spent at each school.
"""
function examples(::Val{:multi_member_weighted})
    d = load(Val(:multi_member))
    return ("y ~ 1 + (1 | mm(s1, s2, weights = cbind(w1, w2)))", d)
end
