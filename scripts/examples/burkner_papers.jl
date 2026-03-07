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

##############################################################################
# JSS 2017 — Example 1: Epileptic seizure counts
# Source: Thall & Vail (1990); equivalent to MASS::epil.
#   Available via Rdatasets as MASS::epil (same observations, different column names).
#
# Dataset: 236 observations on 59 epileptic patients across 4 biweekly visits.
#   Columns (MASS::epil): y (seizure count), trt (progabide/placebo), base (8-week
#   baseline count), age, subject, period, lbase (log base/4), lage (log age).
#   In the brms paper: count, Trt (0/1), zBase (standardised log base rate),
#   zAge (standardised log age), patient, visit.
#
# brms model formulas:
#   "count ~ zBase * Trt + (1 | patient)"
#     (Poisson GLMM; interaction of standardised baseline × treatment; random patient intercept)
#   "count | trunc(ub = 104) ~ zBase * Trt + (1 | patient)"
#     (same with upper truncation at 104 — the maximum possible count)
#   "count ~ Trt + (1 | patient)"
#     (simplified main-effects Poisson; comparison model)
##############################################################################

const EPIL_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/epil.csv"

function load(::Val{:epilepsy})
    d = CSV.read(Downloads.download(EPIL_URL), DataFrame)
    # Align column names with brms epilepsy conventions
    rename!(d, :y => :count, :subject => :patient, :period => :visit)
    d.Trt   = ifelse.(d.trt .== "progabide", 1, 0)
    d.zBase = (d.lbase .- mean(d.lbase)) ./ std(d.lbase)
    d.zAge  = (d.lage  .- mean(d.lage))  ./ std(d.lage)
    d.patient = string.(d.patient)
    return d
end

function examples(::Val{:epilepsy})
    d = load(Val(:epilepsy))
    return [
        ("count ~ zBase * Trt + (1 | patient)", d),
        ("count | trunc(ub = 104) ~ zBase * Trt + (1 | patient)", d),
        ("count ~ Trt + (1 | patient)", d),
    ]
end

##############################################################################
# JSS 2017 — Example 2: Inhaler ordinal ratings (brms::inhaler)
# Source: Ezzet & Whitehead (1991) — crossover clinical trial on asthma inhalers.
#   The original dataset is built into brms and not publicly hosted as CSV.
#   To export from R:  data("inhaler", package = "brms"); write.csv(inhaler, "inhaler.csv")
#
# Dataset structure: 572 observations on 286 patients × 2 periods.
#   rating (1–4 ordinal ease-of-use score), treat (0=reference/1=active),
#   period (1/2), carry (carryover: -1/0/1), subject.
#
# brms model formula:
#   "rating ~ period + carry + cs(treat)"
#     (sequential ratio ordinal model; cs() = category-specific coefficient for treat;
#      sratio("logit") family estimates P(Y > k | Y ≥ k) at each threshold)
#
# Synthetic data generated below to match the structure.
##############################################################################

function load(::Val{:inhaler})
    # Synthetic equivalent of brms::inhaler (Ezzet & Whitehead 1991 crossover trial)
    rng = MersenneTwister(2017)
    n_subj = 143
    rows   = NamedTuple[]
    for subj in 1:n_subj
        treat_first = rand(rng) > 0.5
        for period in 1:2
            treat  = (period == 1) == treat_first ? 1 : 0
            carry  = period == 2 ? (treat_first ? 1 : -1) : 0
            # Baseline ordinal probabilities; treatment shifts upward
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

function examples(::Val{:inhaler})
    d = load(Val(:inhaler))
    return [
        ("rating ~ period + carry + cs(treat)", d),
    ]
end

##############################################################################
# JSS 2017 — Example 3: Kidney recurrence times (brms::kidney)
# Source: McGilchrist & Aisbett (1991) — recurrence of infection in kidney patients.
#   Built into brms; not publicly hosted as CSV.
#   To export from R:  data("kidney", package = "brms"); write.csv(kidney, "kidney.csv")
#
# Dataset structure: 863 observations on 863 patients (each with 2 kidneys).
#   time (days until recurrence), censored (0=observed/1=censored), age (years),
#   sex (male/female), disease (GN/AN/PKD/other), patient.
#
# brms model formula:
#   "time | cens(censored) ~ age * sex + disease + (1 | patient)"
#     (lognormal survival; | cens() handles right-censored observations;
#      random intercept per patient accounts for within-patient correlation)
#
# Synthetic data generated below to match the structure.
##############################################################################

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
        u_pat   = 0.5 * randn(rng)            # patient-level frailty
        for _ in 1:2                           # two kidneys per patient
            mu_log  = 3.5 - 0.01 * (age - 40) + (sex == "male" ? 0.3 : 0.0) + u_pat
            time    = round(Int, max(1, exp(mu_log + 0.8 * randn(rng))))
            censored = rand(rng) < 0.4 ? 1 : 0
            push!(rows, (; patient = string(patient), time, censored, age, sex, disease))
        end
    end
    return DataFrame(rows)
end

function examples(::Val{:kidney})
    d = load(Val(:kidney))
    return [
        ("time | cens(censored) ~ age * sex + disease + (1 | patient)", d),
    ]
end

##############################################################################
# JSS 2017 — Example 4: Heteroscedastic and quantile models (synthetic)
# Data simulated in the paper to demonstrate distributional regression.
#
# brms model formulas:
#   "bf(y ~ x, sigma ~ 0 + x)"
#     (Gaussian with group-specific sigma; sigma modelled as function of x)
#   "bf(y ~ x, quantile = 0.25)"
#     (asymmetric Laplace for Bayesian quantile regression at the 25th percentile)
##############################################################################

function load(::Val{:hetero_jss})
    rng = MersenneTwister(42)
    n   = 100
    x   = rand(rng, ["lo", "hi"], n)
    y   = [xi == "hi" ? 1.0 + 2.0 * randn(rng) : 0.0 + 0.5 * randn(rng) for xi in x]
    return DataFrame(; y, x)
end

function examples(::Val{:hetero_jss})
    d = load(Val(:hetero_jss))
    return [
        ("bf(y ~ x, sigma ~ 0 + x)", d),
        ("bf(y ~ x, quantile = 0.25)", d),
    ]
end

##############################################################################
# RJ 2018 — Example 1: Fish counts (zero-inflated Poisson / negative binomial)
# Source: http://paulbuerkner.com/data/fish.csv (UCLA IDRE fishing dataset)
# Note: also covered in scripts/examples/brms.jl (:distreg_fish).
#
# Dataset: 250 fishing-trip records.
#   Columns: nofish, livebait, camper, persons, child, xb, zg, count.
#
# brms model formulas:
#   "count ~ persons + child + camper"
#     (Poisson; baseline model without zero-inflation)
#   "bf(count ~ persons + child + camper, zi ~ child)"
#     (zero-inflated Poisson; zi sub-model predicts structural-zero probability)
##############################################################################

const FISH_URL = "http://paulbuerkner.com/data/fish.csv"

load(::Val{:fish_rj}) = CSV.read(Downloads.download(FISH_URL), DataFrame)

function examples(::Val{:fish_rj})
    d = load(Val(:fish_rj))
    return [
        ("count ~ persons + child + camper", d),
        ("bf(count ~ persons + child + camper, zi ~ child)", d),
    ]
end

##############################################################################
# RJ 2018 — Example 2: Munich rent data (smooth distributional regression)
# Source: Fahrmeir et al. (2013); originally in gamlss.data::rent99.
#   The gamlss.data package dataset is not on Rdatasets; synthetic data below.
#   To obtain the original: data("rent99", package = "gamlss.data")
#
# Dataset structure: ~3 082 Munich rental apartments (1999).
#   rentsqm (rent per m², EUR), area (floor area m²), yearc (year of construction
#   1918–1997), district (1–90, city district), rooms, bath, kitchen, cheating.
#
# brms model formulas:
#   "rentsqm ~ t2(area, yearc) + (1 | district)"
#     (tensor-product spline of area × year of construction; random intercept per district)
#   "bf(rentsqm ~ t2(area, yearc) + (1 | ID1 | district), sigma ~ t2(area, yearc) + (1 | ID1 | district))"
#     (distributional model: both mean and log(sigma) are smooth functions;
#      ID1 label allows correlation between the two sets of district random effects)
##############################################################################

function load(::Val{:rent99})
    # Synthetic equivalent of gamlss.data::rent99
    rng = MersenneTwister(1999)
    n   = 3082
    area  = 20.0 .+ 120.0 .* rand(rng, n)
    yearc = round.(Int, 1918 .+ 79 .* rand(rng, n))
    dist  = rand(rng, 1:90, n)
    # Rent model: newer + larger → higher rent/m²; district random effect
    u_dist = randn(rng, 90) .* 0.4
    mu     = 6.0 .- 0.005 .* (area .- 60) .+ 0.008 .* (yearc .- 1960) .+ u_dist[dist]
    sigma  = exp.(-0.5 .+ 0.003 .* (yearc .- 1960))
    rentsqm = mu .+ sigma .* randn(rng, n)
    return DataFrame(; rentsqm, area, yearc, district = string.(dist))
end

function examples(::Val{:rent99})
    d = load(Val(:rent99))
    return [
        ("rentsqm ~ t2(area, yearc) + (1 | district)", d),
        ("bf(rentsqm ~ t2(area, yearc) + (1 | ID1 | district), sigma ~ t2(area, yearc) + (1 | ID1 | district))", d),
    ]
end

##############################################################################
# RJ 2018 — Example 3: Nonlinear loss development (brms::loss)
# Note: also covered in scripts/examples/brms.jl (:loss).
# Synthetic data generated to match the actuarial triangle structure.
#
# brms model formula:
#   "bf(cum ~ ult * (1 - exp(-(dev/theta)^omega)), ult ~ 1 + (1 | AY), omega ~ 1, theta ~ 1, nl = TRUE)"
#     (Weibull loss development; ult = ultimate loss varies by accident year;
#      omega = shape, theta = scale; nl = TRUE enables the nonlinear formula)
##############################################################################

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

function examples(::Val{:loss_rj})
    d = load(Val(:loss_rj))
    return [
        ("bf(cum ~ ult * (1 - exp(-(dev/theta)^omega)), ult ~ 1 + (1 | AY), omega ~ 1, theta ~ 1, nl = TRUE)", d),
    ]
end

##############################################################################
# RJ 2018 — Example 4: Membership (mm) model — multiple membership
# Synthetic data: 1 000 students nested in 10 schools, each attending two schools
# with weights w1, w2 (summing to 1). Used to demonstrate brms's mm() syntax.
#
# brms model formulas:
#   "y ~ 1 + (1 | mm(s1, s2))"
#     (equal-weight multiple membership; student outcome depends on two schools)
#   "y ~ 1 + (1 | mm(s1, s2, weights = cbind(w1, w2)))"
#     (weighted multiple membership; w1 + w2 = 1 for each student)
##############################################################################

function load(::Val{:multi_member})
    rng      = MersenneTwister(2018)
    nschools = 10
    nstudents = 1000
    u_school = randn(rng, nschools) .* 0.4   # school random effects
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

function examples(::Val{:multi_member})
    d = load(Val(:multi_member))
    return [
        ("y ~ 1 + (1 | mm(s1, s2))", d),
        ("y ~ 1 + (1 | mm(s1, s2, weights = cbind(w1, w2)))", d),
    ]
end
