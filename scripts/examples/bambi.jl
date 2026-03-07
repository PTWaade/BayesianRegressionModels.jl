using CSV, DataFrames, Downloads, Random, Statistics

##############################################################################
# Bambi example: Multiple Linear Regression
# Source: https://bambinos.github.io/bambi/notebooks/ESCS_multiple_regression.html
#
# Dataset: Eugene-Springfield Community Sample (ESCS)
#   604 adults, longitudinal self-report + behavioral measures over 15 years.
#   Outcome: `drugs` — index of self-reported illegal drug use (~1 to 4.3)
#   Predictors: Big Five personality sum-scores
#     o = Openness to experience
#     c = Conscientiousness
#     e = Extraversion
#     a = Agreeableness
#     n = Neuroticism
#
# Bambi model formula:
#   "drugs ~ o + c + e + a + n"
##############################################################################

const ESCS_URL = "https://ndownloader.figshare.com/files/28870722"

load(::Val{:escs}) = CSV.read(Downloads.download(ESCS_URL), DataFrame)

# Returns a vector of (formula, data) pairs for the ESCS multiple regression example.
function examples(::Val{:escs})
    data = load(Val(:escs))
    return [
        ("drugs ~ o + c + e + a + n", data),
    ]
end

##############################################################################
# Bambi example: Regression Splines
# Source: https://bambinos.github.io/bambi/notebooks/splines_cherry_blossoms.html
#
# Dataset: Cherry Blossoms
#   Day of year of first cherry blossom bloom in Japan, years 801–2015.
#   827 observations after removing rows with missing `doy`.
#
# Bambi model formulas (iknots = 15 quantile-based internal knots from `year`):
#   "doy ~ bs(year, knots=iknots, intercept=True)"   (spline basis with explicit intercept)
#   "doy ~ bs(year, knots=iknots)"                   (spline basis, intercept absorbed by model)
#
# Pre-processing: drop rows with missing `doy`; cast `year` to Float.
##############################################################################

const CHERRY_BLOSSOMS_URL = "https://ndownloader.figshare.com/files/31072807"

load(::Val{:cherry_blossoms}) = CSV.read(Downloads.download(CHERRY_BLOSSOMS_URL), DataFrame)

function examples(::Val{:cherry_blossoms})
    data = dropmissing(load(Val(:cherry_blossoms)), :doy)
    data.year = Float64.(data.year)
    return [
        ("doy ~ bs(year, knots=iknots, intercept=True)", data),
        ("doy ~ bs(year, knots=iknots)", data),
    ]
end

##############################################################################
# Bambi example: Hierarchical Linear Regression — Pigs growth
# Source: https://bambinos.github.io/bambi/notebooks/multi-level_regression.html
#
# Dataset: Dietox (from R package geepack)
#   Longitudinal pig weight measurements; 861 observations across 72 pigs.
#
# Bambi model formula:
#   "Weight ~ Time + (Time|Pig)"
#
# Pre-processing: cast `Time` to Float.
##############################################################################

const DIETOX_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv"

load(::Val{:dietox}) = CSV.read(Downloads.download(DIETOX_URL), DataFrame)

function examples(::Val{:dietox})
    data = load(Val(:dietox))
    data.Time = Float64.(data.Time)
    return [
        ("Weight ~ Time + (Time|Pig)", data),
    ]
end

##############################################################################
# Bambi example: Hierarchical Linear Regression — Sleep deprivation
# Source: https://bambinos.github.io/bambi/notebooks/sleepstudy.html
#
# Dataset: Sleepstudy (from R package lme4; Belenky et al. 2003)
#   180 observations: average reaction time (ms) over 10 days of sleep deprivation
#   across 18 subjects.
#
# Bambi model formula:
#   "Reaction ~ 1 + Days + (Days | Subject)"
##############################################################################

const SLEEPSTUDY_URL = "https://ndownloader.figshare.com/files/31181002"

load(::Val{:sleepstudy}) = CSV.read(Downloads.download(SLEEPSTUDY_URL), DataFrame)

function examples(::Val{:sleepstudy})
    data = load(Val(:sleepstudy))
    return [
        ("Reaction ~ 1 + Days + (Days | Subject)", data),
    ]
end

##############################################################################
# Bambi example: Hierarchical Linear Regression — Radon contamination
# Source: https://bambinos.github.io/bambi/notebooks/radon_example.html
#
# Datasets: SRRS2 radon measurements (Minnesota) merged with county uranium data;
#   both from the PyMC examples repository.
#   Derived columns: log_radon = log(activity + 0.1), log_u = log(Uppm).
#
# Bambi model formulas:
#   "log_radon ~ 0 + floor"                         (complete pooling)
#   "log_radon ~ 0 + county:floor"                  (no pooling)
#   "log_radon ~ 1 + (1|county)"                    (partial pooling)
#   "log_radon ~ 1 + floor + (1|county)"            (county-specific intercepts)
#   "log_radon ~ floor + (floor|county)"            (varying slopes)
#   "log_radon ~ floor + log_u + (1|county)"        (with group-level predictor)
##############################################################################

const SRRS2_URL = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/srrs2.dat"
const CTY_URL   = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/cty.dat"

function load(::Val{:radon})
    srrs2 = CSV.read(Downloads.download(SRRS2_URL), DataFrame)
    cty   = CSV.read(Downloads.download(CTY_URL), DataFrame)
    mn = filter(r -> strip(r.state) == "MN", srrs2)
    mn.log_radon = log.(mn.activity .+ 0.1)
    mn.fips = mn.stfips .* 1000 .+ mn.cntyfips
    cty.fips = cty.stfips .* 1000 .+ cty.ctfips
    cty.log_u = log.(cty.Uppm)
    merged = innerjoin(mn, select(cty, [:fips, :log_u]), on = :fips)
    merged.floor = [x == 0 ? "Basement" : "Floor" for x in merged.floor]
    return unique(merged, :idnum)
end

function examples(::Val{:radon})
    data = load(Val(:radon))
    return [
        ("log_radon ~ 0 + floor",                  data),
        ("log_radon ~ 0 + county:floor",            data),
        ("log_radon ~ 1 + (1|county)",              data),
        ("log_radon ~ 1 + floor + (1|county)",      data),
        ("log_radon ~ floor + (floor|county)",      data),
        ("log_radon ~ floor + log_u + (1|county)",  data),
    ]
end

##############################################################################
# Bambi example: Bayesian Workflow — Strack RRR re-analysis
# Source: https://bambinos.github.io/bambi/notebooks/Strack_RRR_re_analysis.html
#
# Dataset: 17 CSV files from the Registered Replication Report of Strack,
#   Martin & Stepper (1988) facial feedback hypothesis.
#   Data directory: docs/notebooks/data/facial_feedback/ in the Bambi repo
#   (https://github.com/bambinos/bambi). Files need preprocessing: skip first
#   two rows, select 22 columns, assign standardised column names, concatenate.
#
# Bambi model formulas:
#   "value ~ condition + (1|uid)"
#   "value ~ condition + age + gender + (1|uid) + (condition|study) + (condition|stimulus)"
##############################################################################

function load(::Val{:strack_rrr})
    error("""
    The Strack RRR dataset consists of 17 CSV files.
    Clone the Bambi repo and load from docs/notebooks/data/facial_feedback/:
      https://github.com/bambinos/bambi
    """)
end

function examples(::Val{:strack_rrr})
    data = load(Val(:strack_rrr))
    return [
        ("value ~ condition + (1|uid)", data),
        ("value ~ condition + age + gender + (1|uid) + (condition|study) + (condition|stimulus)", data),
    ]
end

##############################################################################
# Bambi example: Bayesian Workflow — Police Officer's Dilemma (shooter task)
# Source: https://bambinos.github.io/bambi/notebooks/shooter_crossed_random_ANOVA.html
#
# Dataset: shooter.csv — 3,600 responses from 36 participants (100 trials each).
#   Distributed with the Bambi notebook examples.
#   Derived: rate = 1000 / time; shoot_or_not recoded from response × object type.
#   S() denotes sum-to-zero (effects) coding.
#
# Bambi model formulas:
#   "rate ~ S(race) * S(object) + (S(race) * S(object) | subject)"
#   "rate ~ S(race) * S(object) + (S(race) * S(object) | subject) + (S(object) | target)"
#   "shoot_or_not[shoot] ~ S(race)*S(object) + (S(race)*S(object) | subject) + (S(object) | target)"
##############################################################################

const SHOOTER_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/shooter.csv"

function load(::Val{:shooter})
    data = CSV.read(Downloads.download(SHOOTER_URL), DataFrame)
    data.rate = 1000.0 ./ data.time
    return data
end

function examples(::Val{:shooter})
    data = load(Val(:shooter))
    return [
        ("rate ~ S(race) * S(object) + (S(race) * S(object) | subject)", data),
        ("rate ~ S(race) * S(object) + (S(race) * S(object) | subject) + (S(object) | target)", data),
        ("shoot_or_not[shoot] ~ S(race)*S(object) + (S(race)*S(object) | subject) + (S(object) | target)", data),
    ]
end

##############################################################################
# Bambi example: Fixed, Random Effects and Mundlak Machines
# Source: https://bambinos.github.io/bambi/notebooks/fixed_random.html
#
# Dataset: Synthetically generated from a causal DAG (30 groups, 2000 obs).
#   Columns: x (continuous), y (binary), z (group-level), group, xbar (group mean of x).
#
# Bambi model formulas (all with family="bernoulli"):
#   "y ~ x + z"                          (naive)
#   "y ~ 0 + C(group) + x + z"           (fixed effects)
#   "y ~ x + z + (1|group)"              (multilevel)
#   "y ~ x + z + xbar + (1|group)"       (Mundlak)
##############################################################################

function load(::Val{:fixed_random})
    rng = MersenneTwister(12345)
    n_groups, n_obs = 30, 2000
    group = rand(rng, 1:n_groups, n_obs)
    z = randn(rng, n_groups)[group]
    x = randn(rng, n_obs) .+ z
    xbar = [mean(x[group .== g]) for g in group]
    logit_p = -1.5 .+ 0.5 .* x .+ 0.8 .* z
    y = Int.(rand(rng, n_obs) .< 1 ./ (1 .+ exp.(-logit_p)))
    return DataFrame(; x, y, z, group, xbar)
end

function examples(::Val{:fixed_random})
    data = load(Val(:fixed_random))
    return [
        ("y ~ x + z",                    data),
        ("y ~ 0 + C(group) + x + z",     data),
        ("y ~ x + z + (1|group)",        data),
        ("y ~ x + z + xbar + (1|group)", data),
    ]
end

##############################################################################
# Bambi example: Robust Linear Regression (Student's t)
# Source: https://bambinos.github.io/bambi/notebooks/t_regression.html
#
# Dataset: Synthetically generated — 100 linear observations plus 3 outliers.
#   Columns: x, y.
#
# Bambi model formulas:
#   "y ~ x"  with family="gaussian"
#   "y ~ x"  with family="t"
##############################################################################

function load(::Val{:t_regression})
    rng = MersenneTwister(42)
    x = randn(rng, 100)
    y = 1.0 .+ 2.0 .* x .+ 0.5 .* randn(rng, 100)
    x = vcat(x, [-1.5, 0.0, 1.5])
    y = vcat(y, [20.0, 20.0, 20.0])
    return DataFrame(; x, y)
end

function examples(::Val{:t_regression})
    data = load(Val(:t_regression))
    return [
        ("y ~ x", data),  # family="gaussian"
        ("y ~ x", data),  # family="t"
    ]
end

##############################################################################
# Bambi example: Predict New Groups (Hierarchical models)
# Source: https://bambinos.github.io/bambi/notebooks/predict_new_groups.html
#
# Dataset: OSIC Pulmonary Fibrosis Progression (Kaggle competition subset).
#   Columns: patient (int-encoded), weeks (scaled [0,1]), fvc (scaled [0,1]),
#   smoking_status.
#
# Bambi model formula:
#   "fvc ~ 0 + weeks + smoking_status + (0 + weeks | patient)"
##############################################################################

const OSIC_URL = "https://gist.githubusercontent.com/ucals/2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/43034c39052dcf97d4b894d2ec1bc3f90f3623d9/osic_pulmonary_fibrosis.csv"

function load(::Val{:predict_new_groups})
    df = CSV.read(Downloads.download(OSIC_URL), DataFrame)
    rename!(df, lowercase.(names(df)))
    rename!(df, :smokingstatus => :smoking_status)
    ids = Dict(p => i for (i, p) in enumerate(unique(df.patient)))
    df.patient = [ids[p] for p in df.patient]
    df.weeks = (df.weeks .- minimum(df.weeks)) ./ (maximum(df.weeks) - minimum(df.weeks))
    df.fvc   = (df.fvc   .- minimum(df.fvc))   ./ (maximum(df.fvc)   - minimum(df.fvc))
    return select(df, [:patient, :weeks, :fvc, :smoking_status])
end

function examples(::Val{:predict_new_groups})
    data = load(Val(:predict_new_groups))
    return [
        ("fvc ~ 0 + weeks + smoking_status + (0 + weeks | patient)", data),
    ]
end

##############################################################################
# Bambi example: Polynomial Regression — Learning gravity with Bayesian stats
# Source: https://bambinos.github.io/bambi/notebooks/polynomial_regression.html
#
# Datasets: Three synthetically generated scenarios.
#   1. Falling ball:   t ∈ [0, 2], x ≈ 5 - 0.5·g·t²  (100 obs)
#   2. Projectile:     t ∈ [0, 2], x ≈ 10·t - 0.5·g·t²  (filtered to x ≥ 0)
#   3. Multi-planet:   projectile on Earth, Mars, PlanetX  (columns: Planet, Time, Height)
#
# Bambi model formulas:
#   "x ~ I(t**2) + 1"                          (falling ball)
#   "x ~ {t**2} + 1"                           (falling ball, alternative syntax)
#   "x ~ tsquared + 1"                         (falling ball, pre-computed column)
#   "x ~ I(t**2) + t + 1"                      (projectile)
#   "x ~ poly(t, 2, raw=True)"                 (projectile, raw polynomial)
#   "Height ~ I(Time**2):Planet + Time + 0"    (multi-planet)
##############################################################################

function load(::Val{:polynomial_regression})
    rng = MersenneTwister(0)
    t = range(0, 2, length=100)
    g = 9.8
    x = 5.0 .- 0.5 .* g .* t .^ 2 .+ 0.5 .* randn(rng, 100)
    return DataFrame(; t=collect(t), x, tsquared=collect(t) .^ 2)
end

function examples(::Val{:polynomial_regression})
    rng = MersenneTwister(0)
    g = 9.8
    # Falling ball
    t = collect(range(0, 2, length=100))
    falling = DataFrame(
        t = t,
        tsquared = t .^ 2,
        x = 5.0 .- 0.5 .* g .* t .^ 2 .+ 0.5 .* randn(rng, 100),
    )
    # Projectile
    tp = collect(range(0, 2, length=100))
    xp = 10 .* tp .- 0.5 .* g .* tp .^ 2 .+ 0.3 .* randn(rng, 100)
    projectile = DataFrame(t=tp, x=xp) |> df -> filter(r -> r.x >= 0, df)
    # Multi-planet
    function throw_data(rng, g_planet, planet)
        tt = collect(range(0, 2, length=50))
        h  = 10 .* tt .- 0.5 .* g_planet .* tt .^ 2 .+ 0.3 .* randn(rng, 50)
        DataFrame(Planet=fill(planet, 50), Time=tt, Height=h)
    end
    planets = vcat(throw_data(rng, 9.8, "Earth"),
                   throw_data(rng, 3.7, "Mars"),
                   throw_data(rng, 5.0, "PlanetX"))
    return [
        ("x ~ I(t**2) + 1",               falling),
        ("x ~ {t**2} + 1",                falling),
        ("x ~ tsquared + 1",              falling),
        ("x ~ I(t**2) + t + 1",           projectile),
        ("x ~ poly(t, 2, raw=True)",      projectile),
        ("Height ~ I(Time**2):Planet + Time + 0", planets),
    ]
end

##############################################################################
# Bambi example: Logistic Regression — Vote intention
# Source: https://bambinos.github.io/bambi/notebooks/logistic_regression.html
#
# Dataset: American National Election Studies 2016 pilot (ANES), filtered to
#   Clinton vs. Trump voters (373 obs); loaded via bmb.load_data("ANES").
#
# Bambi model formula (family="bernoulli"):
#   "vote['clinton'] ~ party_id + party_id:age"
##############################################################################

const ANES_URL = "https://ndownloader.figshare.com/files/28870740"

function load(::Val{:logistic_anes})
    df = CSV.read(Downloads.download(ANES_URL), DataFrame)
    return filter(r -> r.vote in ("clinton", "trump"), df)
end

function examples(::Val{:logistic_anes})
    data = load(Val(:logistic_anes))
    return [
        ("vote['clinton'] ~ party_id + party_id:age", data),
    ]
end

##############################################################################
# Bambi example: Logistic Regression — Model comparison with ArviZ
# Source: https://bambinos.github.io/bambi/notebooks/model_comparison.html
#
# Dataset: 1994 US Census adults (adults); loaded via bmb.load_data("adults").
#   Filtered to Black/White; age and hs_week scaled; outcome: income > $50k.
#
# Bambi model formulas (family="bernoulli"):
#   "income['>50K'] ~ sex + race + scale(age) + scale(hs_week)"
#   "income['>50K'] ~ sex + race + scale(age) + I(scale(age)**2) + scale(hs_week) + I(scale(hs_week)**2)"
#   "income['>50K'] ~ sex + race + scale(age) + I(scale(age)**2) + I(scale(age)**3) + scale(hs_week) + I(scale(hs_week)**2) + I(scale(hs_week)**3)"
##############################################################################

const ADULTS_URL = "https://ndownloader.figshare.com/files/28870743"

function load(::Val{:model_comparison})
    df = CSV.read(Downloads.download(ADULTS_URL), DataFrame)
    df = filter(r -> r.race in ("White", "Black"), df)
    df.age = Float64.(df.age)
    return df
end

function examples(::Val{:model_comparison})
    data = load(Val(:model_comparison))
    return [
        ("income['>50K'] ~ sex + race + scale(age) + scale(hs_week)", data),
        ("income['>50K'] ~ sex + race + scale(age) + I(scale(age)**2) + scale(hs_week) + I(scale(hs_week)**2)", data),
        ("income['>50K'] ~ sex + race + scale(age) + I(scale(age)**2) + I(scale(age)**3) + scale(hs_week) + I(scale(hs_week)**2) + I(scale(hs_week)**3)", data),
    ]
end

##############################################################################
# Bambi example: Hierarchical Logistic Regression — Binomial family
# Source: https://bambinos.github.io/bambi/notebooks/hierarchical_binomial_bambi.html
#
# Dataset: Baseball batting statistics; loaded via bmb.load_data("batting").
#   Filtered to 2016+, first 15 obs; batting_avg = H / AB.
#
# Bambi model formulas (family="binomial"):
#   "p(H, AB) ~ 0 + playerID"      (non-hierarchical)
#   "p(H, AB) ~ 1 + (1|playerID)"  (hierarchical)
##############################################################################

const BATTING_URL = "https://ndownloader.figshare.com/files/29749140"

function load(::Val{:hierarchical_binomial})
    df = CSV.read(Downloads.download(BATTING_URL), DataFrame)
    df = dropmissing(df)
    df = filter(r -> r.AB > 0, df)
    df = filter(r -> r.yearID >= 2016, df)
    df.batting_avg = df.H ./ df.AB
    return first(df, 15)
end

function examples(::Val{:hierarchical_binomial})
    data = load(Val(:hierarchical_binomial))
    return [
        ("p(H, AB) ~ 0 + playerID", data),
        ("p(H, AB) ~ 1 + (1|playerID)", data),
    ]
end

##############################################################################
# Bambi example: Binary Response Regression — Alternative link functions
# Source: https://bambinos.github.io/bambi/notebooks/alternative_links_binary.html
#
# Dataset: Bliss (1935) beetle mortality data — 8 aggregated observations.
#   Columns: x (log dose), n (beetles exposed), y (beetles killed).
#
# Bambi model formula (family="binomial", p(y, n) ~ x):
#   "p(y, n) ~ x"  with link="logit"   (default)
#   "p(y, n) ~ x"  with link="probit"
#   "p(y, n) ~ x"  with link="cloglog"
##############################################################################

function load(::Val{:alternative_links})
    return DataFrame(
        x = [1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839],
        n = [59, 60, 62, 56, 63, 59, 62, 60],
        y = [ 6,  13,  18,  28,  52,  53,  61,  60],
    )
end

function examples(::Val{:alternative_links})
    data = load(Val(:alternative_links))
    return [
        ("p(y, n) ~ x", data),  # link="logit"
        ("p(y, n) ~ x", data),  # link="probit"
        ("p(y, n) ~ x", data),  # link="cloglog"
    ]
end

##############################################################################
# Bambi example: Wald and Gamma Regression — Australian insurance claims
# Source: https://bambinos.github.io/bambi/notebooks/wald_gamma_glm.html
#
# Dataset: Australian car insurance claims 2004-2005; loaded via bmb.load_data("carclaims").
#   Filtered to claimcst0 > 0 (records with a claim).
#
# Bambi model formulas:
#   "claimcst0 ~ C(agecat) + gender + area"   with family="wald",  link="log"
#   "claimcst0 ~ agecat + gender + area"      with family="gamma", link="log", categorical="agecat"
##############################################################################

const CARCLAIMS_URL = "https://ndownloader.figshare.com/files/28870713"

function load(::Val{:wald_gamma})
    df = CSV.read(Downloads.download(CARCLAIMS_URL), DataFrame)
    return filter(r -> r.claimcst0 > 0, df)
end

function examples(::Val{:wald_gamma})
    data = load(Val(:wald_gamma))
    return [
        ("claimcst0 ~ C(agecat) + gender + area", data),  # family="wald",  link="log"
        ("claimcst0 ~ agecat + gender + area",    data),  # family="gamma", link="log"
    ]
end

##############################################################################
# Bambi example: Negative Binomial Regression — Students absence
# Source: https://bambinos.github.io/bambi/notebooks/negative_binomial.html
#
# Dataset: nb_data.dta — 314 high school juniors; daysabs, prog, math.
#   Source: UCLA IDRE (https://stats.idre.ucla.edu/stat/stata/dae/nb_data.dta)
#   NOTE: Stata .dta format — requires ReadStatTables.jl or similar to load.
#   prog recoded: 1→"General", 2→"Academic", 3→"Vocational".
#
# Bambi model formulas (family="negativebinomial"):
#   "daysabs ~ 0 + prog + scale(math)"
#   "daysabs ~ 0 + prog + scale(math) + prog:scale(math)"
##############################################################################

const NB_DATA_URL = "https://stats.idre.ucla.edu/stat/stata/dae/nb_data.dta"

function load(::Val{:negative_binomial})
    error("""
    nb_data.dta is a Stata file. Load with ReadStatTables.jl:
      using ReadStatTables
      df = DataFrame(readstat("$NB_DATA_URL"))
    """)
end

function examples(::Val{:negative_binomial})
    data = load(Val(:negative_binomial))
    return [
        ("daysabs ~ 0 + prog + scale(math)",                   data),
        ("daysabs ~ 0 + prog + scale(math) + prog:scale(math)", data),
    ]
end

##############################################################################
# Bambi example: Count Regression with Variable Exposure — Offsets
# Source: https://bambinos.github.io/bambi/notebooks/count_roaches.html
#
# Dataset: Roaches pest-control study (Gelman et al., Regression and Other Stories).
#   roach1 rescaled by /100.
#
# Bambi model formulas:
#   "y ~ roach1 + treatment + senior + offset(log(exposure2))"   family="poisson"
#   "y ~ roach1 + treatment + senior + offset(log(exposure2))"   family="negativebinomial"
##############################################################################

const ROACHES_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/roaches.csv"

function load(::Val{:count_roaches})
    df = CSV.read(Downloads.download(ROACHES_URL), DataFrame)
    df.roach1 = df.roach1 ./ 100
    return df
end

function examples(::Val{:count_roaches})
    data = load(Val(:count_roaches))
    return [
        ("y ~ roach1 + treatment + senior + offset(log(exposure2))", data),  # family="poisson"
        ("y ~ roach1 + treatment + senior + offset(log(exposure2))", data),  # family="negativebinomial"
    ]
end

##############################################################################
# Bambi example: Beta Regression — Multiple datasets
# Source: https://bambinos.github.io/bambi/notebooks/beta_regression.html
#
# Datasets:
#   1. Synthetic beta probabilities (intercept-only demo)
#   2. Synthetic coin dirt data: delta_d = heads_bias_dirt - tails_bias_dirt
#   3. Baseball batting averages (bmb.load_data("batting"), filtered to AB > 100, 1990-2018)
#      batting_avg = H/AB; batting_avg_shift = previous year's batting_avg per player
#
# Bambi model formulas (all family="beta"):
#   "probabilities ~ 1"
#   "p ~ delta_d"
#   "batting_avg ~ 1"
#   "batting_avg ~ batting_avg_shift"
##############################################################################

load(::Val{:beta_regression}) = CSV.read(Downloads.download(BATTING_URL), DataFrame)

function examples(::Val{:beta_regression})
    rng = MersenneTwister(1)
    # 1. Synthetic beta probabilities
    probs_df = DataFrame(probabilities = rand(rng, 500) .* 0.8 .+ 0.1)
    # 2. Synthetic coin dirt
    n = 200
    heads_dirt = abs.(randn(rng, n) .* 25)
    tails_dirt = abs.(randn(rng, n) .* 25)
    p_vals = clamp.(0.5 .+ (heads_dirt .- tails_dirt) ./ 200, 0.01, 0.99)
    coin_df = DataFrame(delta_d = heads_dirt .- tails_dirt, p = p_vals)
    # 3. Batting averages
    batting = load(Val(:beta_regression))
    batting = dropmissing(filter(r -> r.AB > 100 && r.yearID >= 1990, batting))
    batting.batting_avg = batting.H ./ batting.AB
    sort!(batting, [:playerID, :yearID])
    batting.batting_avg_shift = vcat([missing],
        [batting.playerID[i] == batting.playerID[i-1] ?
         batting.batting_avg[i-1] : missing for i in 2:nrow(batting)])
    batting = dropmissing(batting, :batting_avg_shift)
    return [
        ("probabilities ~ 1",              probs_df),
        ("p ~ delta_d",                    coin_df),
        ("batting_avg ~ 1",                batting),
        ("batting_avg ~ batting_avg_shift", batting),
    ]
end

##############################################################################
# Bambi example: Categorical Outcomes — Non-numeric outcomes
# Source: https://bambinos.github.io/bambi/notebooks/categorical_regression.html
#
# Datasets:
#   1. Synthetic toy data: three Gaussian classes (N(-2.5,1.2), N(0,0.5), N(2.5,1.2))
#   2. Iris (seaborn): 150 observations, four morphological predictors, three species
#   3. Alligator food choice (Agresti 2002, Ch.8): 64 obs, columns: choice, length, sex
#
# Bambi model formulas (family="categorical"):
#   "y ~ x"
#   "species ~ sepal_length + sepal_width + petal_length + petal_width"
#   "choice ~ length + sex"
##############################################################################

const IRIS_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv"

load(::Val{:categorical_regression}) = CSV.read(Downloads.download(IRIS_URL), DataFrame)

function examples(::Val{:categorical_regression})
    rng = MersenneTwister(7)
    # 1. Synthetic toy data
    toy_df = vcat(
        DataFrame(x = randn(rng, 50) .- 2.5, y = fill("A", 50)),
        DataFrame(x = randn(rng, 50) .* 0.5, y = fill("B", 50)),
        DataFrame(x = randn(rng, 50) .+ 2.5, y = fill("C", 50)),
    )
    # 2. Iris
    iris = load(Val(:categorical_regression))
    rename!(iris, "Species" => "species", "Sepal.Length" => "sepal_length",
            "Sepal.Width" => "sepal_width", "Petal.Length" => "petal_length",
            "Petal.Width" => "petal_width")
    # 3. Alligator food choice (Table 8.1, Agresti 2002)
    alligator = DataFrame(
        choice = ["Fish","Invertebrates","Fish","Invertebrates","Fish","Other",
                  "Fish","Invertebrates","Fish","Other","Fish","Invertebrates",
                  "Fish","Fish","Invertebrates","Other","Fish","Invertebrates",
                  "Invertebrates","Fish","Invertebrates","Other","Fish","Fish",
                  "Invertebrates","Fish","Invertebrates","Fish","Other","Fish",
                  "Invertebrates","Fish","Fish","Invertebrates","Other","Fish",
                  "Invertebrates","Invertebrates","Fish","Other","Fish","Fish",
                  "Invertebrates","Fish","Invertebrates","Other","Invertebrates",
                  "Fish","Fish","Invertebrates","Fish","Other","Fish","Fish",
                  "Invertebrates","Fish","Invertebrates","Fish","Other","Fish",
                  "Invertebrates","Invertebrates","Fish","Other"],
        length = vcat(fill(1.24, 5), fill(1.45, 5), fill(1.63, 5), fill(1.78, 5),
                      fill(1.98, 5), fill(2.36, 5), fill(2.79, 5), fill(2.99, 5),
                      fill(3.25, 5), fill(3.28, 5), fill(3.33, 5), fill(3.78, 5),
                      fill(3.78, 4)),
        sex = vcat([fill("M",3); fill("F",2)], [fill("M",3); fill("F",2)],
                   [fill("M",3); fill("F",2)], [fill("M",3); fill("F",2)],
                   [fill("M",3); fill("F",2)], [fill("M",3); fill("F",2)],
                   [fill("M",3); fill("F",2)], [fill("M",3); fill("F",2)],
                   [fill("M",3); fill("F",2)], [fill("M",3); fill("F",2)],
                   [fill("M",3); fill("F",2)], [fill("M",3); fill("F",2)],
                   [fill("M",2); fill("F",2)]),
    )
    return [
        ("y ~ x",                                                              toy_df),
        ("species ~ sepal_length + sepal_width + petal_length + petal_width",  iris),
        ("choice ~ length + sex",                                              alligator),
    ]
end

##############################################################################
# Bambi example: Circular Regression — Directional statistics
# Source: https://bambinos.github.io/bambi/notebooks/circular_regression.html
#
# Dataset: Periwinkles — 31 sea snails relocated; direction and distance columns.
#   Loaded via bmb.load_data("periwinkles"). distance cast to Float64.
#
# Bambi model formulas:
#   "direction ~ distance"   with family="vonmises"
#   "direction ~ distance"   with family="gaussian"  (comparison model)
##############################################################################

const PERIWINKLES_URL = "https://ndownloader.figshare.com/files/34446077"

function load(::Val{:circular_regression})
    df = CSV.read(Downloads.download(PERIWINKLES_URL), DataFrame)
    df.distance = Float64.(df.distance)
    return df
end

function examples(::Val{:circular_regression})
    data = load(Val(:circular_regression))
    return [
        ("direction ~ distance", data),  # family="vonmises"
        ("direction ~ distance", data),  # family="gaussian"
    ]
end

##############################################################################
# Bambi example: Quantile Regression — Percentile modeling
# Source: https://bambinos.github.io/bambi/notebooks/quantile_regression.html
#
# Dataset: BMI measurements of Dutch children and young adults (bmi.csv).
#   Distributed with the Bambi notebook examples.
#   knots = quantile(age, LinRange(0,1,10))[2:end-1]  (8 interior knots)
#
# Bambi model formulas (family="asymmetriclaplace", varying kappa):
#   "bmi ~ bs(age, knots=knots)"   for quantiles 0.1, 0.5, 0.9
#   "bmi ~ bs(age, knots=knots)"   with family="gaussian"  (comparison)
##############################################################################

const BMI_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/bmi.csv"

load(::Val{:quantile_regression}) = CSV.read(Downloads.download(BMI_URL), DataFrame)

function examples(::Val{:quantile_regression})
    data = load(Val(:quantile_regression))
    return [
        ("bmi ~ bs(age, knots=knots)", data),  # family="asymmetriclaplace", kappa=q0.1
        ("bmi ~ bs(age, knots=knots)", data),  # family="asymmetriclaplace", kappa=q0.5
        ("bmi ~ bs(age, knots=knots)", data),  # family="asymmetriclaplace", kappa=q0.9
        ("bmi ~ bs(age, knots=knots)", data),  # family="gaussian" (comparison)
    ]
end

##############################################################################
# Bambi example: MrP — Multilevel Regression and Post-stratification
# Source: https://bambinos.github.io/bambi/notebooks/mister_p.html
#
# Datasets:
#   1. CCES 2018 Common Content (Schaffner, Ansolabehere & Luks 2018), Harvard Dataverse.
#      Outcome: binary abortion opinion (from CC18_321d).
#   2. State-level predictors CSV (repvote etc.), distributed with Bambi notebook.
#
# Bambi model formulas:
#   "p(abortion, n) ~ male + repvote + (1|state) + (1|eth) + (1|edu) + (1|male:eth) + (1|edu:age) + (1|edu:eth)"
#     (family="binomial")
##############################################################################

function load(::Val{:mister_p})
    error("""
    The MrP example uses the CCES 2018 survey dataset from Harvard Dataverse
    (https://dataverse.harvard.edu/). Download and preprocess per the notebook:
      https://bambinos.github.io/bambi/notebooks/mister_p.html
    """)
end

function examples(::Val{:mister_p})
    data = load(Val(:mister_p))
    return [
        ("p(abortion, n) ~ male + repvote + (1|state) + (1|eth) + (1|edu) + (1|male:eth) + (1|edu:age) + (1|edu:eth)", data),
    ]
end

##############################################################################
# Bambi example: Zero Inflated Models — Overdispersed outcomes
# Source: https://bambinos.github.io/bambi/notebooks/zero_inflated_regression.html
#
# Dataset: Fish catch survey — 250 groups at a state park; count, livebait, camper,
#   persons, child. Filtered to count < 60 (248 obs).
#   Source: https://stats.idre.ucla.edu/stat/data/fish.csv
#
# Bambi model formulas (dual-formula models: mu + psi component):
#   mu:  "count ~ livebait + camper + persons + child"   family="zero_inflated_poisson"
#   psi: "psi ~ livebait + camper + persons + child"
#   mu:  "count ~ livebait + camper + persons + child"   family="hurdle_poisson"
#   psi: "psi ~ livebait + camper + persons + child"
##############################################################################

const FISH_URL = "https://stats.idre.ucla.edu/stat/data/fish.csv"

function load(::Val{:zero_inflated})
    df = CSV.read(Downloads.download(FISH_URL), DataFrame)
    return filter(r -> r.count < 60, df)
end

function examples(::Val{:zero_inflated})
    data = load(Val(:zero_inflated))
    formula = "count ~ livebait + camper + persons + child"
    psi_formula = "psi ~ livebait + camper + persons + child"
    return [
        (formula, data),      # mu component, family="zero_inflated_poisson"
        (psi_formula, data),  # psi component, family="zero_inflated_poisson"
        (formula, data),      # mu component, family="hurdle_poisson"
        (psi_formula, data),  # psi component, family="hurdle_poisson"
    ]
end

##############################################################################
# Bambi example: Ordinal Regression — Ordered categories
# Source: https://bambinos.github.io/bambi/notebooks/ordinal_regression.html
#
# Datasets:
#   1. Trolley (McElreath, Statistical Rethinking): 9930 obs; response = 1–7 moral intuition.
#      action, intention, contact as unordered categorical; response as ordered categorical.
#   2. IBM HR Employee Attrition (Kaggle): filtered to Attrition="No";
#      YearsAtCompany as ordered categorical.
#      NOTE: hr_employee_attrition.tsv.txt not publicly hosted; see Kaggle or Bambi repo.
#
# Bambi model formulas:
#   "response ~ 0"                                                    (cumulative, intercept-only)
#   "response ~ 0 + action + intention + contact + action:intention + contact:intention"  (cumulative)
#   "YearsAtCompany ~ 0 + TotalWorkingYears"                          (sequential/sratio)
##############################################################################

const TROLLEY_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Trolley.csv"
const HR_ATTRITION_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/hr_employee_attrition.tsv.txt"

function load(::Val{:ordinal_regression})
    df = CSV.read(Downloads.download(TROLLEY_URL), DataFrame; delim=";")
    for col in (:action, :intention, :contact)
        df[!, col] = string.(df[!, col])
    end
    return df
end

function examples(::Val{:ordinal_regression})
    trolley = load(Val(:ordinal_regression))
    hr = CSV.read(Downloads.download(HR_ATTRITION_URL), DataFrame; delim="\t")
    hr = filter(r -> r.Attrition == "No", hr)
    return [
        ("response ~ 0", trolley),
        ("response ~ 0 + action + intention + contact + action:intention + contact:intention", trolley),
        ("YearsAtCompany ~ 0 + TotalWorkingYears", hr),
    ]
end

##############################################################################
# Bambi example: Distributional Models — Multiple parameters
# Source: https://bambinos.github.io/bambi/notebooks/distributional_models.html
#
# Datasets:
#   1. Synthetic Gamma data: x ~ Uniform(-1.5, 1.5), y ~ Gamma with varying alpha (200 obs).
#   2. Bikes dataset — hourly bike counts; loaded via bmb.load_data("bikes"),
#      subsampled every 50th row (348 obs).
#
# Bambi model formulas (compound formula: main + auxiliary parameter):
#   "y ~ x"                                                (constant alpha)
#   "y ~ x"  +  "alpha ~ x"                               (varying alpha)
#   "count ~ 0 + bs(hour, 8, intercept=True)"  +  "alpha ~ 0 + bs(hour, 8, intercept=True)"
##############################################################################

const BIKES_URL = "https://ndownloader.figshare.com/files/38737026"

load(::Val{:distributional_models}) = CSV.read(Downloads.download(BIKES_URL), DataFrame)

function examples(::Val{:distributional_models})
    rng = MersenneTwister(121195)
    x = rand(rng, 200) .* 3 .- 1.5
    alpha = exp.(0.5 .+ 1.5 .* x)
    mu = exp.(1.0 .+ 0.5 .* x)
    y = [rand(rng, Gamma(a, m / a)) for (a, m) in zip(alpha, mu)]  # Gamma(shape, scale)
    synth = DataFrame(; x, y)
    bikes = load(Val(:distributional_models))
    bikes = bikes[1:50:nrow(bikes), :]
    return [
        ("y ~ x",                                    synth),  # constant alpha
        ("y ~ x  +  alpha ~ x",                      synth),  # varying alpha (compound formula)
        ("count ~ 0 + bs(hour, 8, intercept=True)  +  alpha ~ 0 + bs(hour, 8, intercept=True)", bikes),
    ]
end

##############################################################################
# Bambi example: Gaussian Processes — One dimension
# Source: https://bambinos.github.io/bambi/notebooks/hsgp_1d.html
#
# Datasets:
#   1. Synthetic smooth-function data (100 obs, spline-based simulation).
#   2. GAM simulation data (gam_data.csv, 300 obs with categorical grouping variable fac).
#      Simulated with gamSim() from R package mgcv; distributed with Bambi notebook examples.
#
# Bambi model formulas:
#   "y ~ 0 + hsgp(x, m=10, c=2)"
#   "y ~ 0 + hsgp(x, m=10, c=2, centered=True)"
#   "y ~ 0 + hsgp(x2, by=fac, m=12, c=1.5)"
#   "y ~ 0 + hsgp(x2, by=fac, m=12, c=1.5, share_cov=False)"
##############################################################################

const GAM_DATA_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/gam_data.csv"

function load(::Val{:hsgp_1d})
    rng = MersenneTwister(0)
    x = collect(range(-3, 3, length=100))
    y = sin.(x) .+ 0.3 .* randn(rng, 100)
    return DataFrame(; x, y)
end

function examples(::Val{:hsgp_1d})
    synth = load(Val(:hsgp_1d))
    gam_data = CSV.read(Downloads.download(GAM_DATA_URL), DataFrame)
    gam_data.fac = string.(gam_data.fac)
    return [
        ("y ~ 0 + hsgp(x, m=10, c=2)",                            synth),
        ("y ~ 0 + hsgp(x, m=10, c=2, centered=True)",             synth),
        ("y ~ 0 + hsgp(x2, by=fac, m=12, c=1.5)",                 gam_data),
        ("y ~ 0 + hsgp(x2, by=fac, m=12, c=1.5, share_cov=False)", gam_data),
    ]
end

##############################################################################
# Bambi example: Gaussian Processes — Multiple dimensions
# Source: https://bambinos.github.io/bambi/notebooks/hsgp_2d.html
#
# Datasets:
#   1. Synthetic 2D GP data (multivariate normal draws on a grid).
#   2. poisson_data.csv — count data with lat/lon coordinates and year category.
#      Distributed with Bambi notebook examples.
#
# Bambi model formulas:
#   "outcome ~ 0 + hsgp(x, y, c=1.5, m=10)"
#   "outcome ~ 0 + hsgp(x, y, by=group, c=1.5, m=10)"
#   "outcome ~ 0 + hsgp(x, y, by=group, c=1.5, m=10, share_cov=False)"
#   "outcome ~ 0 + hsgp(x, y, c=1.5, m=10, iso=False)"                   (anisotropic)
#   "Count ~ 0 + Year + X1:Year + (1|Site) + hsgp(Lon, Lat, by=Year, m=5, c=1.5)"
##############################################################################

const POISSON_DATA_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/poisson_data.csv"

function load(::Val{:hsgp_2d})
    rng = MersenneTwister(1)
    n = 100
    x = randn(rng, n)
    y = randn(rng, n)
    outcome = sin.(x) .* cos.(y) .+ 0.2 .* randn(rng, n)
    group = rand(rng, ["A", "B"], n)
    return DataFrame(; x, y, outcome, group)
end

function examples(::Val{:hsgp_2d})
    synth = load(Val(:hsgp_2d))
    poisson = CSV.read(Downloads.download(POISSON_DATA_URL), DataFrame)
    poisson.Year = string.(poisson.Year)
    return [
        ("outcome ~ 0 + hsgp(x, y, c=1.5, m=10)",                          synth),
        ("outcome ~ 0 + hsgp(x, y, by=group, c=1.5, m=10)",                 synth),
        ("outcome ~ 0 + hsgp(x, y, by=group, c=1.5, m=10, share_cov=False)", synth),
        ("outcome ~ 0 + hsgp(x, y, c=1.5, m=10, iso=False)",                synth),
        ("Count ~ 0 + Year + X1:Year + (1|Site) + hsgp(Lon, Lat, by=Year, m=5, c=1.5)", poisson),
    ]
end

##############################################################################
# Bambi example: Survival Modeling — Cat adoption times
# Source: https://bambinos.github.io/bambi/notebooks/survival_model.html
#
# Dataset: Austin Cats (City of Austin Open Data / McElreath rethinking repo).
#   days_to_event scaled to months (/ 31); adopt = "right" for non-adoptions,
#   "none" for adoptions; color_id = 1 (black) / 0 (other).
#
# Bambi model formulas (family="exponential", link="log"):
#   "censored(days_to_event / 31, adopt) ~ 1"
#   "censored(days_to_event / 31, adopt) ~ 0 + color_id"
##############################################################################

const AUSTIN_CATS_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/AustinCats.csv"

function load(::Val{:survival_model})
    df = CSV.read(Downloads.download(AUSTIN_CATS_URL), DataFrame)
    df.adopt    = [r.out_event == "Adoption" ? "none" : "right" for r in eachrow(df)]
    df.color_id = [r.color == "Black" ? 1 : 0 for r in eachrow(df)]
    return select(df, [:days_to_event, :adopt, :color_id])
end

function examples(::Val{:survival_model})
    data = load(Val(:survival_model))
    return [
        ("censored(days_to_event / 31, adopt) ~ 1",         data),
        ("censored(days_to_event / 31, adopt) ~ 0 + color_id", data),
    ]
end

##############################################################################
# Bambi example: Survival Models in Discrete Time
# Source: https://bambinos.github.io/bambi/notebooks/survival_discrete_time_notebook.html
#
# Datasets:
#   1. Simulation study data (synthetic, bernoulli).
#   2. child.csv — Swedish 19th-century parish child mortality records.
#      Distributed with the Bambi notebook examples.
#      Aggregated to (events, at_risk) per stratum via person-period expansion.
#
# Bambi model formulas:
#   "event ~ treatment + age + time"                                    (simulation; family="bernoulli", link="cloglog")
#   "p(events, at_risk) ~ sex + socBranch + period + scale(birth_decade)"       (family="binomial", link="cloglog")
#   "p(events, at_risk) ~ sex + socBranch + bs(period, df=4) + scale(birth_decade)"  (spline baseline)
#   "events ~ sex + socBranch + period + scale(birth_decade) + offset(log(at_risk))"  (Poisson alternative)
##############################################################################

const CHILD_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/child.csv"

function load(::Val{:survival_discrete_time})
    rng = MersenneTwister(99)
    n = 200
    treatment = rand(rng, 0:1, n)
    age       = rand(rng, 20:60, n)
    time      = rand(rng, 1:10, n)
    event     = Int.(rand(rng, n) .< 0.1 .+ 0.05 .* treatment)
    return DataFrame(; treatment, age, time, event)
end

function examples(::Val{:survival_discrete_time})
    synth = load(Val(:survival_discrete_time))
    child = CSV.read(Downloads.download(CHILD_URL), DataFrame)
    return [
        ("event ~ treatment + age + time",                                                          synth),
        ("p(events, at_risk) ~ sex + socBranch + period + scale(birth_decade)",                     child),
        ("p(events, at_risk) ~ sex + socBranch + bs(period, df=4) + scale(birth_decade)",           child),
        ("events ~ sex + socBranch + period + scale(birth_decade) + offset(log(at_risk))",          child),
    ]
end

##############################################################################
# Bambi example: Survival Models in Continuous Time — Competing hazards
# Source: https://bambinos.github.io/bambi/notebooks/survival_continuous_time_notebook.html
#
# Datasets:
#   1. Synthetic Weibull survival data (1000 subjects; columns: time, censoring, treatment, age).
#   2. retention.csv — 3770 employees, monthly tenure data; distributed with Bambi notebook.
#
# Bambi model formulas:
#   "censored(time, censoring) ~ treatment + age"        (family="weibull" and family="exponential")
#   "censored(month, censoring) ~ C(gender) + C(level) + C(field) + sentiment + intention"
#   "censored(month, censoring) ~ C(gender) + C(level) + sentiment + intention + (1|field)"
##############################################################################

const RETENTION_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/retention.csv"

function load(::Val{:survival_continuous_time})
    rng = MersenneTwister(42)
    n = 1000
    treatment = rand(rng, 0:1, n)
    age       = randn(rng, n)
    log_scale = 3.0 .+ 0.6 .* treatment .- 0.4 .* age
    time      = [rand(rng, Weibull(2.0, exp(s))) for s in log_scale]
    censoring = [t > 20 ? "right" : "none" for t in time]
    time      = min.(time, 20.0)
    return DataFrame(; time, censoring, treatment, age)
end

function examples(::Val{:survival_continuous_time})
    synth     = load(Val(:survival_continuous_time))
    retention = CSV.read(Downloads.download(RETENTION_URL), DataFrame)
    return [
        ("censored(time, censoring) ~ treatment + age",                                                     synth),
        ("censored(month, censoring) ~ C(gender) + C(level) + C(field) + sentiment + intention",           retention),
        ("censored(month, censoring) ~ C(gender) + C(level) + sentiment + intention + (1|field)",          retention),
    ]
end

##############################################################################
# Bambi example: Orthogonal Polynomial Regression — Avoid multicollinearity
# Source: https://bambinos.github.io/bambi/notebooks/orthogonal_polynomial_reg.html
#
# Datasets:
#   1. Synthetic projectile motion (t, x columns; filtered to x ≥ 0).
#   2. MPG dataset (seaborn / EPA): 1970–1982 automobiles; rows with missing horsepower dropped.
#
# Bambi model formulas:
#   "x ~ I(t**2) + t + 1"            (explicit polynomial)
#   "x ~ poly(t, 2) + 1"             (orthogonal polynomial, degree 2)
#   "mpg ~ horsepower"               (linear baseline)
#   "mpg ~ poly(horsepower, 2)"      (quadratic)
#   "mpg ~ poly(horsepower, {degree})"   for degree ∈ 1:9 (model comparison)
##############################################################################

const MPG_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"

load(::Val{:orthogonal_polynomial}) = CSV.read(Downloads.download(MPG_URL), DataFrame)

function examples(::Val{:orthogonal_polynomial})
    rng = MersenneTwister(0)
    g = 9.8
    tp = collect(range(0, 2, length=100))
    xp = 10 .* tp .- 0.5 .* g .* tp .^ 2 .+ 0.3 .* randn(rng, 100)
    projectile = DataFrame(t=tp, x=xp) |> df -> filter(r -> r.x >= 0, df)
    mpg = dropmissing(load(Val(:orthogonal_polynomial)), [:horsepower, :mpg])
    return [
        ("x ~ I(t**2) + t + 1",           projectile),
        ("x ~ poly(t, 2) + 1",            projectile),
        ("mpg ~ horsepower",              mpg),
        ("mpg ~ poly(horsepower, 2)",     mpg),
        ("mpg ~ poly(horsepower, degree)", mpg),  # degree ∈ 1:9
    ]
end

##############################################################################
# Bambi example: Predictions — New data predictions
# Source: https://bambinos.github.io/bambi/notebooks/plot_predictions.html
#
# Datasets:
#   1. mtcars (bmb.load_data("mtcars")): 32 automobiles; hp→Float32, cyl→category.
#   2. Student absences (UCLA nb_data.dta) — see :negative_binomial for loading note.
#   3. IMDB movies (ggplot2movies): 28,819 films; style from Action/Comedy/Drama flags.
#   4. Synthetic Gamma data (200 obs; see :distributional_models).
#
# Bambi model formulas:
#   "mpg ~ 0 + hp * wt + cyl + gear"                       (linear regression)
#   "daysabs ~ 0 + prog + scale(math) + prog:scale(math)"  (negative binomial)
#   "certified_fresh ~ 0 + scale(length) * style"          (logistic, family="bernoulli")
#   "y ~ x"  +  "alpha ~ x"                                (distributional Gamma)
##############################################################################

const MTCARS_URL = "https://ndownloader.figshare.com/files/40208785"
const IMDB_URL   = "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2movies/movies.csv"

load(::Val{:plot_predictions}) = CSV.read(Downloads.download(MTCARS_URL), DataFrame)

function examples(::Val{:plot_predictions})
    mtcars = load(Val(:plot_predictions))
    mtcars.hp  = Float32.(mtcars.hp)
    mtcars.cyl = [c <= 4 ? "low" : c <= 6 ? "medium" : "high" for c in mtcars.cyl]
    mtcars.gear = [g == 3 ? "A" : g == 4 ? "B" : "C" for g in mtcars.gear]
    movies = dropmissing(filter(r -> r.length < 240,
                CSV.read(Downloads.download(IMDB_URL), DataFrame)))
    movies.style = [r.Action == 1 ? "Action" : r.Comedy == 1 ? "Comedy" : "Drama"
                    for r in eachrow(movies)]
    movies.certified_fresh = Int.(movies.rating .>= 8)
    rng = MersenneTwister(121195)
    x = rand(rng, 200) .* 3 .- 1.5
    y = exp.(1.0 .+ 0.5 .* x .+ 0.2 .* randn(rng, 200))
    synth = DataFrame(; x, y)
    return [
        ("mpg ~ 0 + hp * wt + cyl + gear",                      mtcars),
        ("daysabs ~ 0 + prog + scale(math) + prog:scale(math)",  DataFrame()),  # see :negative_binomial
        ("certified_fresh ~ 0 + scale(length) * style",         movies),
        ("y ~ x  +  alpha ~ x",                                  synth),
    ]
end

##############################################################################
# Bambi example: Comparisons — Group comparisons
# Source: https://bambinos.github.io/bambi/notebooks/plot_comparisons.html
#
# Datasets:
#   1. Fish catch survey (UCLA) — see :zero_inflated for details.
#   2. Titanic survival (Stat2Data via Rdatasets): PClass and SexCode as ordered categorical.
#
# Bambi model formulas:
#   "count ~ livebait + camper + persons + child"   (family="zero_inflated_poisson")
#   "Survived ~ PClass * SexCode * Age"             (family="bernoulli")
##############################################################################

const TITANIC_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Titanic.csv"

load(::Val{:plot_comparisons}) = CSV.read(Downloads.download(FISH_URL), DataFrame)

function examples(::Val{:plot_comparisons})
    fish    = filter(r -> r.count < 60, load(Val(:plot_comparisons)))
    titanic = dropmissing(CSV.read(Downloads.download(TITANIC_URL), DataFrame))
    return [
        ("count ~ livebait + camper + persons + child", fish),
        ("Survived ~ PClass * SexCode * Age",           titanic),
    ]
end

##############################################################################
# Bambi example: Slopes — Response changes
# Source: https://bambinos.github.io/bambi/notebooks/plot_slopes.html
#
# Dataset: Bangladesh well-switching data (carData via Rdatasets): 3020 obs.
#   dist100 = distance / 100; educ4 = education / 4.
#
# Bambi model formulas (family="bernoulli"):
#   "switch ~ dist100 + arsenic + educ4"
#   "switch ~ dist100 + arsenic + educ4 + dist100:educ4 + arsenic:educ4"
##############################################################################

const WELLS_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Wells.csv"

function load(::Val{:plot_slopes})
    df = CSV.read(Downloads.download(WELLS_URL), DataFrame)
    df.dist100 = df.distance ./ 100
    df.educ4   = df.education ./ 4
    df.switch  = Int.(df.switch)
    return df
end

function examples(::Val{:plot_slopes})
    data = load(Val(:plot_slopes))
    return [
        ("switch ~ dist100 + arsenic + educ4",                                        data),
        ("switch ~ dist100 + arsenic + educ4 + dist100:educ4 + arsenic:educ4",        data),
    ]
end

##############################################################################
# Bambi example: Using Other Samplers — JAX-based samplers
# Source: https://bambinos.github.io/bambi/notebooks/alternative_samplers.html
#
# Dataset: Synthetically generated linear regression data (100 obs; columns: x, y).
#   Demonstrates fitting the same model with blackjax, numpyro, and nutpie backends.
#
# Bambi model formula:
#   "y ~ x"
##############################################################################

function load(::Val{:alternative_samplers})
    rng = MersenneTwister(0)
    x = randn(rng, 100)
    y = 2.0 .+ 1.5 .* x .+ randn(rng, 100)
    return DataFrame(; x, y)
end

function examples(::Val{:alternative_samplers})
    data = load(Val(:alternative_samplers))
    return [
        ("y ~ x", data),
    ]
end
