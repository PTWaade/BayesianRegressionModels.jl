using CSV, DataFrames, Distributions, Downloads, Random, Statistics

const ESCS_URL = "https://ndownloader.figshare.com/files/28870722"

"""
name: escs — Eugene-Springfield Community Sample
source: https://ndownloader.figshare.com/files/28870722
----

604 adults, longitudinal self-report + behavioral measures over 15 years. Outcome: `drugs`
(index of self-reported illegal drug use, ~1 to 4.3). Predictors: Big Five personality
sum-scores — `o` (Openness), `c` (Conscientiousness), `e` (Extraversion),
`a` (Agreeableness), `n` (Neuroticism).
"""
load(::Val{:escs}) = CSV.read(Downloads.download(ESCS_URL), DataFrame)

"""
name: Multiple Linear Regression
source: https://bambinos.github.io/bambi/notebooks/ESCS_multiple_regression.html
example: escs
dataset: escs
formula: "drugs ~ o + c + e + a + n"
verified: true
----

Linear regression of illegal drug use on Big Five personality scores. Family: gaussian.
"""
function examples(::Val{:escs})
    data = load(Val(:escs))
    return ("drugs ~ o + c + e + a + n", data)
end

const CHERRY_BLOSSOMS_URL = "https://ndownloader.figshare.com/files/31072807"

"""
name: cherry_blossoms — Cherry Blossom Bloom Data
source: https://ndownloader.figshare.com/files/31072807
----

Day of year of first cherry blossom bloom in Japan, years 801–2015. 827 observations
after removing rows with missing `doy`. Pre-processing: drop rows with missing `doy`;
cast `year` to Float. `iknots` = 15 quantile-based internal knots from `year`.
"""
load(::Val{:cherry_blossoms}) = CSV.read(Downloads.download(CHERRY_BLOSSOMS_URL), DataFrame)

"""
name: Regression Splines — Explicit Intercept
source: https://bambinos.github.io/bambi/notebooks/splines_cherry_blossoms.html
example: cherry_blossoms
dataset: cherry_blossoms
formula: "doy ~ bs(year, knots=iknots, intercept=True)"
----

Spline basis regression of bloom day-of-year on year, with explicit intercept term in
the basis expansion. Family: gaussian.
"""
function examples(::Val{:cherry_blossoms_explicit})
    data = dropmissing(load(Val(:cherry_blossoms)), :doy)
    data.year = Float64.(data.year)
    return ("doy ~ bs(year, knots=iknots, intercept=True)", data)
end

"""
name: Regression Splines — Absorbed Intercept
source: https://bambinos.github.io/bambi/notebooks/splines_cherry_blossoms.html
example: cherry_blossoms
dataset: cherry_blossoms
formula: "doy ~ bs(year, knots=iknots)"
----

Spline basis regression of bloom day-of-year on year, with model intercept absorbed
(not explicit in the basis). Family: gaussian.
"""
function examples(::Val{:cherry_blossoms_absorbed})
    data = dropmissing(load(Val(:cherry_blossoms)), :doy)
    data.year = Float64.(data.year)
    return ("doy ~ bs(year, knots=iknots)", data)
end

const DIETOX_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv"

"""
name: dietox — Pig Growth Longitudinal Data
source: https://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv
----

From the R package geepack — longitudinal pig weight measurements; 861 observations
across 72 pigs. Pre-processing: cast `Time` to Float.
"""
load(::Val{:dietox}) = CSV.read(Downloads.download(DIETOX_URL), DataFrame)

"""
name: Hierarchical Linear Regression — Pig Growth
source: https://bambinos.github.io/bambi/notebooks/multi-level_regression.html
example: dietox
dataset: dietox
formula: "Weight ~ Time + (Time|Pig)"
----

Mixed-effects model of pig weight over time with random slope and intercept per pig.
Family: gaussian.
"""
function examples(::Val{:dietox})
    data = load(Val(:dietox))
    data.Time = Float64.(data.Time)
    return ("Weight ~ Time + (Time|Pig)", data)
end

const SLEEPSTUDY_URL = "https://ndownloader.figshare.com/files/31181002"

"""
name: sleepstudy — Sleep Deprivation Reaction Times
source: https://ndownloader.figshare.com/files/31181002
----

From the R package lme4 (Belenky et al. 2003) — 180 observations: average reaction
time (ms) over 10 days of sleep deprivation across 18 subjects.
"""
load(::Val{:sleepstudy}) = CSV.read(Downloads.download(SLEEPSTUDY_URL), DataFrame)

"""
name: Hierarchical Linear Regression — Sleep Deprivation
source: https://bambinos.github.io/bambi/notebooks/sleepstudy.html
example: sleepstudy
dataset: sleepstudy
formula: "Reaction ~ 1 + Days + (Days | Subject)"
----

Mixed-effects model of reaction time on days of sleep deprivation with random slope and
intercept per subject. Family: gaussian.
"""
function examples(::Val{:sleepstudy})
    data = load(Val(:sleepstudy))
    return ("Reaction ~ 1 + Days + (Days | Subject)", data)
end

const SRRS2_URL = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/srrs2.dat"
const CTY_URL   = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/cty.dat"

"""
name: radon — SRRS2 Minnesota Radon Measurements
source: https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/srrs2.dat
----

SRRS2 radon measurements (Minnesota) merged with county uranium data; both from the
PyMC examples repository. Derived columns: `log_radon = log(activity + 0.1)`,
`log_u = log(Uppm)`. Deduplicated by `idnum`.
"""
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

"""
name: Radon — Complete Pooling
source: https://bambinos.github.io/bambi/notebooks/radon_example.html
example: radon
dataset: radon
formula: "log_radon ~ 0 + floor"
----

Complete pooling model: single intercept per floor type, ignoring county structure.
Family: gaussian.
"""
function examples(::Val{:radon_pooled})
    data = load(Val(:radon))
    return ("log_radon ~ 0 + floor", data)
end

"""
name: Radon — No Pooling
source: https://bambinos.github.io/bambi/notebooks/radon_example.html
example: radon
dataset: radon
formula: "log_radon ~ 0 + county:floor"
----

No-pooling model: separate intercept for each county-floor combination.
Family: gaussian.
"""
function examples(::Val{:radon_nopooling})
    data = load(Val(:radon))
    return ("log_radon ~ 0 + county:floor", data)
end

"""
name: Radon — Partial Pooling (Intercept Only)
source: https://bambinos.github.io/bambi/notebooks/radon_example.html
example: radon
dataset: radon
formula: "log_radon ~ 1 + (1|county)"
----

Partial pooling with random county intercepts, no floor predictor. Family: gaussian.
"""
function examples(::Val{:radon_partial})
    data = load(Val(:radon))
    return ("log_radon ~ 1 + (1|county)", data)
end

"""
name: Radon — Partial Pooling with Floor
source: https://bambinos.github.io/bambi/notebooks/radon_example.html
example: radon
dataset: radon
formula: "log_radon ~ 1 + floor + (1|county)"
----

Partial pooling with floor as fixed effect and random county intercepts. Family: gaussian.
"""
function examples(::Val{:radon_floor})
    data = load(Val(:radon))
    return ("log_radon ~ 1 + floor + (1|county)", data)
end

"""
name: Radon — Varying Slopes
source: https://bambinos.github.io/bambi/notebooks/radon_example.html
example: radon
dataset: radon
formula: "log_radon ~ floor + (floor|county)"
----

Partial pooling with random slope and intercept for floor by county. Family: gaussian.
"""
function examples(::Val{:radon_slopes})
    data = load(Val(:radon))
    return ("log_radon ~ floor + (floor|county)", data)
end

"""
name: Radon — County-Level Uranium Predictor
source: https://bambinos.github.io/bambi/notebooks/radon_example.html
example: radon
dataset: radon
formula: "log_radon ~ floor + log_u + (1|county)"
----

Partial pooling with floor and county-level uranium predictor (log_u). Family: gaussian.
"""
function examples(::Val{:radon_county_pred})
    data = load(Val(:radon))
    return ("log_radon ~ floor + log_u + (1|county)", data)
end

"""
name: strack_rrr — Strack RRR Facial Feedback Data
source: https://github.com/bambinos/bambi
----

17 CSV files from the Registered Replication Report of Strack, Martin & Stepper (1988)
facial feedback hypothesis. Data directory: `docs/notebooks/data/facial_feedback/` in
the Bambi repo (https://github.com/bambinos/bambi). Files need preprocessing: skip first
two rows, select 22 columns, assign standardised column names, concatenate.
"""
function load(::Val{:strack_rrr})
    error("""
    The Strack RRR dataset consists of 17 CSV files.
    Clone the Bambi repo and load from docs/notebooks/data/facial_feedback/:
      https://github.com/bambinos/bambi
    """)
end

"""
name: Strack RRR — Simple Random Effects
source: https://bambinos.github.io/bambi/notebooks/Strack_RRR_re_analysis.html
example: strack_rrr
dataset: strack_rrr
formula: "value ~ condition + (1|uid)"
----

Simple model with condition as fixed effect and random participant intercepts.
Family: gaussian.
"""
function examples(::Val{:strack_rrr_simple})
    data = load(Val(:strack_rrr))
    return ("value ~ condition + (1|uid)", data)
end

"""
name: Strack RRR — Full Crossed Random Effects
source: https://bambinos.github.io/bambi/notebooks/Strack_RRR_re_analysis.html
example: strack_rrr
dataset: strack_rrr
formula: "value ~ condition + age + gender + (1|uid) + (condition|study) + (condition|stimulus)"
----

Full model with condition, age, gender, random participant intercepts, and crossed random
effects for study and stimulus. Family: gaussian.
"""
function examples(::Val{:strack_rrr_full})
    data = load(Val(:strack_rrr))
    return ("value ~ condition + age + gender + (1|uid) + (condition|study) + (condition|stimulus)", data)
end

const SHOOTER_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/shooter.csv"

"""
name: shooter — Police Officer's Dilemma (Shooter Task)
source: https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/shooter.csv
----

3,600 responses from 36 participants (100 trials each); distributed with the Bambi notebook
examples. Derived: `rate = 1000 / time`; `shoot_or_not` recoded from response × object type.
`S()` denotes sum-to-zero (effects) coding.
"""
function load(::Val{:shooter})
    data = CSV.read(Downloads.download(SHOOTER_URL), DataFrame)
    data.rate = 1000.0 ./ data.time
    return data
end

"""
name: Shooter Task — Rate (No Target Effect)
source: https://bambinos.github.io/bambi/notebooks/shooter_crossed_random_ANOVA.html
example: shooter
dataset: shooter
formula: "rate ~ S(race) * S(object) + (S(race) * S(object) | subject)"
----

Reaction rate modeled by race × object interaction with crossed random effects per subject,
without target random effects. Family: gaussian.
"""
function examples(::Val{:shooter_rate_simple})
    data = load(Val(:shooter))
    return ("rate ~ S(race) * S(object) + (S(race) * S(object) | subject)", data)
end

"""
name: Shooter Task — Rate (With Target Effect)
source: https://bambinos.github.io/bambi/notebooks/shooter_crossed_random_ANOVA.html
example: shooter
dataset: shooter
formula: "rate ~ S(race) * S(object) + (S(race) * S(object) | subject) + (S(object) | target)"
----

Reaction rate modeled by race × object interaction with crossed random effects for both
subject and target. Family: gaussian.
"""
function examples(::Val{:shooter_rate_target})
    data = load(Val(:shooter))
    return ("rate ~ S(race) * S(object) + (S(race) * S(object) | subject) + (S(object) | target)", data)
end

"""
name: Shooter Task — Binary Shoot Decision
source: https://bambinos.github.io/bambi/notebooks/shooter_crossed_random_ANOVA.html
example: shooter
dataset: shooter
formula: "shoot_or_not[shoot] ~ S(race)*S(object) + (S(race)*S(object) | subject) + (S(object) | target)"
----

Binary shoot/don't-shoot decision modeled by race × object interaction with crossed random
effects for subject and target. Family: bernoulli.
"""
function examples(::Val{:shooter_binary})
    data = load(Val(:shooter))
    return ("shoot_or_not[shoot] ~ S(race)*S(object) + (S(race)*S(object) | subject) + (S(object) | target)", data)
end

"""
name: fixed_random — Synthetic Causal DAG Data
source: synthetic
----

Synthetically generated from a causal DAG (30 groups, 2000 obs). Columns: `x`
(continuous), `y` (binary), `z` (group-level), `group`, `xbar` (group mean of `x`).
All models use `family="bernoulli"`.
"""
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

"""
name: Fixed/Random Effects — Naive Model
source: https://bambinos.github.io/bambi/notebooks/fixed_random.html
example: fixed_random
dataset: fixed_random
formula: "y ~ x + z"
----

Naive logistic regression ignoring group structure entirely. Family: bernoulli.
"""
function examples(::Val{:fixed_random_naive})
    data = load(Val(:fixed_random))
    return ("y ~ x + z", data)
end

"""
name: Fixed/Random Effects — Fixed Effects Model
source: https://bambinos.github.io/bambi/notebooks/fixed_random.html
example: fixed_random
dataset: fixed_random
formula: "y ~ 0 + C(group) + x + z"
----

Fixed effects logistic regression with a dummy indicator per group. Family: bernoulli.
"""
function examples(::Val{:fixed_random_fe})
    data = load(Val(:fixed_random))
    return ("y ~ 0 + C(group) + x + z", data)
end

"""
name: Fixed/Random Effects — Random Effects Model
source: https://bambinos.github.io/bambi/notebooks/fixed_random.html
example: fixed_random
dataset: fixed_random
formula: "y ~ x + z + (1|group)"
----

Multilevel logistic regression with random group intercepts. Family: bernoulli.
"""
function examples(::Val{:fixed_random_re})
    data = load(Val(:fixed_random))
    return ("y ~ x + z + (1|group)", data)
end

"""
name: Fixed/Random Effects — Mundlak Machine
source: https://bambinos.github.io/bambi/notebooks/fixed_random.html
example: fixed_random
dataset: fixed_random
formula: "y ~ x + z + xbar + (1|group)"
----

Mundlak machine: random intercepts plus group-mean of x to separate within/between
effects. Family: bernoulli.
"""
function examples(::Val{:fixed_random_mundlak})
    data = load(Val(:fixed_random))
    return ("y ~ x + z + xbar + (1|group)", data)
end

"""
name: t_regression — Synthetic Outlier Data
source: synthetic
----

Synthetically generated — 100 linear observations plus 3 outliers. Columns: `x`, `y`.
"""
function load(::Val{:t_regression})
    rng = MersenneTwister(42)
    x = randn(rng, 100)
    y = 1.0 .+ 2.0 .* x .+ 0.5 .* randn(rng, 100)
    x = vcat(x, [-1.5, 0.0, 1.5])
    y = vcat(y, [20.0, 20.0, 20.0])
    return DataFrame(; x, y)
end

"""
name: Robust Regression — Gaussian Family
source: https://bambinos.github.io/bambi/notebooks/t_regression.html
example: t_regression
dataset: t_regression
formula: "y ~ x"
----

Standard linear regression on data with outliers; used as baseline. Family: gaussian.
"""
function examples(::Val{:t_regression_gaussian})
    data = load(Val(:t_regression))
    return ("y ~ x", data)
end

"""
name: Robust Regression — Student's t Family
source: https://bambinos.github.io/bambi/notebooks/t_regression.html
example: t_regression
dataset: t_regression
formula: "y ~ x"
----

Robust linear regression using Student's t family, which down-weights outliers.
Family: t.
"""
function examples(::Val{:t_regression_t})
    data = load(Val(:t_regression))
    return ("y ~ x", data)
end

const OSIC_URL = "https://gist.githubusercontent.com/ucals/2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/43034c39052dcf97d4b894d2ec1bc3f90f3623d9/osic_pulmonary_fibrosis.csv"

"""
name: predict_new_groups — OSIC Pulmonary Fibrosis Progression
source: https://gist.githubusercontent.com/ucals/2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/43034c39052dcf97d4b894d2ec1bc3f90f3623d9/osic_pulmonary_fibrosis.csv
----

OSIC Pulmonary Fibrosis Progression (Kaggle competition subset). Columns: `patient`
(int-encoded), `weeks` (scaled [0,1]), `fvc` (scaled [0,1]), `smoking_status`.
"""
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

"""
name: Predict New Groups
source: https://bambinos.github.io/bambi/notebooks/predict_new_groups.html
example: predict_new_groups
dataset: predict_new_groups
formula: "fvc ~ 0 + weeks + smoking_status + (0 + weeks | patient)"
----

Mixed-effects model predicting FVC with random slopes per patient; demonstrates
out-of-sample prediction for new groups. Family: gaussian.
"""
function examples(::Val{:predict_new_groups})
    data = load(Val(:predict_new_groups))
    return ("fvc ~ 0 + weeks + smoking_status + (0 + weeks | patient)", data)
end

"""
name: poly_falling — Falling Ball Simulation Data
source: synthetic
----

Synthetically generated: `t` ∈ [0, 2], `x ≈ 5 - 0.5·g·t²` (100 obs, g=9.8).
Includes pre-computed `tsquared` column.
"""
function load(::Val{:poly_falling})
    rng = MersenneTwister(0)
    t = collect(range(0, 2, length=100))
    g = 9.8
    x = 5.0 .- 0.5 .* g .* t .^ 2 .+ 0.5 .* randn(rng, 100)
    return DataFrame(; t, x, tsquared=t .^ 2)
end

"""
name: poly_projectile — Projectile Motion Simulation Data
source: synthetic
----

Synthetically generated projectile motion: `t` ∈ [0, 2], `x ≈ 10·t - 0.5·g·t²`
(100 obs, g=9.8), filtered to `x ≥ 0`.
"""
function load(::Val{:poly_projectile})
    rng = MersenneTwister(0)
    g = 9.8
    tp = collect(range(0, 2, length=100))
    xp = 10 .* tp .- 0.5 .* g .* tp .^ 2 .+ 0.3 .* randn(rng, 100)
    return filter(r -> r.x >= 0, DataFrame(t=tp, x=xp))
end

"""
name: poly_planets — Multi-Planet Projectile Data
source: synthetic
----

Synthetic projectile motion on Earth (g=9.8), Mars (g=3.7), and PlanetX (g=5.0);
50 observations per planet. Columns: `Planet`, `Time`, `Height`.
"""
function load(::Val{:poly_planets})
    rng = MersenneTwister(0)
    # advance rng past the falling/projectile draws used in poly_falling and poly_projectile
    rand(rng, 100)  # falling ball noise
    rand(rng, 100)  # projectile noise
    function throw_data(rng, g_planet, planet)
        tt = collect(range(0, 2, length=50))
        h  = 10 .* tt .- 0.5 .* g_planet .* tt .^ 2 .+ 0.3 .* randn(rng, 50)
        DataFrame(Planet=fill(planet, 50), Time=tt, Height=h)
    end
    return vcat(throw_data(rng, 9.8, "Earth"),
                throw_data(rng, 3.7, "Mars"),
                throw_data(rng, 5.0, "PlanetX"))
end

"""
name: Polynomial Regression — Falling Ball (Explicit)
source: https://bambinos.github.io/bambi/notebooks/polynomial_regression.html
example: polynomial_regression
dataset: poly_falling
formula: "x ~ I(t**2) + 1"
----

Polynomial regression using inline transformation `I(t**2)` to recover gravitational
constant from falling ball data. Family: gaussian.
"""
function examples(::Val{:poly_falling_explicit})
    data = load(Val(:poly_falling))
    return ("x ~ I(t**2) + 1", data)
end

"""
name: Polynomial Regression — Falling Ball (Curly-Brace Syntax)
source: https://bambinos.github.io/bambi/notebooks/polynomial_regression.html
example: polynomial_regression
dataset: poly_falling
formula: "x ~ {t**2} + 1"
----

Same as :poly_falling_explicit but using Bambi's alternative curly-brace inline
transformation syntax. Family: gaussian.
"""
function examples(::Val{:poly_falling_alt})
    data = load(Val(:poly_falling))
    return ("x ~ {t**2} + 1", data)
end

"""
name: Polynomial Regression — Falling Ball (Pre-Computed Column)
source: https://bambinos.github.io/bambi/notebooks/polynomial_regression.html
example: polynomial_regression
dataset: poly_falling
formula: "x ~ tsquared + 1"
----

Polynomial regression using a pre-computed `tsquared` column rather than inline
transformation. Family: gaussian.
"""
function examples(::Val{:poly_falling_precomp})
    data = load(Val(:poly_falling))
    return ("x ~ tsquared + 1", data)
end

"""
name: Polynomial Regression — Projectile (Power Basis)
source: https://bambinos.github.io/bambi/notebooks/polynomial_regression.html
example: polynomial_regression
dataset: poly_projectile
formula: "x ~ I(t**2) + t + 1"
----

Explicit quadratic polynomial regression on projectile data with both linear and
quadratic terms. Family: gaussian.
"""
function examples(::Val{:poly_projectile_power})
    data = load(Val(:poly_projectile))
    return ("x ~ I(t**2) + t + 1", data)
end

"""
name: Polynomial Regression — Projectile (poly() Syntax)
source: https://bambinos.github.io/bambi/notebooks/polynomial_regression.html
example: polynomial_regression
dataset: poly_projectile
formula: "x ~ poly(t, 2, raw=True)"
----

Quadratic polynomial regression using Bambi's `poly()` function with raw (non-orthogonal)
polynomials. Family: gaussian.
"""
function examples(::Val{:poly_projectile_poly})
    data = load(Val(:poly_projectile))
    return ("x ~ poly(t, 2, raw=True)", data)
end

"""
name: Polynomial Regression — Multi-Planet
source: https://bambinos.github.io/bambi/notebooks/polynomial_regression.html
example: polynomial_regression
dataset: poly_planets
formula: "Height ~ I(Time**2):Planet + Time + 0"
----

Planet-specific quadratic coefficient interacted with `Planet` to recover different
gravitational constants per planet. Family: gaussian.
"""
function examples(::Val{:poly_planets})
    data = load(Val(:poly_planets))
    return ("Height ~ I(Time**2):Planet + Time + 0", data)
end

const ANES_URL = "https://ndownloader.figshare.com/files/28870740"

"""
name: logistic_anes — ANES 2016 Vote Intention Data
source: https://ndownloader.figshare.com/files/28870740
----

American National Election Studies 2016 pilot (ANES) filtered to Clinton vs. Trump
voters (373 obs); loaded via `bmb.load_data("ANES")`. Family: bernoulli.
"""
function load(::Val{:logistic_anes})
    df = CSV.read(Downloads.download(ANES_URL), DataFrame)
    return filter(r -> r.vote in ("clinton", "trump"), df)
end

"""
name: Logistic Regression — Vote Intention
source: https://bambinos.github.io/bambi/notebooks/logistic_regression.html
example: logistic_anes
dataset: logistic_anes
formula: "vote['clinton'] ~ party_id + party_id:age"
----

Logistic regression of Clinton vote on party ID and party-age interaction.
Family: bernoulli.
"""
function examples(::Val{:logistic_anes})
    data = load(Val(:logistic_anes))
    return ("vote['clinton'] ~ party_id + party_id:age", data)
end

const ADULTS_URL = "https://ndownloader.figshare.com/files/28870743"

"""
name: model_comparison — 1994 US Census Adults
source: https://ndownloader.figshare.com/files/28870743
----

1994 US Census adults loaded via `bmb.load_data("adults")`; filtered to Black/White
respondents; `age` cast to Float64. Outcome: `income > \$50k`. Family: bernoulli.
"""
function load(::Val{:model_comparison})
    df = CSV.read(Downloads.download(ADULTS_URL), DataFrame)
    df = filter(r -> r.race in ("White", "Black"), df)
    df.age = Float64.(df.age)
    return df
end

"""
name: Model Comparison — Linear
source: https://bambinos.github.io/bambi/notebooks/model_comparison.html
example: model_comparison
dataset: model_comparison
formula: "income['>50K'] ~ sex + race + scale(age) + scale(hs_week)"
----

Linear logistic regression of income on sex, race, age, and weekly hours worked.
Family: bernoulli.
"""
function examples(::Val{:model_comparison_linear})
    data = load(Val(:model_comparison))
    return ("income['>50K'] ~ sex + race + scale(age) + scale(hs_week)", data)
end

"""
name: Model Comparison — Quadratic
source: https://bambinos.github.io/bambi/notebooks/model_comparison.html
example: model_comparison
dataset: model_comparison
formula: "income['>50K'] ~ sex + race + scale(age) + I(scale(age)**2) + scale(hs_week) + I(scale(hs_week)**2)"
----

Quadratic logistic regression adding squared age and hours terms. Family: bernoulli.
"""
function examples(::Val{:model_comparison_quad})
    data = load(Val(:model_comparison))
    return ("income['>50K'] ~ sex + race + scale(age) + I(scale(age)**2) + scale(hs_week) + I(scale(hs_week)**2)", data)
end

"""
name: Model Comparison — Cubic
source: https://bambinos.github.io/bambi/notebooks/model_comparison.html
example: model_comparison
dataset: model_comparison
formula: "income['>50K'] ~ sex + race + scale(age) + I(scale(age)**2) + I(scale(age)**3) + scale(hs_week) + I(scale(hs_week)**2) + I(scale(hs_week)**3)"
----

Cubic logistic regression adding cubic age and hours terms. Family: bernoulli.
"""
function examples(::Val{:model_comparison_cubic})
    data = load(Val(:model_comparison))
    return ("income['>50K'] ~ sex + race + scale(age) + I(scale(age)**2) + I(scale(age)**3) + scale(hs_week) + I(scale(hs_week)**2) + I(scale(hs_week)**3)", data)
end

const BATTING_URL = "https://ndownloader.figshare.com/files/29749140"

"""
name: hierarchical_binomial — Baseball Batting Statistics
source: https://ndownloader.figshare.com/files/29749140
----

Baseball batting statistics loaded via `bmb.load_data("batting")`; filtered to 2016+,
first 15 obs; `batting_avg = H / AB`. Family: binomial.
"""
function load(::Val{:hierarchical_binomial})
    df = CSV.read(Downloads.download(BATTING_URL), DataFrame)
    df = dropmissing(df)
    df = filter(r -> r.AB > 0, df)
    df = filter(r -> r.yearID >= 2016, df)
    df.batting_avg = df.H ./ df.AB
    return first(df, 15)
end

"""
name: Hierarchical Binomial — No Pooling
source: https://bambinos.github.io/bambi/notebooks/hierarchical_binomial_bambi.html
example: hierarchical_binomial
dataset: hierarchical_binomial
formula: "p(H, AB) ~ 0 + playerID"
----

Non-hierarchical binomial model: separate intercept per player. Family: binomial.
"""
function examples(::Val{:hierarchical_binomial_nopooling})
    data = load(Val(:hierarchical_binomial))
    return ("p(H, AB) ~ 0 + playerID", data)
end

"""
name: Hierarchical Binomial — Partial Pooling
source: https://bambinos.github.io/bambi/notebooks/hierarchical_binomial_bambi.html
example: hierarchical_binomial
dataset: hierarchical_binomial
formula: "p(H, AB) ~ 1 + (1|playerID)"
----

Hierarchical binomial model with random player intercepts. Family: binomial.
"""
function examples(::Val{:hierarchical_binomial_partial})
    data = load(Val(:hierarchical_binomial))
    return ("p(H, AB) ~ 1 + (1|playerID)", data)
end

"""
name: alternative_links — Bliss Beetle Mortality Data
source: synthetic
----

Bliss (1935) beetle mortality data — 8 aggregated observations. Columns: `x`
(log dose), `n` (beetles exposed), `y` (beetles killed). Family: binomial.
"""
function load(::Val{:alternative_links})
    return DataFrame(
        x = [1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839],
        n = [59, 60, 62, 56, 63, 59, 62, 60],
        y = [ 6,  13,  18,  28,  52,  53,  61,  60],
    )
end

"""
name: Alternative Links — Logit
source: https://bambinos.github.io/bambi/notebooks/alternative_links_binary.html
example: alternative_links
dataset: alternative_links
formula: "p(y, n) ~ x"
----

Binomial regression with logit link (default). Family: binomial, link="logit".
"""
function examples(::Val{:alternative_links_logit})
    data = load(Val(:alternative_links))
    return ("p(y, n) ~ x", data)
end

"""
name: Alternative Links — Probit
source: https://bambinos.github.io/bambi/notebooks/alternative_links_binary.html
example: alternative_links
dataset: alternative_links
formula: "p(y, n) ~ x"
----

Binomial regression with probit link. Family: binomial, link="probit".
"""
function examples(::Val{:alternative_links_probit})
    data = load(Val(:alternative_links))
    return ("p(y, n) ~ x", data)
end

"""
name: Alternative Links — Complementary Log-Log
source: https://bambinos.github.io/bambi/notebooks/alternative_links_binary.html
example: alternative_links
dataset: alternative_links
formula: "p(y, n) ~ x"
----

Binomial regression with complementary log-log link. Family: binomial, link="cloglog".
"""
function examples(::Val{:alternative_links_cloglog})
    data = load(Val(:alternative_links))
    return ("p(y, n) ~ x", data)
end

const CARCLAIMS_URL = "https://ndownloader.figshare.com/files/28870713"

"""
name: wald_gamma — Australian Car Insurance Claims
source: https://ndownloader.figshare.com/files/28870713
----

Australian car insurance claims 2004–2005 loaded via `bmb.load_data("carclaims")`;
filtered to `claimcst0 > 0` (records with a claim).
"""
function load(::Val{:wald_gamma})
    df = CSV.read(Downloads.download(CARCLAIMS_URL), DataFrame)
    return filter(r -> r.claimcst0 > 0, df)
end

"""
name: Wald Regression — Insurance Claims (Wald Family)
source: https://bambinos.github.io/bambi/notebooks/wald_gamma_glm.html
example: wald_gamma
dataset: wald_gamma
formula: "claimcst0 ~ C(agecat) + gender + area"
----

Wald (inverse Gaussian) regression of claim costs on age category (as categorical),
gender and area. Family: wald, link="log".
"""
function examples(::Val{:wald_gamma_wald})
    data = load(Val(:wald_gamma))
    return ("claimcst0 ~ C(agecat) + gender + area", data)
end

"""
name: Gamma Regression — Insurance Claims (Gamma Family)
source: https://bambinos.github.io/bambi/notebooks/wald_gamma_glm.html
example: wald_gamma
dataset: wald_gamma
formula: "claimcst0 ~ agecat + gender + area"
----

Gamma regression of claim costs on age category (numeric), gender and area.
Family: gamma, link="log".
"""
function examples(::Val{:wald_gamma_gamma})
    data = load(Val(:wald_gamma))
    return ("claimcst0 ~ agecat + gender + area", data)
end

const NB_DATA_URL = "https://stats.idre.ucla.edu/stat/stata/dae/nb_data.dta"

"""
name: negative_binomial — UCLA Student Absences Data
source: https://stats.idre.ucla.edu/stat/stata/dae/nb_data.dta
----

314 high school juniors; `daysabs`, `prog`, `math`. Source: UCLA IDRE. NOTE: Stata `.dta`
format — requires ReadStatTables.jl or similar to load. `prog` recoded: 1→"General",
2→"Academic", 3→"Vocational". Family: negativebinomial.
"""
function load(::Val{:negative_binomial})
    error("""
    nb_data.dta is a Stata file. Load with ReadStatTables.jl:
      using ReadStatTables
      df = DataFrame(readstat("$NB_DATA_URL"))
    """)
end

"""
name: Negative Binomial — Main Effects
source: https://bambinos.github.io/bambi/notebooks/negative_binomial.html
example: negative_binomial
dataset: negative_binomial
formula: "daysabs ~ 0 + prog + scale(math)"
----

Negative binomial regression of student absences on program type and math score.
Family: negativebinomial.
"""
function examples(::Val{:negative_binomial_main})
    data = load(Val(:negative_binomial))
    return ("daysabs ~ 0 + prog + scale(math)", data)
end

"""
name: Negative Binomial — Interaction
source: https://bambinos.github.io/bambi/notebooks/negative_binomial.html
example: negative_binomial
dataset: negative_binomial
formula: "daysabs ~ 0 + prog + scale(math) + prog:scale(math)"
----

Negative binomial regression adding program-by-math interaction term.
Family: negativebinomial.
"""
function examples(::Val{:negative_binomial_interaction})
    data = load(Val(:negative_binomial))
    return ("daysabs ~ 0 + prog + scale(math) + prog:scale(math)", data)
end

const ROACHES_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/roaches.csv"

"""
name: count_roaches — Roach Pest-Control Study
source: https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/roaches.csv
----

Pest-control study (Gelman et al., Regression and Other Stories). `roach1` rescaled
by /100. Columns: `y`, `roach1`, `treatment`, `senior`, `exposure2`.
"""
function load(::Val{:count_roaches})
    df = CSV.read(Downloads.download(ROACHES_URL), DataFrame)
    df.roach1 = df.roach1 ./ 100
    return df
end

"""
name: Count Regression with Offset — Poisson
source: https://bambinos.github.io/bambi/notebooks/count_roaches.html
example: count_roaches
dataset: count_roaches
formula: "y ~ roach1 + treatment + senior + offset(log(exposure2))"
----

Poisson regression of roach counts with log-exposure offset.
Family: poisson.
"""
function examples(::Val{:count_roaches_poisson})
    data = load(Val(:count_roaches))
    return ("y ~ roach1 + treatment + senior + offset(log(exposure2))", data)
end

"""
name: Count Regression with Offset — Negative Binomial
source: https://bambinos.github.io/bambi/notebooks/count_roaches.html
example: count_roaches
dataset: count_roaches
formula: "y ~ roach1 + treatment + senior + offset(log(exposure2))"
----

Negative binomial regression of roach counts with log-exposure offset; accounts for
overdispersion relative to Poisson. Family: negativebinomial.
"""
function examples(::Val{:count_roaches_nb})
    data = load(Val(:count_roaches))
    return ("y ~ roach1 + treatment + senior + offset(log(exposure2))", data)
end

"""
name: beta_regression — Baseball Batting Statistics (Beta)
source: https://ndownloader.figshare.com/files/29749140
----

Baseball batting statistics; filtered to `AB > 100` and `yearID >= 1990`;
`batting_avg = H/AB`; `batting_avg_shift` = previous year's batting average per player.
Family: beta.
"""
load(::Val{:beta_regression}) = CSV.read(Downloads.download(BATTING_URL), DataFrame)

"""
name: beta_probs — Synthetic Beta Probabilities
source: synthetic
----

Synthetically generated: 500 uniform probabilities in [0.1, 0.9]. Column: `probabilities`.
Family: beta.
"""
function load(::Val{:beta_probs})
    rng = MersenneTwister(1)
    return DataFrame(probabilities = rand(rng, 500) .* 0.8 .+ 0.1)
end

"""
name: beta_coin — Synthetic Coin Dirt Data
source: synthetic
----

Synthetically generated: 200 observations of `delta_d` (heads_dirt - tails_dirt) and
`p` (probability clamped to [0.01, 0.99]). Family: beta.
"""
function load(::Val{:beta_coin})
    rng = MersenneTwister(1)
    rand(rng, 500)  # advance past beta_probs draw
    n = 200
    heads_dirt = abs.(randn(rng, n) .* 25)
    tails_dirt = abs.(randn(rng, n) .* 25)
    p_vals = clamp.(0.5 .+ (heads_dirt .- tails_dirt) ./ 200, 0.01, 0.99)
    return DataFrame(delta_d = heads_dirt .- tails_dirt, p = p_vals)
end

"""
name: beta_batting — Baseball Batting Averages (Processed)
source: https://ndownloader.figshare.com/files/29749140
----

Baseball batting statistics filtered to `AB > 100` and `yearID >= 1990`, sorted by
player and year; `batting_avg = H/AB`; `batting_avg_shift` = previous year's batting
average per player (missing for first year of each player). Family: beta.
"""
function load(::Val{:beta_batting})
    batting = load(Val(:beta_regression))
    batting = dropmissing(filter(r -> r.AB > 100 && r.yearID >= 1990, batting))
    batting.batting_avg = batting.H ./ batting.AB
    sort!(batting, [:playerID, :yearID])
    batting.batting_avg_shift = vcat([missing],
        [batting.playerID[i] == batting.playerID[i-1] ?
         batting.batting_avg[i-1] : missing for i in 2:nrow(batting)])
    return dropmissing(batting, :batting_avg_shift)
end

"""
name: Beta Regression — Intercept Only (Probabilities)
source: https://bambinos.github.io/bambi/notebooks/beta_regression.html
example: beta_regression
dataset: beta_probs
formula: "probabilities ~ 1"
----

Intercept-only beta regression on synthetic probability data. Family: beta.
"""
function examples(::Val{:beta_probs_intercept})
    data = load(Val(:beta_probs))
    return ("probabilities ~ 1", data)
end

"""
name: Beta Regression — Coin Dirt
source: https://bambinos.github.io/bambi/notebooks/beta_regression.html
example: beta_regression
dataset: beta_coin
formula: "p ~ delta_d"
----

Beta regression of probability on coin dirt differential. Family: beta.
"""
function examples(::Val{:beta_coin})
    data = load(Val(:beta_coin))
    return ("p ~ delta_d", data)
end

"""
name: Beta Regression — Batting Average Intercept Only
source: https://bambinos.github.io/bambi/notebooks/beta_regression.html
example: beta_regression
dataset: beta_batting
formula: "batting_avg ~ 1"
----

Intercept-only beta regression on baseball batting averages. Family: beta.
"""
function examples(::Val{:beta_batting_intercept})
    data = load(Val(:beta_batting))
    return ("batting_avg ~ 1", data)
end

"""
name: Beta Regression — Batting Average with Shift Predictor
source: https://bambinos.github.io/bambi/notebooks/beta_regression.html
example: beta_regression
dataset: beta_batting
formula: "batting_avg ~ batting_avg_shift"
----

Beta regression of batting average on prior-year batting average. Family: beta.
"""
function examples(::Val{:beta_batting_shift})
    data = load(Val(:beta_batting))
    return ("batting_avg ~ batting_avg_shift", data)
end

const IRIS_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv"

"""
name: categorical_regression — Iris Dataset
source: https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv
----

Classic iris dataset — 150 obs, four morphological predictors (`sepal_length`,
`sepal_width`, `petal_length`, `petal_width`), three species. Family: categorical.
"""
load(::Val{:categorical_regression}) = CSV.read(Downloads.download(IRIS_URL), DataFrame)

"""
name: categorical_toy — Synthetic Three-Class Data
source: synthetic
----

Synthetic data with three Gaussian classes: N(-2.5, 1.2), N(0, 0.5), N(2.5, 1.2);
50 obs per class. Columns: `x`, `y` (class label A/B/C). Family: categorical.
"""
function load(::Val{:categorical_toy})
    rng = MersenneTwister(7)
    return vcat(
        DataFrame(x = randn(rng, 50) .- 2.5, y = fill("A", 50)),
        DataFrame(x = randn(rng, 50) .* 0.5, y = fill("B", 50)),
        DataFrame(x = randn(rng, 50) .+ 2.5, y = fill("C", 50)),
    )
end

"""
name: categorical_alligator — Alligator Food Choice Data
source: synthetic
----

Alligator food choice data (Agresti 2002, Ch. 8) — 64 obs; columns: `choice`
(Fish/Invertebrates/Other), `length`, `sex`. Family: categorical.
"""
function load(::Val{:categorical_alligator})
    return DataFrame(
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
end

"""
name: Categorical Regression — Toy Dataset
source: https://bambinos.github.io/bambi/notebooks/categorical_regression.html
example: categorical_regression
dataset: categorical_toy
formula: "y ~ x"
----

Categorical regression on three Gaussian classes. Family: categorical.
"""
function examples(::Val{:categorical_toy})
    data = load(Val(:categorical_toy))
    return ("y ~ x", data)
end

"""
name: Categorical Regression — Iris Species
source: https://bambinos.github.io/bambi/notebooks/categorical_regression.html
example: categorical_regression
dataset: categorical_regression
formula: "species ~ sepal_length + sepal_width + petal_length + petal_width"
----

Categorical regression predicting iris species from morphological measurements.
Family: categorical.
"""
function examples(::Val{:categorical_iris})
    iris = load(Val(:categorical_regression))
    rename!(iris, "Species" => "species", "Sepal.Length" => "sepal_length",
            "Sepal.Width" => "sepal_width", "Petal.Length" => "petal_length",
            "Petal.Width" => "petal_width")
    return ("species ~ sepal_length + sepal_width + petal_length + petal_width", iris)
end

"""
name: Categorical Regression — Alligator Food Choice
source: https://bambinos.github.io/bambi/notebooks/categorical_regression.html
example: categorical_regression
dataset: categorical_alligator
formula: "choice ~ length + sex"
----

Categorical regression of alligator food choice on body length and sex.
Family: categorical.
"""
function examples(::Val{:categorical_alligator})
    data = load(Val(:categorical_alligator))
    return ("choice ~ length + sex", data)
end

const PERIWINKLES_URL = "https://ndownloader.figshare.com/files/34446077"

"""
name: circular_regression — Periwinkle Directional Data
source: https://ndownloader.figshare.com/files/34446077
----

31 sea snails relocated; `direction` and `distance` columns. Loaded via
`bmb.load_data("periwinkles")`; `distance` cast to Float64.
"""
function load(::Val{:circular_regression})
    df = CSV.read(Downloads.download(PERIWINKLES_URL), DataFrame)
    df.distance = Float64.(df.distance)
    return df
end

"""
name: Circular Regression — Von Mises Family
source: https://bambinos.github.io/bambi/notebooks/circular_regression.html
example: circular_regression
dataset: circular_regression
formula: "direction ~ distance"
----

Circular regression of periwinkle direction on distance using von Mises distribution.
Family: vonmises.
"""
function examples(::Val{:circular_vonmises})
    data = load(Val(:circular_regression))
    return ("direction ~ distance", data)
end

"""
name: Circular Regression — Gaussian Comparison
source: https://bambinos.github.io/bambi/notebooks/circular_regression.html
example: circular_regression
dataset: circular_regression
formula: "direction ~ distance"
----

Gaussian regression of circular direction data; used as a misspecified comparison model
against the von Mises. Family: gaussian.
"""
function examples(::Val{:circular_gaussian})
    data = load(Val(:circular_regression))
    return ("direction ~ distance", data)
end

const BMI_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/bmi.csv"

"""
name: quantile_regression — BMI of Dutch Children
source: https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/bmi.csv
----

BMI measurements of Dutch children and young adults; distributed with the Bambi notebook
examples. `knots = quantile(age, LinRange(0,1,10))[2:end-1]` (8 interior knots).
"""
load(::Val{:quantile_regression}) = CSV.read(Downloads.download(BMI_URL), DataFrame)

"""
name: Quantile Regression — p10 (Asymmetric Laplace)
source: https://bambinos.github.io/bambi/notebooks/quantile_regression.html
example: quantile_regression
dataset: quantile_regression
formula: "bmi ~ bs(age, knots=knots)"
----

Spline regression targeting the 10th percentile of BMI. Family: asymmetriclaplace, kappa=0.1.
"""
function examples(::Val{:quantile_p10})
    data = load(Val(:quantile_regression))
    return ("bmi ~ bs(age, knots=knots)", data)
end

"""
name: Quantile Regression — p50 (Asymmetric Laplace)
source: https://bambinos.github.io/bambi/notebooks/quantile_regression.html
example: quantile_regression
dataset: quantile_regression
formula: "bmi ~ bs(age, knots=knots)"
----

Spline regression targeting the 50th percentile (median) of BMI.
Family: asymmetriclaplace, kappa=0.5.
"""
function examples(::Val{:quantile_p50})
    data = load(Val(:quantile_regression))
    return ("bmi ~ bs(age, knots=knots)", data)
end

"""
name: Quantile Regression — p90 (Asymmetric Laplace)
source: https://bambinos.github.io/bambi/notebooks/quantile_regression.html
example: quantile_regression
dataset: quantile_regression
formula: "bmi ~ bs(age, knots=knots)"
----

Spline regression targeting the 90th percentile of BMI. Family: asymmetriclaplace, kappa=0.9.
"""
function examples(::Val{:quantile_p90})
    data = load(Val(:quantile_regression))
    return ("bmi ~ bs(age, knots=knots)", data)
end

"""
name: Quantile Regression — Gaussian Comparison
source: https://bambinos.github.io/bambi/notebooks/quantile_regression.html
example: quantile_regression
dataset: quantile_regression
formula: "bmi ~ bs(age, knots=knots)"
----

Spline regression using Gaussian family; used as comparison model to quantile approach.
Family: gaussian.
"""
function examples(::Val{:quantile_gaussian})
    data = load(Val(:quantile_regression))
    return ("bmi ~ bs(age, knots=knots)", data)
end

"""
name: mister_p — CCES 2018 Survey Data
source: https://dataverse.harvard.edu/
----

Two datasets: (1) CCES 2018 Common Content (Schaffner, Ansolabehere & Luks 2018),
Harvard Dataverse. Outcome: binary abortion opinion (from CC18_321d). (2) State-level
predictors CSV (`repvote` etc.), distributed with Bambi notebook. See
https://bambinos.github.io/bambi/notebooks/mister_p.html for download instructions.
"""
function load(::Val{:mister_p})
    error("""
    The MrP example uses the CCES 2018 survey dataset from Harvard Dataverse
    (https://dataverse.harvard.edu/). Download and preprocess per the notebook:
      https://bambinos.github.io/bambi/notebooks/mister_p.html
    """)
end

"""
name: MrP — Multilevel Regression and Post-Stratification
source: https://bambinos.github.io/bambi/notebooks/mister_p.html
example: mister_p
dataset: mister_p
formula: "p(abortion, n) ~ male + repvote + (1|state) + (1|eth) + (1|edu) + (1|male:eth) + (1|edu:age) + (1|edu:eth)"
----

Multilevel binomial regression with crossed random effects for state, ethnicity,
education and interactions; used for post-stratification. Family: binomial.
"""
function examples(::Val{:mister_p})
    data = load(Val(:mister_p))
    return ("p(abortion, n) ~ male + repvote + (1|state) + (1|eth) + (1|edu) + (1|male:eth) + (1|edu:age) + (1|edu:eth)", data)
end

const FISH_URL = "https://stats.idre.ucla.edu/stat/data/fish.csv"

"""
name: zero_inflated — Fish Catch Survey
source: https://stats.idre.ucla.edu/stat/data/fish.csv
----

250 groups at a state park; `count`, `livebait`, `camper`, `persons`, `child`.
Filtered to `count < 60` (248 obs).
"""
function load(::Val{:zero_inflated})
    df = CSV.read(Downloads.download(FISH_URL), DataFrame)
    return filter(r -> r.count < 60, df)
end

"""
name: Zero Inflated Poisson — Mu Component
source: https://bambinos.github.io/bambi/notebooks/zero_inflated_regression.html
example: zero_inflated
dataset: zero_inflated
formula: "count ~ livebait + camper + persons + child"
----

ZIP mu component: models the expected count given non-zero. Family: zero_inflated_poisson.
"""
function examples(::Val{:zip_mu})
    data = load(Val(:zero_inflated))
    return ("count ~ livebait + camper + persons + child", data)
end

"""
name: Zero Inflated Poisson — Psi Component
source: https://bambinos.github.io/bambi/notebooks/zero_inflated_regression.html
example: zero_inflated
dataset: zero_inflated
formula: "psi ~ livebait + camper + persons + child"
----

ZIP psi component: models the probability of excess zeros. Family: zero_inflated_poisson.
"""
function examples(::Val{:zip_psi})
    data = load(Val(:zero_inflated))
    return ("psi ~ livebait + camper + persons + child", data)
end

"""
name: Hurdle Poisson — Mu Component
source: https://bambinos.github.io/bambi/notebooks/zero_inflated_regression.html
example: zero_inflated
dataset: zero_inflated
formula: "count ~ livebait + camper + persons + child"
----

Hurdle Poisson mu component: models the expected count given positive. Family: hurdle_poisson.
"""
function examples(::Val{:hurdle_mu})
    data = load(Val(:zero_inflated))
    return ("count ~ livebait + camper + persons + child", data)
end

"""
name: Hurdle Poisson — Psi Component
source: https://bambinos.github.io/bambi/notebooks/zero_inflated_regression.html
example: zero_inflated
dataset: zero_inflated
formula: "psi ~ livebait + camper + persons + child"
----

Hurdle Poisson psi component: models the probability of a non-zero outcome.
Family: hurdle_poisson.
"""
function examples(::Val{:hurdle_psi})
    data = load(Val(:zero_inflated))
    return ("psi ~ livebait + camper + persons + child", data)
end

const TROLLEY_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Trolley.csv"
const HR_ATTRITION_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/hr_employee_attrition.tsv.txt"

"""
name: ordinal_regression — Trolley Moral Intuition Data
source: https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Trolley.csv
----

Trolley (McElreath, Statistical Rethinking) — 9930 obs; `response` = 1–7 moral
intuition rating; `action`, `intention`, `contact` as unordered categorical strings.
"""
function load(::Val{:ordinal_regression})
    df = CSV.read(Downloads.download(TROLLEY_URL), DataFrame; delim=";")
    for col in (:action, :intention, :contact)
        df[!, col] = string.(df[!, col])
    end
    return df
end

"""
name: hr_attrition — IBM HR Employee Attrition Data
source: https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/hr_employee_attrition.tsv.txt
----

IBM HR Employee Attrition (Kaggle) — filtered to `Attrition="No"`. Columns include
`YearsAtCompany`, `TotalWorkingYears`. Family: sratio.
"""
function load(::Val{:hr_attrition})
    hr = CSV.read(Downloads.download(HR_ATTRITION_URL), DataFrame; delim="\t")
    return filter(r -> r.Attrition == "No", hr)
end

"""
name: Ordinal Regression — Trolley Intercept Only
source: https://bambinos.github.io/bambi/notebooks/ordinal_regression.html
example: ordinal_regression
dataset: ordinal_regression
formula: "response ~ 0"
----

Cumulative ordinal model with only threshold parameters, no predictors.
Family: cumulative.
"""
function examples(::Val{:ordinal_trolley_intercept})
    data = load(Val(:ordinal_regression))
    return ("response ~ 0", data)
end

"""
name: Ordinal Regression — Trolley Action/Intention/Contact
source: https://bambinos.github.io/bambi/notebooks/ordinal_regression.html
example: ordinal_regression
dataset: ordinal_regression
formula: "response ~ 0 + action + intention + contact + action:intention + contact:intention"
----

Cumulative ordinal model with action, intention, contact and their interactions.
Family: cumulative.
"""
function examples(::Val{:ordinal_trolley_effects})
    data = load(Val(:ordinal_regression))
    return ("response ~ 0 + action + intention + contact + action:intention + contact:intention", data)
end

"""
name: Ordinal Regression — HR Years at Company
source: https://bambinos.github.io/bambi/notebooks/ordinal_regression.html
example: ordinal_regression
dataset: hr_attrition
formula: "YearsAtCompany ~ 0 + TotalWorkingYears"
----

Sequential ratio (sratio) ordinal model predicting years at company from total
working years. Family: sratio.
"""
function examples(::Val{:ordinal_hr_years})
    data = load(Val(:hr_attrition))
    return ("YearsAtCompany ~ 0 + TotalWorkingYears", data)
end

const BIKES_URL = "https://ndownloader.figshare.com/files/38737026"

"""
name: distributional_models — Bikes Hourly Count Data
source: https://ndownloader.figshare.com/files/38737026
----

Hourly bike counts loaded via `bmb.load_data("bikes")`; subsampled every 50th row
(348 obs). Columns include `count`, `hour`.
"""
load(::Val{:distributional_models}) = CSV.read(Downloads.download(BIKES_URL), DataFrame)

"""
name: distributional_synth — Synthetic Gamma Data
source: synthetic
----

Synthetically generated: 200 observations. `x ~ Uniform(-1.5, 1.5)`,
`alpha = exp(0.5 + 1.5·x)`, `mu = exp(1 + 0.5·x)`, `y ~ Gamma(alpha, mu/alpha)`.
Family: gamma.
"""
function load(::Val{:distributional_synth})
    rng = MersenneTwister(121195)
    x = rand(rng, 200) .* 3 .- 1.5
    alpha = exp.(0.5 .+ 1.5 .* x)
    mu = exp.(1.0 .+ 0.5 .* x)
    y = [rand(rng, Gamma(a, m / a)) for (a, m) in zip(alpha, mu)]
    return DataFrame(; x, y)
end

"""
name: Distributional Models — Constant Alpha
source: https://bambinos.github.io/bambi/notebooks/distributional_models.html
example: distributional_models
dataset: distributional_synth
formula: "y ~ x"
----

Gamma regression with a single (constant) shape parameter alpha. Family: gamma.
"""
function examples(::Val{:distributional_const_alpha})
    data = load(Val(:distributional_synth))
    return ("y ~ x", data)
end

"""
name: Distributional Models — Varying Alpha
source: https://bambinos.github.io/bambi/notebooks/distributional_models.html
example: distributional_models
dataset: distributional_synth
formula: "y ~ x  +  alpha ~ x"
----

Distributional gamma regression: compound formula models both mu and alpha as functions
of x. Family: gamma.
"""
function examples(::Val{:distributional_var_alpha})
    data = load(Val(:distributional_synth))
    return ("y ~ x  +  alpha ~ x", data)
end

"""
name: Distributional Models — Bikes Spline
source: https://bambinos.github.io/bambi/notebooks/distributional_models.html
example: distributional_models
dataset: distributional_models
formula: "count ~ 0 + bs(hour, 8, intercept=True)  +  alpha ~ 0 + bs(hour, 8, intercept=True)"
----

Distributional gamma regression on hourly bike count data; both mu and alpha modeled
with spline basis functions. Family: gamma.
"""
function examples(::Val{:distributional_bikes})
    bikes = load(Val(:distributional_models))
    bikes = bikes[1:50:nrow(bikes), :]
    return ("count ~ 0 + bs(hour, 8, intercept=True)  +  alpha ~ 0 + bs(hour, 8, intercept=True)", bikes)
end

const GAM_DATA_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/gam_data.csv"

"""
name: hsgp_1d — Synthetic Smooth-Function Data (1D GP)
source: synthetic
----

100 observations; `x` ∈ [-3, 3], `y = sin(x) + noise`. Used for HSGP demonstrations
in one dimension.
"""
function load(::Val{:hsgp_1d})
    rng = MersenneTwister(0)
    x = collect(range(-3, 3, length=100))
    y = sin.(x) .+ 0.3 .* randn(rng, 100)
    return DataFrame(; x, y)
end

"""
name: gam_data — GAM Simulation Data
source: https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/gam_data.csv
----

300 observations with categorical grouping variable `fac`; simulated with `gamSim()` from
R package mgcv; distributed with Bambi notebook examples.
"""
function load(::Val{:gam_data})
    gam_data = CSV.read(Downloads.download(GAM_DATA_URL), DataFrame)
    gam_data.fac = string.(gam_data.fac)
    return gam_data
end

"""
name: HSGP 1D — Basic
source: https://bambinos.github.io/bambi/notebooks/hsgp_1d.html
example: hsgp_1d
dataset: hsgp_1d
formula: "y ~ 0 + hsgp(x, m=10, c=2)"
----

Hilbert-space Gaussian process approximation on 1D synthetic data. Family: gaussian.
"""
function examples(::Val{:hsgp_1d_basic})
    data = load(Val(:hsgp_1d))
    return ("y ~ 0 + hsgp(x, m=10, c=2)", data)
end

"""
name: HSGP 1D — Centered Parameterization
source: https://bambinos.github.io/bambi/notebooks/hsgp_1d.html
example: hsgp_1d
dataset: hsgp_1d
formula: "y ~ 0 + hsgp(x, m=10, c=2, centered=True)"
----

HSGP with centered parameterization on 1D synthetic data. Family: gaussian.
"""
function examples(::Val{:hsgp_1d_centered})
    data = load(Val(:hsgp_1d))
    return ("y ~ 0 + hsgp(x, m=10, c=2, centered=True)", data)
end

"""
name: HSGP 1D — By Group (Shared Covariance)
source: https://bambinos.github.io/bambi/notebooks/hsgp_1d.html
example: hsgp_1d
dataset: gam_data
formula: "y ~ 0 + hsgp(x2, by=fac, m=12, c=1.5)"
----

HSGP on GAM data with group-specific processes sharing covariance parameters.
Family: gaussian.
"""
function examples(::Val{:hsgp_1d_by_group})
    data = load(Val(:gam_data))
    return ("y ~ 0 + hsgp(x2, by=fac, m=12, c=1.5)", data)
end

"""
name: HSGP 1D — By Group (No Shared Covariance)
source: https://bambinos.github.io/bambi/notebooks/hsgp_1d.html
example: hsgp_1d
dataset: gam_data
formula: "y ~ 0 + hsgp(x2, by=fac, m=12, c=1.5, share_cov=False)"
----

HSGP on GAM data with group-specific processes and independent covariance parameters.
Family: gaussian.
"""
function examples(::Val{:hsgp_1d_nocov})
    data = load(Val(:gam_data))
    return ("y ~ 0 + hsgp(x2, by=fac, m=12, c=1.5, share_cov=False)", data)
end

const POISSON_DATA_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/poisson_data.csv"

"""
name: hsgp_2d — Synthetic 2D GP Data
source: synthetic
----

100 observations; `outcome = sin(x)·cos(y) + noise`; `group` ∈ {A, B}. Used for
HSGP demonstrations in two dimensions.
"""
function load(::Val{:hsgp_2d})
    rng = MersenneTwister(1)
    n = 100
    x = randn(rng, n)
    y = randn(rng, n)
    outcome = sin.(x) .* cos.(y) .+ 0.2 .* randn(rng, n)
    group = rand(rng, ["A", "B"], n)
    return DataFrame(; x, y, outcome, group)
end

"""
name: poisson_data — Spatial Count Data
source: https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/poisson_data.csv
----

Count data with lat/lon coordinates and year category; distributed with Bambi notebook
examples. Columns include `Count`, `Lon`, `Lat`, `Year`, `Site`, `X1`.
"""
function load(::Val{:poisson_data})
    poisson = CSV.read(Downloads.download(POISSON_DATA_URL), DataFrame)
    poisson.Year = string.(poisson.Year)
    return poisson
end

"""
name: HSGP 2D — Isotropic
source: https://bambinos.github.io/bambi/notebooks/hsgp_2d.html
example: hsgp_2d
dataset: hsgp_2d
formula: "outcome ~ 0 + hsgp(x, y, c=1.5, m=10)"
----

Isotropic 2D Hilbert-space Gaussian process on synthetic data. Family: gaussian.
"""
function examples(::Val{:hsgp_2d_iso})
    data = load(Val(:hsgp_2d))
    return ("outcome ~ 0 + hsgp(x, y, c=1.5, m=10)", data)
end

"""
name: HSGP 2D — By Group (Shared Covariance)
source: https://bambinos.github.io/bambi/notebooks/hsgp_2d.html
example: hsgp_2d
dataset: hsgp_2d
formula: "outcome ~ 0 + hsgp(x, y, by=group, c=1.5, m=10)"
----

2D HSGP with group-specific processes sharing covariance parameters. Family: gaussian.
"""
function examples(::Val{:hsgp_2d_by_group})
    data = load(Val(:hsgp_2d))
    return ("outcome ~ 0 + hsgp(x, y, by=group, c=1.5, m=10)", data)
end

"""
name: HSGP 2D — By Group (No Shared Covariance)
source: https://bambinos.github.io/bambi/notebooks/hsgp_2d.html
example: hsgp_2d
dataset: hsgp_2d
formula: "outcome ~ 0 + hsgp(x, y, by=group, c=1.5, m=10, share_cov=False)"
----

2D HSGP with group-specific processes and independent covariance parameters.
Family: gaussian.
"""
function examples(::Val{:hsgp_2d_nocov})
    data = load(Val(:hsgp_2d))
    return ("outcome ~ 0 + hsgp(x, y, by=group, c=1.5, m=10, share_cov=False)", data)
end

"""
name: HSGP 2D — Anisotropic
source: https://bambinos.github.io/bambi/notebooks/hsgp_2d.html
example: hsgp_2d
dataset: hsgp_2d
formula: "outcome ~ 0 + hsgp(x, y, c=1.5, m=10, iso=False)"
----

Anisotropic 2D HSGP allowing different length scales per dimension. Family: gaussian.
"""
function examples(::Val{:hsgp_2d_aniso})
    data = load(Val(:hsgp_2d))
    return ("outcome ~ 0 + hsgp(x, y, c=1.5, m=10, iso=False)", data)
end

"""
name: HSGP 2D — Spatial Poisson Count Data
source: https://bambinos.github.io/bambi/notebooks/hsgp_2d.html
example: hsgp_2d
dataset: poisson_data
formula: "Count ~ 0 + Year + X1:Year + (1|Site) + hsgp(Lon, Lat, by=Year, m=5, c=1.5)"
----

Poisson regression combining fixed effects, random site intercepts, and a 2D spatial
HSGP term varying by year. Family: poisson.
"""
function examples(::Val{:hsgp_2d_poisson})
    data = load(Val(:poisson_data))
    return ("Count ~ 0 + Year + X1:Year + (1|Site) + hsgp(Lon, Lat, by=Year, m=5, c=1.5)", data)
end

const AUSTIN_CATS_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/AustinCats.csv"

"""
name: survival_model — Austin Cats Adoption Data
source: https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/AustinCats.csv
----

Austin Cats (City of Austin Open Data / McElreath rethinking repo). `days_to_event`
scaled to months (÷ 31); `adopt = "right"` for non-adoptions, `"none"` for adoptions;
`color_id = 1` (black) / `0` (other). Family: exponential, link="log".
"""
function load(::Val{:survival_model})
    df = CSV.read(Downloads.download(AUSTIN_CATS_URL), DataFrame)
    df.adopt    = [r.out_event == "Adoption" ? "none" : "right" for r in eachrow(df)]
    df.color_id = [r.color == "Black" ? 1 : 0 for r in eachrow(df)]
    return select(df, [:days_to_event, :adopt, :color_id])
end

"""
name: Survival Model — Intercept Only
source: https://bambinos.github.io/bambi/notebooks/survival_model.html
example: survival_model
dataset: survival_model
formula: "censored(days_to_event / 31, adopt) ~ 1"
----

Exponential survival model with only an intercept (constant hazard). Family: exponential,
link="log".
"""
function examples(::Val{:survival_intercept})
    data = load(Val(:survival_model))
    return ("censored(days_to_event / 31, adopt) ~ 1", data)
end

"""
name: Survival Model — Color Effect
source: https://bambinos.github.io/bambi/notebooks/survival_model.html
example: survival_model
dataset: survival_model
formula: "censored(days_to_event / 31, adopt) ~ 0 + color_id"
----

Exponential survival model with cat color as a predictor. Family: exponential, link="log".
"""
function examples(::Val{:survival_color})
    data = load(Val(:survival_model))
    return ("censored(days_to_event / 31, adopt) ~ 0 + color_id", data)
end

const CHILD_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/child.csv"

"""
name: survival_discrete_time — Synthetic Discrete Survival Data
source: synthetic
----

200 observations synthetically generated; columns: `treatment`, `age`, `time`, `event`.
Family: bernoulli, link="cloglog".
"""
function load(::Val{:survival_discrete_time})
    rng = MersenneTwister(99)
    n = 200
    treatment = rand(rng, 0:1, n)
    age       = rand(rng, 20:60, n)
    time      = rand(rng, 1:10, n)
    event     = Int.(rand(rng, n) .< 0.1 .+ 0.05 .* treatment)
    return DataFrame(; treatment, age, time, event)
end

"""
name: child_mortality — Swedish Child Mortality Records
source: https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/child.csv
----

Swedish 19th-century parish child mortality records; distributed with Bambi notebook
examples. Aggregated to `(events, at_risk)` per stratum via person-period expansion.
Columns include `events`, `at_risk`, `sex`, `socBranch`, `period`, `birth_decade`.
"""
function load(::Val{:child_mortality})
    return CSV.read(Downloads.download(CHILD_URL), DataFrame)
end

"""
name: Discrete Time Survival — Simulated Data
source: https://bambinos.github.io/bambi/notebooks/survival_discrete_time_notebook.html
example: survival_discrete_time
dataset: survival_discrete_time
formula: "event ~ treatment + age + time"
----

Bernoulli survival model with complementary log-log link on synthetic discrete-time data.
Family: bernoulli, link="cloglog".
"""
function examples(::Val{:survival_disc_sim})
    data = load(Val(:survival_discrete_time))
    return ("event ~ treatment + age + time", data)
end

"""
name: Discrete Time Survival — Binomial Child Mortality
source: https://bambinos.github.io/bambi/notebooks/survival_discrete_time_notebook.html
example: survival_discrete_time
dataset: child_mortality
formula: "p(events, at_risk) ~ sex + socBranch + period + scale(birth_decade)"
----

Binomial discrete-time survival model on Swedish child mortality data.
Family: binomial, link="cloglog".
"""
function examples(::Val{:survival_disc_binomial})
    data = load(Val(:child_mortality))
    return ("p(events, at_risk) ~ sex + socBranch + period + scale(birth_decade)", data)
end

"""
name: Discrete Time Survival — Spline Baseline Hazard
source: https://bambinos.github.io/bambi/notebooks/survival_discrete_time_notebook.html
example: survival_discrete_time
dataset: child_mortality
formula: "p(events, at_risk) ~ sex + socBranch + bs(period, df=4) + scale(birth_decade)"
----

Binomial discrete-time survival with spline baseline hazard. Family: binomial, link="cloglog".
"""
function examples(::Val{:survival_disc_spline})
    data = load(Val(:child_mortality))
    return ("p(events, at_risk) ~ sex + socBranch + bs(period, df=4) + scale(birth_decade)", data)
end

"""
name: Discrete Time Survival — Poisson Alternative
source: https://bambinos.github.io/bambi/notebooks/survival_discrete_time_notebook.html
example: survival_discrete_time
dataset: child_mortality
formula: "events ~ sex + socBranch + period + scale(birth_decade) + offset(log(at_risk))"
----

Poisson formulation of discrete-time survival with log-offset for person-time.
Family: poisson.
"""
function examples(::Val{:survival_disc_poisson})
    data = load(Val(:child_mortality))
    return ("events ~ sex + socBranch + period + scale(birth_decade) + offset(log(at_risk))", data)
end

const RETENTION_URL = "https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/retention.csv"

"""
name: survival_continuous_time — Synthetic Weibull Survival Data
source: synthetic
----

1000 subjects; columns: `time`, `censoring` ("right" or "none"), `treatment`, `age`.
Generated from Weibull distribution with treatment and age effects.
"""
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

"""
name: retention — Employee Retention Data
source: https://raw.githubusercontent.com/bambinos/bambi/main/docs/notebooks/data/retention.csv
----

3770 employees, monthly tenure data; distributed with Bambi notebook examples.
Columns include `month`, `censoring`, `gender`, `level`, `field`, `sentiment`, `intention`.
"""
function load(::Val{:retention})
    return CSV.read(Downloads.download(RETENTION_URL), DataFrame)
end

"""
name: Continuous Time Survival — Weibull
source: https://bambinos.github.io/bambi/notebooks/survival_continuous_time_notebook.html
example: survival_continuous_time
dataset: survival_continuous_time
formula: "censored(time, censoring) ~ treatment + age"
----

Weibull survival model on synthetic data with treatment and age effects.
Family: weibull.
"""
function examples(::Val{:survival_cont_weibull})
    data = load(Val(:survival_continuous_time))
    return ("censored(time, censoring) ~ treatment + age", data)
end

"""
name: Continuous Time Survival — Retention Fixed Effects
source: https://bambinos.github.io/bambi/notebooks/survival_continuous_time_notebook.html
example: survival_continuous_time
dataset: retention
formula: "censored(month, censoring) ~ C(gender) + C(level) + C(field) + sentiment + intention"
----

Weibull survival model of employee tenure with fixed effects only.
Family: weibull.
"""
function examples(::Val{:survival_cont_retention_fe})
    data = load(Val(:retention))
    return ("censored(month, censoring) ~ C(gender) + C(level) + C(field) + sentiment + intention", data)
end

"""
name: Continuous Time Survival — Retention Random Effects
source: https://bambinos.github.io/bambi/notebooks/survival_continuous_time_notebook.html
example: survival_continuous_time
dataset: retention
formula: "censored(month, censoring) ~ C(gender) + C(level) + sentiment + intention + (1|field)"
----

Weibull survival model of employee tenure with random intercepts for field.
Family: weibull.
"""
function examples(::Val{:survival_cont_retention_re})
    data = load(Val(:retention))
    return ("censored(month, censoring) ~ C(gender) + C(level) + sentiment + intention + (1|field)", data)
end

const MPG_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"

"""
name: orthogonal_polynomial — MPG Dataset
source: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
----

1970–1982 automobiles (seaborn / EPA); rows with missing `horsepower` dropped.
Columns include `mpg`, `horsepower`.
"""
load(::Val{:orthogonal_polynomial}) = CSV.read(Downloads.download(MPG_URL), DataFrame)

"""
name: orthopoly_projectile — Synthetic Projectile Motion Data
source: synthetic
----

Synthetic projectile motion: `t` ∈ [0, 2], `x ≈ 10·t - 0.5·g·t²` (100 obs, g=9.8),
filtered to `x ≥ 0`.
"""
function load(::Val{:orthopoly_projectile})
    rng = MersenneTwister(0)
    g = 9.8
    tp = collect(range(0, 2, length=100))
    xp = 10 .* tp .- 0.5 .* g .* tp .^ 2 .+ 0.3 .* randn(rng, 100)
    return filter(r -> r.x >= 0, DataFrame(t=tp, x=xp))
end

"""
name: Orthogonal Polynomial — Explicit Quadratic
source: https://bambinos.github.io/bambi/notebooks/orthogonal_polynomial_reg.html
example: orthogonal_polynomial
dataset: orthopoly_projectile
formula: "x ~ I(t**2) + t + 1"
----

Explicit (raw) quadratic polynomial on projectile data. Family: gaussian.
"""
function examples(::Val{:orthopoly_explicit})
    data = load(Val(:orthopoly_projectile))
    return ("x ~ I(t**2) + t + 1", data)
end

"""
name: Orthogonal Polynomial — poly() Degree 2
source: https://bambinos.github.io/bambi/notebooks/orthogonal_polynomial_reg.html
example: orthogonal_polynomial
dataset: orthopoly_projectile
formula: "x ~ poly(t, 2) + 1"
----

Orthogonal polynomial degree 2 on projectile data. Family: gaussian.
"""
function examples(::Val{:orthopoly_poly})
    data = load(Val(:orthopoly_projectile))
    return ("x ~ poly(t, 2) + 1", data)
end

"""
name: Orthogonal Polynomial — MPG Linear
source: https://bambinos.github.io/bambi/notebooks/orthogonal_polynomial_reg.html
example: orthogonal_polynomial
dataset: orthogonal_polynomial
formula: "mpg ~ horsepower"
----

Linear regression of fuel efficiency on horsepower; baseline model. Family: gaussian.
"""
function examples(::Val{:orthopoly_linear})
    mpg = dropmissing(load(Val(:orthogonal_polynomial)), [:horsepower, :mpg])
    return ("mpg ~ horsepower", mpg)
end

"""
name: Orthogonal Polynomial — MPG Quadratic
source: https://bambinos.github.io/bambi/notebooks/orthogonal_polynomial_reg.html
example: orthogonal_polynomial
dataset: orthogonal_polynomial
formula: "mpg ~ poly(horsepower, 2)"
----

Orthogonal polynomial degree 2 regression of mpg on horsepower. Family: gaussian.
"""
function examples(::Val{:orthopoly_quad})
    mpg = dropmissing(load(Val(:orthogonal_polynomial)), [:horsepower, :mpg])
    return ("mpg ~ poly(horsepower, 2)", mpg)
end

"""
name: Orthogonal Polynomial — MPG Degree Comparison
source: https://bambinos.github.io/bambi/notebooks/orthogonal_polynomial_reg.html
example: orthogonal_polynomial
dataset: orthogonal_polynomial
formula: "mpg ~ poly(horsepower, degree)"
----

Orthogonal polynomial regression template for degree ∈ 1:9; used for model comparison.
Family: gaussian.
"""
function examples(::Val{:orthopoly_compare})
    mpg = dropmissing(load(Val(:orthogonal_polynomial)), [:horsepower, :mpg])
    return ("mpg ~ poly(horsepower, degree)", mpg)
end

const MTCARS_URL = "https://ndownloader.figshare.com/files/40208785"
const IMDB_URL   = "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2movies/movies.csv"

"""
name: plot_predictions — mtcars Dataset
source: https://ndownloader.figshare.com/files/40208785
----

32 automobiles (`bmb.load_data("mtcars")`); `hp` cast to Float32, `cyl` recoded to
"low"/"medium"/"high", `gear` recoded to "A"/"B"/"C".
"""
load(::Val{:plot_predictions}) = CSV.read(Downloads.download(MTCARS_URL), DataFrame)

"""
name: plot_pred_movies — IMDB Movies Data
source: https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2movies/movies.csv
----

28,819 IMDB films from ggplot2movies; filtered to `length < 240` and complete cases.
`style` derived from Action/Comedy/Drama flags; `certified_fresh = rating >= 8`.
"""
function load(::Val{:plot_pred_movies})
    movies = dropmissing(filter(r -> r.length < 240,
                CSV.read(Downloads.download(IMDB_URL), DataFrame)))
    movies.style = [r.Action == 1 ? "Action" : r.Comedy == 1 ? "Comedy" : "Drama"
                    for r in eachrow(movies)]
    movies.certified_fresh = Int.(movies.rating .>= 8)
    return movies
end

"""
name: Plot Predictions — Linear Regression (mtcars)
source: https://bambinos.github.io/bambi/notebooks/plot_predictions.html
example: plot_predictions
dataset: plot_predictions
formula: "mpg ~ 0 + hp * wt + cyl + gear"
----

Linear regression of fuel efficiency on horsepower, weight, cylinders and gear.
Family: gaussian.
"""
function examples(::Val{:plot_pred_linear})
    mtcars = load(Val(:plot_predictions))
    mtcars.hp  = Float32.(mtcars.hp)
    mtcars.cyl = [c <= 4 ? "low" : c <= 6 ? "medium" : "high" for c in mtcars.cyl]
    mtcars.gear = [g == 3 ? "A" : g == 4 ? "B" : "C" for g in mtcars.gear]
    return ("mpg ~ 0 + hp * wt + cyl + gear", mtcars)
end

"""
name: Plot Predictions — Negative Binomial (Student Absences)
source: https://bambinos.github.io/bambi/notebooks/plot_predictions.html
example: plot_predictions
dataset: negative_binomial
formula: "daysabs ~ 0 + prog + scale(math) + prog:scale(math)"
----

Negative binomial regression of student absences; dataset requires Stata loading (see
:negative_binomial). Placeholder DataFrame() used here. Family: negativebinomial.
"""
function examples(::Val{:plot_pred_nb})
    return ("daysabs ~ 0 + prog + scale(math) + prog:scale(math)", DataFrame())
end

"""
name: Plot Predictions — Logistic Regression (Movies)
source: https://bambinos.github.io/bambi/notebooks/plot_predictions.html
example: plot_predictions
dataset: plot_pred_movies
formula: "certified_fresh ~ 0 + scale(length) * style"
----

Logistic regression of "certified fresh" rating on film length and genre style.
Family: bernoulli.
"""
function examples(::Val{:plot_pred_logistic})
    movies = load(Val(:plot_pred_movies))
    return ("certified_fresh ~ 0 + scale(length) * style", movies)
end

"""
name: Plot Predictions — Distributional Gamma
source: https://bambinos.github.io/bambi/notebooks/plot_predictions.html
example: plot_predictions
dataset: distributional_synth
formula: "y ~ x  +  alpha ~ x"
----

Distributional gamma regression on synthetic data; same model as :distributional_var_alpha.
Family: gamma.
"""
function examples(::Val{:plot_pred_distr})
    rng = MersenneTwister(121195)
    x = rand(rng, 200) .* 3 .- 1.5
    y = exp.(1.0 .+ 0.5 .* x .+ 0.2 .* randn(rng, 200))
    synth = DataFrame(; x, y)
    return ("y ~ x  +  alpha ~ x", synth)
end

const TITANIC_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Titanic.csv"

"""
name: plot_comparisons — Fish Catch Survey
source: https://stats.idre.ucla.edu/stat/data/fish.csv
----

250 groups at a state park; columns: `count`, `livebait`, `camper`, `persons`, `child`.
Same data as :zero_inflated (not pre-filtered here).
"""
load(::Val{:plot_comparisons}) = CSV.read(Downloads.download(FISH_URL), DataFrame)

"""
name: titanic — Titanic Survival Data
source: https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Titanic.csv
----

Titanic survival data (Stat2Data via Rdatasets); complete cases only. `PClass` and
`SexCode` treated as ordered categorical. Family: bernoulli.
"""
function load(::Val{:titanic})
    return dropmissing(CSV.read(Downloads.download(TITANIC_URL), DataFrame))
end

"""
name: Plot Comparisons — ZIP Fish Count
source: https://bambinos.github.io/bambi/notebooks/plot_comparisons.html
example: plot_comparisons
dataset: plot_comparisons
formula: "count ~ livebait + camper + persons + child"
----

Zero-inflated Poisson model of fish catch count. Family: zero_inflated_poisson.
"""
function examples(::Val{:plot_comp_zip})
    fish = filter(r -> r.count < 60, load(Val(:plot_comparisons)))
    return ("count ~ livebait + camper + persons + child", fish)
end

"""
name: Plot Comparisons — Logistic Titanic Survival
source: https://bambinos.github.io/bambi/notebooks/plot_comparisons.html
example: plot_comparisons
dataset: titanic
formula: "Survived ~ PClass * SexCode * Age"
----

Logistic regression of Titanic survival on passenger class, sex and age with interactions.
Family: bernoulli.
"""
function examples(::Val{:plot_comp_logistic})
    data = load(Val(:titanic))
    return ("Survived ~ PClass * SexCode * Age", data)
end

const WELLS_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Wells.csv"

"""
name: plot_slopes — Bangladesh Well-Switching Data
source: https://vincentarelbundock.github.io/Rdatasets/csv/carData/Wells.csv
----

Bangladesh well-switching data (carData via Rdatasets) — 3020 obs.
`dist100 = distance / 100`; `educ4 = education / 4`; `switch` as integer. Family: bernoulli.
"""
function load(::Val{:plot_slopes})
    df = CSV.read(Downloads.download(WELLS_URL), DataFrame)
    df.dist100 = df.distance ./ 100
    df.educ4   = df.education ./ 4
    df.switch  = Int.(df.switch)
    return df
end

"""
name: Plot Slopes — Main Effects
source: https://bambinos.github.io/bambi/notebooks/plot_slopes.html
example: plot_slopes
dataset: plot_slopes
formula: "switch ~ dist100 + arsenic + educ4"
----

Logistic regression of well-switching on distance, arsenic level and education.
Family: bernoulli.
"""
function examples(::Val{:plot_slopes_main})
    data = load(Val(:plot_slopes))
    return ("switch ~ dist100 + arsenic + educ4", data)
end

"""
name: Plot Slopes — With Interactions
source: https://bambinos.github.io/bambi/notebooks/plot_slopes.html
example: plot_slopes
dataset: plot_slopes
formula: "switch ~ dist100 + arsenic + educ4 + dist100:educ4 + arsenic:educ4"
----

Logistic regression adding education interaction terms with distance and arsenic.
Family: bernoulli.
"""
function examples(::Val{:plot_slopes_interaction})
    data = load(Val(:plot_slopes))
    return ("switch ~ dist100 + arsenic + educ4 + dist100:educ4 + arsenic:educ4", data)
end

"""
name: alternative_samplers — Synthetic Linear Regression Data
source: synthetic
----

Synthetically generated linear regression data (100 obs; columns: `x`, `y`).
Demonstrates fitting the same model with blackjax, numpyro, and nutpie backends.
"""
function load(::Val{:alternative_samplers})
    rng = MersenneTwister(0)
    x = randn(rng, 100)
    y = 2.0 .+ 1.5 .* x .+ randn(rng, 100)
    return DataFrame(; x, y)
end

"""
name: Using Other Samplers — JAX-Based Samplers
source: https://bambinos.github.io/bambi/notebooks/alternative_samplers.html
example: alternative_samplers
dataset: alternative_samplers
formula: "y ~ x"
----

Linear regression on synthetic data; demonstrates blackjax, numpyro and nutpie backends.
Family: gaussian.
"""
function examples(::Val{:alternative_samplers})
    data = load(Val(:alternative_samplers))
    return ("y ~ x", data)
end
