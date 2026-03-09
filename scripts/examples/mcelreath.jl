using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: "Statistical Rethinking with brms, ggplot2, and the tidyverse: 2nd ed."
# A. Solomon Kurz (brms translation of Richard McElreath's Statistical Rethinking, 2nd ed.)
# Source: https://github.com/ASKurz/Statistical_Rethinking_with_brms_ggplot2_and_the_tidyverse_2_ed
# Data:   https://github.com/rmcelreath/rethinking/tree/master/data

const RETHINKING_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/"

"""
name: :globe — Globe Tossing Data
source: synthetic
chapter: SR2 Ch 2–3
----

Binomial likelihood for estimating p(water) from a globe-tossing experiment.
Data: 24 water observations out of 36 tosses (Ch 2); 6 out of 9 (Ch 3 recap).
"""
load(::Val{:globe}) = DataFrame(w = [24], n = [36])

"""
name: Globe Tossing
source: synthetic
example: globe
dataset: globe
chapter: SR2 Ch 2–3
formula: "w | trials(n) ~ 0 + Intercept"
----

Binomial likelihood for estimating p(water) from a globe-tossing experiment.
`family = binomial(link = "identity")`; `0 + Intercept` fixes the link to identity,
estimating `p` directly.
"""
function examples(::Val{:globe})
    data = load(Val(:globe))
    return ("w | trials(n) ~ 0 + Intercept", data)
end

"""
name: :howell1 — !Kung San Census Data
source: https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv
chapter: SR2 Ch 4
----

!Kung San census data (Nancy Howell): 544 individuals with `height` (cm),
`weight` (kg), `age` (years), `male` (0/1).
"""
load(::Val{:howell1}) = CSV.read(Downloads.download(RETHINKING_URL * "Howell1.csv"), DataFrame)

"""
name: Heights and Weights — Intercept Only
source: https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv
example: howell1
dataset: howell1
chapter: SR2 Ch 4
formula: "height ~ 1"
----

Intercept-only Gaussian regression on adult heights (age >= 18, n=352).
Used as a prior predictive / baseline model.
"""
function examples(::Val{:howell1_intercept})
    d  = load(Val(:howell1))
    d2 = filter(r -> r.age >= 18, d)
    d2.weight_c = d2.weight .- mean(d2.weight)
    return ("height ~ 1", d2)
end

"""
name: Heights and Weights — Linear Regression
source: https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv
example: howell1
dataset: howell1
chapter: SR2 Ch 4
formula: "height ~ 1 + weight_c"
----

Simple linear regression of height on mean-centered weight in adults (age >= 18).
`weight_c = weight - mean(weight)`.
"""
function examples(::Val{:howell1_linear})
    d  = load(Val(:howell1))
    d2 = filter(r -> r.age >= 18, d)
    d2.weight_c = d2.weight .- mean(d2.weight)
    return ("height ~ 1 + weight_c", d2)
end

"""
name: Heights and Weights — Quadratic Polynomial
source: https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv
example: howell1
dataset: howell1
chapter: SR2 Ch 4
formula: "height ~ 1 + weight_s + weight_s2"
----

Quadratic polynomial regression of height on standardized weight using the full dataset
(all ages). `weight_s` = standardized weight, `weight_s2 = weight_s^2`.
"""
function examples(::Val{:howell1_quad})
    d = load(Val(:howell1))
    d.weight_s  = (d.weight .- mean(d.weight)) ./ std(d.weight)
    d.weight_s2 = d.weight_s .^ 2
    d.weight_s3 = d.weight_s .^ 3
    return ("height ~ 1 + weight_s + weight_s2", d)
end

"""
name: Heights and Weights — Cubic Polynomial
source: https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv
example: howell1
dataset: howell1
chapter: SR2 Ch 4
formula: "height ~ 1 + weight_s + weight_s2 + weight_s3"
----

Cubic polynomial regression of height on standardized weight using the full dataset
(all ages). `weight_s3 = weight_s^3`.
"""
function examples(::Val{:howell1_cubic})
    d = load(Val(:howell1))
    d.weight_s  = (d.weight .- mean(d.weight)) ./ std(d.weight)
    d.weight_s2 = d.weight_s .^ 2
    d.weight_s3 = d.weight_s .^ 3
    return ("height ~ 1 + weight_s + weight_s2 + weight_s3", d)
end

"""
name: Heights and Weights — Nonlinear Exponential Growth
source: https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv
example: howell1
dataset: howell1
chapter: SR2 Ch 4
formula: "bf(height ~ a + exp(lb) * weight_c, a ~ 1, lb ~ 1, nl = TRUE)"
----

Nonlinear exponential growth model for height on mean-centered weight in adults
(age >= 18). `weight_c = weight - mean(weight)`.
"""
function examples(::Val{:howell1_nl})
    d  = load(Val(:howell1))
    d2 = filter(r -> r.age >= 18, d)
    d2.weight_c = d2.weight .- mean(d2.weight)
    return ("bf(height ~ a + exp(lb) * weight_c, a ~ 1, lb ~ 1, nl = TRUE)", d2)
end

"""
name: Heights and Weights — Nonlinear Volume Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv
example: howell1
dataset: howell1
chapter: SR2 Ch 16
formula: "bf(w ~ log(3.141593 * k * p^2 * h^3), k + p ~ 1, nl = TRUE)"
----

Lognormal nonlinear model treating body weight as a scaled cylinder. `w` = weight,
`h` = height. Models body weight as a scaled cylinder volume.
"""
function examples(::Val{:howell1_volume})
    d = load(Val(:howell1))
    return ("bf(w ~ log(3.141593 * k * p^2 * h^3), k + p ~ 1, nl = TRUE)",
            DataFrame(w = d.weight, h = d.height))
end

"""
name: :cherry_blossoms — Cherry Blossom Bloom Dates
source: https://github.com/rmcelreath/rethinking/blob/master/data/cherry_blossoms.csv
chapter: SR2 Ch 4
----

Cherry blossom bloom dates in Japan, years 801–2015; 827 observations after dropping
rows with missing `doy` (day of year). Also covered in `bambi.jl` (`:cherry_blossoms`)
with Bambi/Patsy spline syntax.
"""
load(::Val{:cherry_blossoms}) =
    CSV.read(Downloads.download(RETHINKING_URL * "cherry_blossoms.csv"), DataFrame)

"""
name: Cherry Blossoms — B-Spline Regression
source: https://github.com/rmcelreath/rethinking/blob/master/data/cherry_blossoms.csv
example: cherry_blossoms
dataset: cherry_blossoms
chapter: SR2 Ch 4
formula: "doy ~ 1 + B"
----

`B` is a B-spline basis matrix (each column is a basis function); constructed via
`bs(year, knots=iknots)` in R with ~17 basis columns.
"""
function examples(::Val{:cherry_blossoms})
    d = dropmissing(load(Val(:cherry_blossoms)), :doy)
    # B is a B-spline design matrix; here we note the formula symbolically.
    # In practice: B = R's bs(d$year, knots=quantile(d$year, probs=seq(0.1,0.9,by=0.1)))
    return ("doy ~ 1 + B", d)
end

"""
name: :waffle_divorce — Waffle House and Divorce Rates
source: https://github.com/rmcelreath/rethinking/blob/master/data/WaffleDivorce.csv
chapter: SR2 Ch 5
----

U.S. state-level divorce and marriage statistics; 50 states: `Divorce` rate,
`Marriage` rate, `MedianAgeMarriage`, `WaffleHouses`, `South`, etc.
"""
load(::Val{:waffle_divorce}) =
    CSV.read(Downloads.download(RETHINKING_URL * "WaffleDivorce.csv"), DataFrame)

"""
name: Waffle Divorce — Age Predicts Divorce
source: https://github.com/rmcelreath/rethinking/blob/master/data/WaffleDivorce.csv
example: waffle_divorce
dataset: waffle_divorce
chapter: SR2 Ch 5
formula: "D ~ 1 + A"
----

Simple regression of standardized divorce rate on standardized median age at marriage.
`D` = standardized Divorce, `A` = standardized MedianAgeMarriage.
"""
function examples(::Val{:waffle_divorce_a})
    d = load(Val(:waffle_divorce))
    d.D = (d.Divorce           .- mean(d.Divorce))           ./ std(d.Divorce)
    d.A = (d.MedianAgeMarriage .- mean(d.MedianAgeMarriage)) ./ std(d.MedianAgeMarriage)
    d.M = (d.Marriage          .- mean(d.Marriage))          ./ std(d.Marriage)
    d.D_sd  = 0.1 .* abs.(d.D) .+ 0.01
    d.D_obs = d.D .+ d.D_sd .* randn(MersenneTwister(15), nrow(d))
    return ("D ~ 1 + A", d)
end

"""
name: Waffle Divorce — Marriage Rate Predicts Divorce
source: https://github.com/rmcelreath/rethinking/blob/master/data/WaffleDivorce.csv
example: waffle_divorce
dataset: waffle_divorce
chapter: SR2 Ch 5
formula: "D ~ 1 + M"
----

Simple regression of standardized divorce rate on standardized marriage rate.
`D` = standardized Divorce, `M` = standardized Marriage.
"""
function examples(::Val{:waffle_divorce_m})
    d = load(Val(:waffle_divorce))
    d.D = (d.Divorce           .- mean(d.Divorce))           ./ std(d.Divorce)
    d.A = (d.MedianAgeMarriage .- mean(d.MedianAgeMarriage)) ./ std(d.MedianAgeMarriage)
    d.M = (d.Marriage          .- mean(d.Marriage))          ./ std(d.Marriage)
    d.D_sd  = 0.1 .* abs.(d.D) .+ 0.01
    d.D_obs = d.D .+ d.D_sd .* randn(MersenneTwister(15), nrow(d))
    return ("D ~ 1 + M", d)
end

"""
name: Waffle Divorce — Multiple Regression (M's effect vanishes)
source: https://github.com/rmcelreath/rethinking/blob/master/data/WaffleDivorce.csv
example: waffle_divorce
dataset: waffle_divorce
chapter: SR2 Ch 5
formula: "D ~ 1 + M + A"
----

Multiple regression; M's effect vanishes after conditioning on A, revealing a spurious
association between marriage rate and divorce.
"""
function examples(::Val{:waffle_divorce_am})
    d = load(Val(:waffle_divorce))
    d.D = (d.Divorce           .- mean(d.Divorce))           ./ std(d.Divorce)
    d.A = (d.MedianAgeMarriage .- mean(d.MedianAgeMarriage)) ./ std(d.MedianAgeMarriage)
    d.M = (d.Marriage          .- mean(d.Marriage))          ./ std(d.Marriage)
    d.D_sd  = 0.1 .* abs.(d.D) .+ 0.01
    d.D_obs = d.D .+ d.D_sd .* randn(MersenneTwister(15), nrow(d))
    return ("D ~ 1 + M + A", d)
end

"""
name: Waffle Divorce — Mediator Regression (A predicts M)
source: https://github.com/rmcelreath/rethinking/blob/master/data/WaffleDivorce.csv
example: waffle_divorce
dataset: waffle_divorce
chapter: SR2 Ch 5
formula: "M ~ 1 + A"
----

Mediator regression: age at marriage predicts marriage rate. Part of the causal model
where A -> M -> D and A -> D.
"""
function examples(::Val{:waffle_divorce_mediator})
    d = load(Val(:waffle_divorce))
    d.D = (d.Divorce           .- mean(d.Divorce))           ./ std(d.Divorce)
    d.A = (d.MedianAgeMarriage .- mean(d.MedianAgeMarriage)) ./ std(d.MedianAgeMarriage)
    d.M = (d.Marriage          .- mean(d.Marriage))          ./ std(d.Marriage)
    d.D_sd  = 0.1 .* abs.(d.D) .+ 0.01
    d.D_obs = d.D .+ d.D_sd .* randn(MersenneTwister(15), nrow(d))
    return ("M ~ 1 + A", d)
end

"""
name: Waffle Divorce — Multivariate Simultaneous Causal Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/WaffleDivorce.csv
example: waffle_divorce
dataset: waffle_divorce
chapter: SR2 Ch 5
formula: "bf(D ~ 1 + M + A) + bf(M ~ 1 + A) + set_rescor(FALSE)"
----

Multivariate brms model; simultaneous causal model for D and M with residual correlation
disabled via `set_rescor(FALSE)`.
"""
function examples(::Val{:waffle_divorce_multivariate})
    d = load(Val(:waffle_divorce))
    d.D = (d.Divorce           .- mean(d.Divorce))           ./ std(d.Divorce)
    d.A = (d.MedianAgeMarriage .- mean(d.MedianAgeMarriage)) ./ std(d.MedianAgeMarriage)
    d.M = (d.Marriage          .- mean(d.Marriage))          ./ std(d.Marriage)
    d.D_sd  = 0.1 .* abs.(d.D) .+ 0.01
    d.D_obs = d.D .+ d.D_sd .* randn(MersenneTwister(15), nrow(d))
    return ("bf(D ~ 1 + M + A) + bf(M ~ 1 + A) + set_rescor(FALSE)", d)
end

"""
name: Waffle Divorce — Measurement Error Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/WaffleDivorce.csv
example: waffle_divorce
dataset: waffle_divorce
chapter: SR2 Ch 15
formula: "D_obs | mi(D_sd) ~ 1 + A + M"
----

Measurement error model from Ch 15; `D_sd` is the known standard error of the divorce
estimate (approximated as 0.1 * |D| + 0.01). `D_obs` is the noisy observation.
"""
function examples(::Val{:waffle_divorce_meas_err})
    d = load(Val(:waffle_divorce))
    d.D = (d.Divorce           .- mean(d.Divorce))           ./ std(d.Divorce)
    d.A = (d.MedianAgeMarriage .- mean(d.MedianAgeMarriage)) ./ std(d.MedianAgeMarriage)
    d.M = (d.Marriage          .- mean(d.Marriage))          ./ std(d.Marriage)
    d.D_sd  = 0.1 .* abs.(d.D) .+ 0.01
    d.D_obs = d.D .+ d.D_sd .* randn(MersenneTwister(15), nrow(d))
    return ("D_obs | mi(D_sd) ~ 1 + A + M", d)
end

"""
name: :milk — Primate Milk Composition
source: https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv
chapter: SR2 Ch 5–6
----

Primate milk composition and life history; 29 species (12 with complete
`neocortex.perc` data). Columns: `clade`, `species`, `kcal.per.g`, `perc.fat`,
`perc.protein`, `perc.lactose`, `mass` (kg), `neocortex.perc` (% brain that is neocortex).
"""
load(::Val{:milk}) =
    CSV.read(Downloads.download(RETHINKING_URL * "milk.csv"), DataFrame)

"""
name: Milk Energy — Neocortex Predictor
source: https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv
example: milk
dataset: milk
chapter: SR2 Ch 5–6
formula: "kcal_s ~ 1 + neo_s"
----

Positive association between neocortex percentage and kcal per gram before controlling
for body mass. Uses complete cases (n=12) with non-missing neocortex.perc.
"""
function examples(::Val{:milk_neo})
    d = load(Val(:milk))
    d.neocortex_perc = d[!, "neocortex.perc"]
    dcc = dropmissing(d, :neocortex_perc)
    for df in (d, dcc)
        df.kcal_s    = (df[!, "kcal.per.g"] .- mean(skipmissing(df[!, "kcal.per.g"]))) ./
                        std(skipmissing(df[!, "kcal.per.g"]))
        df.neo_s     = (coalesce.(df.neocortex_perc, NaN) .-
                        mean(skipmissing(df.neocortex_perc))) ./
                        std(skipmissing(df.neocortex_perc))
        df.logmass_s = (log.(df.mass) .- mean(log.(df.mass))) ./ std(log.(df.mass))
    end
    return ("kcal_s ~ 1 + neo_s", dcc)
end

"""
name: Milk Energy — Log Body Mass Predictor
source: https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv
example: milk
dataset: milk
chapter: SR2 Ch 5–6
formula: "kcal_s ~ 1 + logmass_s"
----

Negative association; larger species produce less calorie-dense milk. Uses complete
cases (n=12) with non-missing neocortex.perc.
"""
function examples(::Val{:milk_mass})
    d = load(Val(:milk))
    d.neocortex_perc = d[!, "neocortex.perc"]
    dcc = dropmissing(d, :neocortex_perc)
    for df in (d, dcc)
        df.kcal_s    = (df[!, "kcal.per.g"] .- mean(skipmissing(df[!, "kcal.per.g"]))) ./
                        std(skipmissing(df[!, "kcal.per.g"]))
        df.neo_s     = (coalesce.(df.neocortex_perc, NaN) .-
                        mean(skipmissing(df.neocortex_perc))) ./
                        std(skipmissing(df.neocortex_perc))
        df.logmass_s = (log.(df.mass) .- mean(log.(df.mass))) ./ std(log.(df.mass))
    end
    return ("kcal_s ~ 1 + logmass_s", dcc)
end

"""
name: Milk Energy — Neocortex and Mass (Masked Association)
source: https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv
example: milk
dataset: milk
chapter: SR2 Ch 5–6
formula: "kcal_s ~ 1 + neo_s + logmass_s"
----

Both effects revealed simultaneously — the masked association. Uses complete cases
(n=12) with non-missing neocortex.perc.
"""
function examples(::Val{:milk_both})
    d = load(Val(:milk))
    d.neocortex_perc = d[!, "neocortex.perc"]
    dcc = dropmissing(d, :neocortex_perc)
    for df in (d, dcc)
        df.kcal_s    = (df[!, "kcal.per.g"] .- mean(skipmissing(df[!, "kcal.per.g"]))) ./
                        std(skipmissing(df[!, "kcal.per.g"]))
        df.neo_s     = (coalesce.(df.neocortex_perc, NaN) .-
                        mean(skipmissing(df.neocortex_perc))) ./
                        std(skipmissing(df.neocortex_perc))
        df.logmass_s = (log.(df.mass) .- mean(log.(df.mass))) ./ std(log.(df.mass))
    end
    return ("kcal_s ~ 1 + neo_s + logmass_s", dcc)
end

"""
name: Milk Energy — Clade Index Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv
example: milk
dataset: milk
chapter: SR2 Ch 5–6
formula: "kcal_s ~ 0 + clade"
----

Clade index model with one intercept per primate clade, no overall intercept.
Uses complete cases (n=12) with non-missing neocortex.perc.
"""
function examples(::Val{:milk_clade})
    d = load(Val(:milk))
    d.neocortex_perc = d[!, "neocortex.perc"]
    dcc = dropmissing(d, :neocortex_perc)
    for df in (d, dcc)
        df.kcal_s    = (df[!, "kcal.per.g"] .- mean(skipmissing(df[!, "kcal.per.g"]))) ./
                        std(skipmissing(df[!, "kcal.per.g"]))
        df.neo_s     = (coalesce.(df.neocortex_perc, NaN) .-
                        mean(skipmissing(df.neocortex_perc))) ./
                        std(skipmissing(df.neocortex_perc))
        df.logmass_s = (log.(df.mass) .- mean(log.(df.mass))) ./ std(log.(df.mass))
    end
    return ("kcal_s ~ 0 + clade", dcc)
end

"""
name: Milk Energy — Joint Missing-Data Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv
example: milk
dataset: milk
chapter: SR2 Ch 15
formula: "bf(k ~ 1 + mi(b) + m) + bf(b | mi() ~ 1) + set_rescor(FALSE)"
----

Joint missing-data model from Ch 15; `b` = neocortex.perc with missing values treated
as partially observed. Uses the full dataset (all 29 species, including those with
missing neocortex data).
"""
function examples(::Val{:milk_mi})
    d = load(Val(:milk))
    d.neocortex_perc = d[!, "neocortex.perc"]
    dcc = dropmissing(d, :neocortex_perc)
    for df in (d, dcc)
        df.kcal_s    = (df[!, "kcal.per.g"] .- mean(skipmissing(df[!, "kcal.per.g"]))) ./
                        std(skipmissing(df[!, "kcal.per.g"]))
        df.neo_s     = (coalesce.(df.neocortex_perc, NaN) .-
                        mean(skipmissing(df.neocortex_perc))) ./
                        std(skipmissing(df.neocortex_perc))
        df.logmass_s = (log.(df.mass) .- mean(log.(df.mass))) ./ std(log.(df.mass))
    end
    # For the mi() model we use the full dataset with missings retained
    rename!(d, "kcal.per.g" => "k", "neocortex.perc" => "b")
    d.m = d.logmass_s
    return ("bf(k ~ 1 + mi(b) + m) + bf(b | mi() ~ 1) + set_rescor(FALSE)", d)
end

"""
name: :plant_growth — Synthetic Plant Growth Experiment
source: synthetic
chapter: SR2 Ch 6
----

Synthetic data: 200 plants with `h0` (initial height), `h1` (final height),
`treatment` (0/1), `fungus` (0/1). Fungus is a mediator: treatment → fungus → h1.
Including fungus blocks the treatment effect (post-treatment bias); excluding it
recovers the causal estimate.
"""
function load(::Val{:plant_growth})
    rng = MersenneTwister(6)
    n = 200
    h0        = rand(rng, n) .+ 10.0
    treatment = repeat([0, 1], n ÷ 2)
    fungus    = [rand(rng) < (treatment[i] == 1 ? 0.1 : 0.5) ? 1 : 0 for i in 1:n]
    h1        = h0 .* (1.5 .- 0.2 .* fungus .+ 0.05 .* randn(rng, n))
    return DataFrame(; h0, h1, treatment = float.(treatment), fungus = float.(fungus))
end

"""
name: Plant Growth — Baseline (No Treatment Effect)
source: synthetic
example: plant_growth
dataset: plant_growth
chapter: SR2 Ch 6
formula: "h1 ~ 0 + h0"
----

Baseline model with no treatment effect. Growth proportional to initial height.
"""
function examples(::Val{:plant_growth_baseline})
    d = load(Val(:plant_growth))
    return ("h1 ~ 0 + h0", d)
end

"""
name: Plant Growth — Post-Treatment Bias
source: synthetic
example: plant_growth
dataset: plant_growth
chapter: SR2 Ch 6
formula: "bf(h1 ~ h0 * (a + t * treatment + f * fungus), a + t + f ~ 1, nl = TRUE)"
----

Post-treatment bias: treatment effect absorbed by conditioning on fungus (a mediator).
Including fungus blocks the path from treatment to h1.
"""
function examples(::Val{:plant_growth_biased})
    d = load(Val(:plant_growth))
    return ("bf(h1 ~ h0 * (a + t * treatment + f * fungus), a + t + f ~ 1, nl = TRUE)", d)
end

"""
name: Plant Growth — Causal Model (Excluding Mediator)
source: synthetic
example: plant_growth
dataset: plant_growth
chapter: SR2 Ch 6
formula: "bf(h1 ~ h0 * (a + t * treatment), a + t ~ 1, nl = TRUE)"
----

Causal model: treatment effect recovered by excluding fungus (the mediator).
This gives an unbiased estimate of the total effect of treatment on growth.
"""
function examples(::Val{:plant_growth_causal})
    d = load(Val(:plant_growth))
    return ("bf(h1 ~ h0 * (a + t * treatment), a + t ~ 1, nl = TRUE)", d)
end

"""
name: :happiness — Synthetic Happiness and Marriage Data
source: synthetic
chapter: SR2 Ch 6
----

Synthetic data: 1,000 individuals followed over 10 simulated years. Columns: `age`
(1–65), `happiness` (−2 to 2), `married` (0/1), `mid` (married+1 as index). Marriage
is a collider between age and happiness; conditioning on it induces a spurious negative
association between age and happiness.
"""
function load(::Val{:happiness})
    # Replicates McElreath's sim_happiness() output
    rng = MersenneTwister(6)
    ages     = 1:65
    n_years  = 10
    all_rows = NamedTuple[]
    for _ in 1:n_years
        for age in ages
            happiness = clamp(rand(rng, -2.0:0.01:2.0), -2.0, 2.0)
            married   = happiness > 1.0 ? 1 : (age > 18 && rand(rng) < 0.05 ? 1 : 0)
            push!(all_rows, (; age, happiness, married))
        end
    end
    d = DataFrame(all_rows)
    d.a   = (d.age .- 18) ./ (65 .- 18)   # rescale adult age to [0,1]
    d.mid = d.married .+ 1                  # marriage index (1=unmarried, 2=married)
    return filter(r -> r.age >= 18, d)
end

"""
name: Happiness — Collider Bias (Spurious Age Effect)
source: synthetic
example: happiness
dataset: happiness
chapter: SR2 Ch 6
formula: "happiness ~ 0 + mid + a"
----

`mid` = marriage as index; conditioning on marriage (a collider) induces a spurious
negative association between age and happiness.
"""
function examples(::Val{:happiness_collider})
    d = load(Val(:happiness))
    return ("happiness ~ 0 + mid + a", d)
end

"""
name: Happiness — Causal Model (No Age Effect)
source: synthetic
example: happiness
dataset: happiness
chapter: SR2 Ch 6
formula: "happiness ~ 0 + Intercept + a"
----

No conditioning on marriage; age has no direct effect on happiness in this causal model.
"""
function examples(::Val{:happiness_causal})
    d = load(Val(:happiness))
    return ("happiness ~ 0 + Intercept + a", d)
end

"""
name: :rugged — Terrain Ruggedness and GDP
source: https://github.com/rmcelreath/rethinking/blob/master/data/rugged.csv
chapter: SR2 Ch 8–9
----

Cross-national terrain ruggedness and economic data; 234 countries: `rugged`
(ruggedness index), `log_gdp` (log GDP per capita 2000), `cont_africa` (1 if African
country), `isocode`. Subset to rows with non-missing `rgdppc_2000`.
"""
load(::Val{:rugged}) =
    CSV.read(Downloads.download(RETHINKING_URL * "rugged.csv"), DataFrame)

"""
name: Terrain Ruggedness — Pooled Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/rugged.csv
example: rugged
dataset: rugged
chapter: SR2 Ch 8–9
formula: "log_gdp_std ~ 1 + rugged_std_c"
----

Pooled model ignoring continent. `log_gdp_std` = log GDP standardized to mean,
`rugged_std_c` = ruggedness standardized and centered.
"""
function examples(::Val{:rugged_pooled})
    d  = dropmissing(load(Val(:rugged)), :rgdppc_2000)
    d.log_gdp     = log.(d.rgdppc_2000)
    d.log_gdp_std = d.log_gdp ./ mean(d.log_gdp)
    d.rugged_std  = d.rugged ./ maximum(d.rugged)
    d.rugged_std_c = d.rugged_std .- mean(d.rugged_std)
    d.cid         = ifelse.(d.cont_africa .== 1, "Africa", "Other")
    return ("log_gdp_std ~ 1 + rugged_std_c", d)
end

"""
name: Terrain Ruggedness — Continent Intercepts
source: https://github.com/rmcelreath/rethinking/blob/master/data/rugged.csv
example: rugged
dataset: rugged
chapter: SR2 Ch 8–9
formula: "log_gdp_std ~ 0 + cid + rugged_std_c"
----

Continent intercepts via index `cid`; same ruggedness slope for Africa and Other.
`cid` = "Africa" or "Other".
"""
function examples(::Val{:rugged_continent})
    d  = dropmissing(load(Val(:rugged)), :rgdppc_2000)
    d.log_gdp     = log.(d.rgdppc_2000)
    d.log_gdp_std = d.log_gdp ./ mean(d.log_gdp)
    d.rugged_std  = d.rugged ./ maximum(d.rugged)
    d.rugged_std_c = d.rugged_std .- mean(d.rugged_std)
    d.cid         = ifelse.(d.cont_africa .== 1, "Africa", "Other")
    return ("log_gdp_std ~ 0 + cid + rugged_std_c", d)
end

"""
name: Terrain Ruggedness — Nonlinear Continent Slopes
source: https://github.com/rmcelreath/rethinking/blob/master/data/rugged.csv
example: rugged
dataset: rugged
chapter: SR2 Ch 8–9
formula: "bf(log_gdp_std ~ 0 + a + b * rugged_std_c, a ~ 0 + cid, b ~ 0 + cid, nl = TRUE)"
----

Continent-specific intercepts AND slopes; ruggedness hurts GDP outside Africa but
helps within Africa.
"""
function examples(::Val{:rugged_nl})
    d  = dropmissing(load(Val(:rugged)), :rgdppc_2000)
    d.log_gdp     = log.(d.rgdppc_2000)
    d.log_gdp_std = d.log_gdp ./ mean(d.log_gdp)
    d.rugged_std  = d.rugged ./ maximum(d.rugged)
    d.rugged_std_c = d.rugged_std .- mean(d.rugged_std)
    d.cid         = ifelse.(d.cont_africa .== 1, "Africa", "Other")
    return ("bf(log_gdp_std ~ 0 + a + b * rugged_std_c, a ~ 0 + cid, b ~ 0 + cid, nl = TRUE)", d)
end

"""
name: :tulips — Greenhouse Tulip Experiment
source: https://github.com/rmcelreath/rethinking/blob/master/data/tulips.csv
chapter: SR2 Ch 8
----

Greenhouse tulip experiment; 27 plants: `blooms` (flower size), `water` (1–3),
`shade` (1–3), `bed` (a/b/c).
"""
load(::Val{:tulips}) =
    CSV.read(Downloads.download(RETHINKING_URL * "tulips.csv"), DataFrame)

"""
name: Tulips — Additive Water and Shade Effects
source: https://github.com/rmcelreath/rethinking/blob/master/data/tulips.csv
example: tulips
dataset: tulips
chapter: SR2 Ch 8
formula: "blooms_std ~ 1 + water_cent + shade_cent"
----

Additive model; `water_cent` and `shade_cent` are mean-centered. No interaction term.
"""
function examples(::Val{:tulips_additive})
    d = load(Val(:tulips))
    d.blooms_std  = d.blooms ./ maximum(d.blooms)
    d.water_cent  = d.water .- mean(d.water)
    d.shade_cent  = d.shade .- mean(d.shade)
    return ("blooms_std ~ 1 + water_cent + shade_cent", d)
end

"""
name: Tulips — Water × Shade Interaction
source: https://github.com/rmcelreath/rethinking/blob/master/data/tulips.csv
example: tulips
dataset: tulips
chapter: SR2 Ch 8
formula: "blooms_std ~ 1 + water_cent + shade_cent + water_cent:shade_cent"
----

Interaction model: the effect of water on blooms depends on shade level.
"""
function examples(::Val{:tulips_interaction})
    d = load(Val(:tulips))
    d.blooms_std  = d.blooms ./ maximum(d.blooms)
    d.water_cent  = d.water .- mean(d.water)
    d.shade_cent  = d.shade .- mean(d.shade)
    return ("blooms_std ~ 1 + water_cent + shade_cent + water_cent:shade_cent", d)
end

"""
name: :hetero — Synthetic Heteroscedastic Data
source: synthetic
chapter: SR2 Ch 10
----

Synthetic data; `sigma` varies with predictor `x`. Used to illustrate distributional
(sigma-regression) models.
"""
function load(::Val{:hetero})
    rng = MersenneTwister(10)
    n = 100
    x = randn(rng, n)
    y = randn(rng, n) .* exp.(0.5 .* x)
    return DataFrame(; y, x)
end

"""
name: Distributional Model — Heteroscedastic Gaussian
source: synthetic
example: hetero
dataset: hetero
chapter: SR2 Ch 10
formula: "bf(y ~ 1, sigma ~ 1 + x)"
----

Distributional model: both mean and log(`sigma`) have their own sub-models.
`sigma` grows with `x`, producing heteroscedastic residuals.
"""
function examples(::Val{:hetero})
    d = load(Val(:hetero))
    return ("bf(y ~ 1, sigma ~ 1 + x)", d)
end

"""
name: :chimpanzees — Prosocial Choice Experiment
source: https://github.com/rmcelreath/rethinking/blob/master/data/chimpanzees.csv
chapter: SR2 Ch 11
----

Lever-pulling experiment across 7 actors and 6 blocks; 504 trials: `pulled_left`
(0/1), `prosoc_left` (0/1), `condition` (0/1), `actor` (1–7), `block` (1–6).
`treatment = 1 + prosoc_left + 2*condition`.
"""
load(::Val{:chimpanzees}) =
    CSV.read(Downloads.download(RETHINKING_URL * "chimpanzees.csv"), DataFrame)

"""
name: Chimpanzees — Intercept Only
source: https://github.com/rmcelreath/rethinking/blob/master/data/chimpanzees.csv
example: chimpanzees
dataset: chimpanzees
chapter: SR2 Ch 11
formula: "pulled_left | trials(1) ~ 1"
----

Intercept-only binomial; baseline probability of pulling left across all actors
and treatments.
"""
function examples(::Val{:chimpanzees_intercept})
    d = load(Val(:chimpanzees))
    d.treatment = string.(1 .+ d.prosoc_left .+ 2 .* d.condition)
    d.actor     = string.(d.actor)
    d.block     = string.(d.block)
    return ("pulled_left | trials(1) ~ 1", d)
end

"""
name: Chimpanzees — Actor-Indexed Intercepts and Treatment Effects
source: https://github.com/rmcelreath/rethinking/blob/master/data/chimpanzees.csv
example: chimpanzees
dataset: chimpanzees
chapter: SR2 Ch 11
formula: "bf(pulled_left | trials(1) ~ a + b, a ~ 0 + actor, b ~ 0 + treatment, nl = TRUE)"
----

Actor-indexed intercepts plus treatment effects; no pooling across actors.
"""
function examples(::Val{:chimpanzees_actors})
    d = load(Val(:chimpanzees))
    d.treatment = string.(1 .+ d.prosoc_left .+ 2 .* d.condition)
    d.actor     = string.(d.actor)
    d.block     = string.(d.block)
    return ("bf(pulled_left | trials(1) ~ a + b, a ~ 0 + actor, b ~ 0 + treatment, nl = TRUE)", d)
end

"""
name: Chimpanzees — Multilevel Partial Pooling
source: https://github.com/rmcelreath/rethinking/blob/master/data/chimpanzees.csv
example: chimpanzees
dataset: chimpanzees
chapter: SR2 Ch 13
formula: "bf(pulled_left | trials(1) ~ a + b, a ~ 1 + (1 | actor) + (1 | block), b ~ 0 + treatment, nl = TRUE)"
----

Multilevel model with partial pooling across actors and blocks. Adaptive regularization
via hierarchical priors on actor and block intercepts.
"""
function examples(::Val{:chimpanzees_multilevel})
    d = load(Val(:chimpanzees))
    d.treatment = string.(1 .+ d.prosoc_left .+ 2 .* d.condition)
    d.actor     = string.(d.actor)
    d.block     = string.(d.block)
    return ("bf(pulled_left | trials(1) ~ a + b, a ~ 1 + (1 | actor) + (1 | block), b ~ 0 + treatment, nl = TRUE)", d)
end

"""
name: Chimpanzees — Varying Slopes by Treatment
source: https://github.com/rmcelreath/rethinking/blob/master/data/chimpanzees.csv
example: chimpanzees
dataset: chimpanzees
chapter: SR2 Ch 14
formula: "pulled_left | trials(1) ~ 0 + treatment + (0 + treatment | actor) + (0 + treatment | block)"
----

Varying slopes; each actor and block has its own treatment-effect vector.
"""
function examples(::Val{:chimpanzees_slopes})
    d = load(Val(:chimpanzees))
    d.treatment = string.(1 .+ d.prosoc_left .+ 2 .* d.condition)
    d.actor     = string.(d.actor)
    d.block     = string.(d.block)
    return ("pulled_left | trials(1) ~ 0 + treatment + (0 + treatment | actor) + (0 + treatment | block)", d)
end

"""
name: :ucbadmit — UC Berkeley Graduate Admissions
source: https://github.com/rmcelreath/rethinking/blob/master/data/UCBadmit.csv
chapter: SR2 Ch 11
----

Graduate admissions by department and gender (1973); 12 rows: `dept` (A–F),
`applicant.gender` (male/female), `admit`, `reject`, `applications`.
"""
load(::Val{:ucbadmit}) =
    CSV.read(Downloads.download(RETHINKING_URL * "UCBadmit.csv"), DataFrame)

"""
name: UC Berkeley Admissions — Gender Gap
source: https://github.com/rmcelreath/rethinking/blob/master/data/UCBadmit.csv
example: ucbadmit
dataset: ucbadmit
chapter: SR2 Ch 11
formula: "admit | trials(applications) ~ 0 + gid"
----

`gid` = gender index; apparent gender gap in admission rates before conditioning on
department.
"""
function examples(::Val{:ucbadmit_gender})
    d = load(Val(:ucbadmit))
    d.gid = string.(d[!, "applicant.gender"])
    return ("admit | trials(applications) ~ 0 + gid", d)
end

"""
name: UC Berkeley Admissions — Gender + Department
source: https://github.com/rmcelreath/rethinking/blob/master/data/UCBadmit.csv
example: ucbadmit
dataset: ucbadmit
chapter: SR2 Ch 11
formula: "bf(admit | trials(applications) ~ a + d, a ~ 0 + gid, d ~ 0 + dept, nl = TRUE)"
----

Gender and department effects; gender gap disappears after conditioning on department
(Simpson's paradox).
"""
function examples(::Val{:ucbadmit_dept})
    d = load(Val(:ucbadmit))
    d.gid = string.(d[!, "applicant.gender"])
    return ("bf(admit | trials(applications) ~ a + d, a ~ 0 + gid, d ~ 0 + dept, nl = TRUE)", d)
end

"""
name: UC Berkeley Admissions — Beta-Binomial
source: https://github.com/rmcelreath/rethinking/blob/master/data/UCBadmit.csv
example: ucbadmit
dataset: ucbadmit
chapter: SR2 Ch 12
formula: "admit | vint(applications) ~ 0 + gid"
----

Custom beta-binomial family from Ch 12; `vint` passes integer auxiliary data
(applications count).
"""
function examples(::Val{:ucbadmit_beta_binomial})
    d = load(Val(:ucbadmit))
    d.gid = string.(d[!, "applicant.gender"])
    return ("admit | vint(applications) ~ 0 + gid", d)
end

"""
name: :kline — Kline Island Tool Counts
source: https://github.com/rmcelreath/rethinking/blob/master/data/Kline.csv
chapter: SR2 Ch 11–12
----

Tool counts for 10 Pacific island societies: `culture`, `population`, `contact`
(high/low), `total.tools`, `mean.TU`.
"""
load(::Val{:kline})  = CSV.read(Downloads.download(RETHINKING_URL * "Kline.csv"),  DataFrame)

"""
name: :kline2 — Kline Island Tool Counts with Geography
source: https://github.com/rmcelreath/rethinking/blob/master/data/Kline2.csv
chapter: SR2 Ch 14
----

Kline data extended with `lat` (latitude) and `lon2` (longitude on [0, 2π]);
10 Pacific island societies. `lat_adj = lat - mean(lat)`.
"""
load(::Val{:kline2}) = CSV.read(Downloads.download(RETHINKING_URL * "Kline2.csv"), DataFrame)

"""
name: Kline Island — Intercept-Only Poisson
source: https://github.com/rmcelreath/rethinking/blob/master/data/Kline.csv
example: kline
dataset: kline
chapter: SR2 Ch 11–12
formula: "total_tools ~ 1"
----

Intercept-only Poisson (log link); baseline tool count model.
"""
function examples(::Val{:kline_intercept})
    d = load(Val(:kline))
    d.log_pop     = log.(d.population)
    d.log_pop_std = (d.log_pop .- mean(d.log_pop)) ./ std(d.log_pop)
    d.cid         = string.(d.contact)
    return ("total_tools ~ 1", d)
end

"""
name: Kline Island — Contact-Indexed Intercepts and Log-Population Slopes
source: https://github.com/rmcelreath/rethinking/blob/master/data/Kline.csv
example: kline
dataset: kline
chapter: SR2 Ch 11–12
formula: "bf(total_tools ~ a + b * log_pop_std, a + b ~ 0 + cid, nl = TRUE)"
----

Contact-index intercepts and slopes on log-population. `cid` = contact level (high/low).
"""
function examples(::Val{:kline_contact})
    d = load(Val(:kline))
    d.log_pop     = log.(d.population)
    d.log_pop_std = (d.log_pop .- mean(d.log_pop)) ./ std(d.log_pop)
    d.cid         = string.(d.contact)
    return ("bf(total_tools ~ a + b * log_pop_std, a + b ~ 0 + cid, nl = TRUE)", d)
end

"""
name: Kline Island — Scientific Power-Law Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/Kline.csv
example: kline
dataset: kline
chapter: SR2 Ch 11–12
formula: "bf(total_tools ~ exp(a) * population^b / g, a + b ~ 0 + cid, g ~ 1, nl = TRUE)"
----

Scientific power-law model with identity link; `a`, `b` vary by contact level `cid`.
"""
function examples(::Val{:kline_power})
    d = load(Val(:kline))
    d.log_pop     = log.(d.population)
    d.log_pop_std = (d.log_pop .- mean(d.log_pop)) ./ std(d.log_pop)
    d.cid         = string.(d.contact)
    return ("bf(total_tools ~ exp(a) * population^b / g, a + b ~ 0 + cid, g ~ 1, nl = TRUE)", d)
end

"""
name: Kline Island — Gaussian Process over Geography
source: https://github.com/rmcelreath/rethinking/blob/master/data/Kline2.csv
example: kline2
dataset: kline2
chapter: SR2 Ch 14
formula: "bf(total_tools ~ exp(a) * population^b / g, a ~ 1 + gp(lat_adj, lon2_adj, scale=FALSE), b + g ~ 1, nl = TRUE)"
----

Power-law model where `a` has a Gaussian process prior over geographic coordinates
(latitude and longitude). Requires Kline2 dataset.
"""
function examples(::Val{:kline2})
    d = load(Val(:kline2))
    d.log_pop     = log.(d.population)
    d.log_pop_std = (d.log_pop .- mean(d.log_pop)) ./ std(d.log_pop)
    d.cid         = string.(d.contact)
    d.lat_adj     = d.lat .- mean(d.lat)
    return ("bf(total_tools ~ exp(a) * population^b / g, a ~ 1 + gp(lat_adj, lon2_adj, scale=FALSE), b + g ~ 1, nl = TRUE)", d)
end

"""
name: :trolley — Trolley Problem Moral Intuitions
source: https://github.com/rmcelreath/rethinking/blob/master/data/Trolley.csv
chapter: SR2 Ch 12
----

Trolley-problem moral intuitions (McElreath & Turpin 2022); 9,930 rows: `response`
(1–7), `action` (0/1), `intention` (0/1), `contact` (0/1), `edu` (8-level ordered
education), `age`, `male`, `id`, `story`.
"""
load(::Val{:trolley}) =
    CSV.read(Downloads.download(RETHINKING_URL * "Trolley.csv"), DataFrame)

"""
name: Trolley — Intercept Only (6 Thresholds)
source: https://github.com/rmcelreath/rethinking/blob/master/data/Trolley.csv
example: trolley
dataset: trolley
chapter: SR2 Ch 12
formula: "response ~ 1"
----

Intercept-only cumulative ordinal model; 6 thresholds estimated for the 7-point
response scale.
"""
function examples(::Val{:trolley_intercept})
    d = load(Val(:trolley))
    edu_order = ["Elementary School", "Middle School", "Some High School",
                 "High School Graduate", "Some College", "Bachelor's Degree",
                 "Master's Degree", "Graduate Degree"]
    edu_map   = Dict(e => i for (i, e) in enumerate(edu_order))
    d.edu_new = [get(edu_map, e, missing) for e in d.edu]
    return ("response ~ 1", d)
end

"""
name: Trolley — Action, Contact, and Intention Effects
source: https://github.com/rmcelreath/rethinking/blob/master/data/Trolley.csv
example: trolley
dataset: trolley
chapter: SR2 Ch 12
formula: "response ~ 1 + action + contact + intention + intention:action + intention:contact"
----

Action, contact, and intention effects (with interactions) on moral acceptability ratings.
"""
function examples(::Val{:trolley_effects})
    d = load(Val(:trolley))
    edu_order = ["Elementary School", "Middle School", "Some High School",
                 "High School Graduate", "Some College", "Bachelor's Degree",
                 "Master's Degree", "Graduate Degree"]
    edu_map   = Dict(e => i for (i, e) in enumerate(edu_order))
    d.edu_new = [get(edu_map, e, missing) for e in d.edu]
    return ("response ~ 1 + action + contact + intention + intention:action + intention:contact", d)
end

"""
name: Trolley — Monotonic Education Effect
source: https://github.com/rmcelreath/rethinking/blob/master/data/Trolley.csv
example: trolley
dataset: trolley
chapter: SR2 Ch 12
formula: "response ~ 1 + action + contact + intention + mo(edu_new)"
----

Monotonic effect of ordered education; `edu_new` = integer-coded `edu` (1–8).
`mo()` constrains the effect to be monotonically increasing.
"""
function examples(::Val{:trolley_edu})
    d = load(Val(:trolley))
    edu_order = ["Elementary School", "Middle School", "Some High School",
                 "High School Graduate", "Some College", "Bachelor's Degree",
                 "Master's Degree", "Graduate Degree"]
    edu_map   = Dict(e => i for (i, e) in enumerate(edu_order))
    d.edu_new = [get(edu_map, e, missing) for e in d.edu]
    return ("response ~ 1 + action + contact + intention + mo(edu_new)", d)
end

"""
name: :reedfrogs — Reed Frog Tadpole Survival
source: https://github.com/rmcelreath/rethinking/blob/master/data/reedfrogs.csv
chapter: SR2 Ch 12–13
----

Tadpole tank survival experiment; 48 tanks: `density` (initial tadpoles), `pred`
(predation: no/pred), `size` (small/big), `surv` (survivors), `propsurv` (proportion
surviving).
"""
load(::Val{:reedfrogs}) =
    CSV.read(Downloads.download(RETHINKING_URL * "reedfrogs.csv"), DataFrame)

"""
name: Reed Frogs — No-Pooling Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/reedfrogs.csv
example: reedfrogs
dataset: reedfrogs
chapter: SR2 Ch 12–13
formula: "surv | trials(density) ~ 0 + factor(tank)"
----

No-pooling model; one parameter per tank. Each tank gets its own intercept with no
sharing of information.
"""
function examples(::Val{:reedfrogs_nopooling})
    d = load(Val(:reedfrogs))
    d.tank = string.(1:nrow(d))
    return ("surv | trials(density) ~ 0 + factor(tank)", d)
end

"""
name: Reed Frogs — Partial Pooling via Multilevel Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/reedfrogs.csv
example: reedfrogs
dataset: reedfrogs
chapter: SR2 Ch 12–13
formula: "surv | trials(density) ~ 1 + (1 | tank)"
----

Partial pooling; adaptive regularization via multilevel model. Tank-level intercepts
share a common hyperprior.
"""
function examples(::Val{:reedfrogs_partial})
    d = load(Val(:reedfrogs))
    d.tank = string.(1:nrow(d))
    return ("surv | trials(density) ~ 1 + (1 | tank)", d)
end

"""
name: :cafe — Synthetic Café Visit Times
source: synthetic
chapter: SR2 Ch 14
----

Synthetic data: 20 cafés, 10 morning and 10 afternoon visits each. `afternoon` (0/1)
effect on wait time; both intercept and slope vary by café. Correlated intercept/slope
with rho = -0.7.
"""
function load(::Val{:cafe})
    rng     = MersenneTwister(14)
    a       = 3.5; b = -1.0; sigma_a = 1.0; sigma_b = 0.5; rho = -0.7
    n_cafes = 20
    rows    = NamedTuple[]
    for cafe in 1:n_cafes
        # Correlated intercept/slope via conditional simulation
        z_a = randn(rng); z_b = randn(rng)
        a_c = a + sigma_a * z_a
        b_c = b + sigma_b * (rho * z_a + sqrt(1 - rho^2) * z_b)
        for am in 0:1, _ in 1:10
            push!(rows, (; cafe = string(cafe), afternoon = am,
                           wait = max(0.0, a_c + b_c * am + 0.5 * randn(rng))))
        end
    end
    return DataFrame(rows)
end

"""
name: Café Visit Times — Varying Slopes
source: synthetic
example: cafe
dataset: cafe
chapter: SR2 Ch 14
formula: "wait ~ 1 + afternoon + (1 + afternoon | cafe)"
----

Varying intercepts AND slopes; café-level covariance between them. The afternoon
effect on wait time differs across cafés.
"""
function examples(::Val{:cafe})
    d = load(Val(:cafe))
    return ("wait ~ 1 + afternoon + (1 + afternoon | cafe)", d)
end

"""
name: :primates301 — Primate Life History and Social Learning
source: https://github.com/rmcelreath/rethinking/blob/master/data/Primates301.csv
chapter: SR2 Ch 14–15
----

Life history and social learning in 301 primate species. Columns include: `brain`
(brain size), `body` (body mass), `group_size`, `social_learning`, `research_effort`,
`name` (species name).
"""
load(::Val{:primates301}) =
    CSV.read(Downloads.download(RETHINKING_URL * "Primates301.csv"), DataFrame)

"""
name: Primates — OLS Without Phylogeny
source: https://github.com/rmcelreath/rethinking/blob/master/data/Primates301.csv
example: primates301
dataset: primates301
chapter: SR2 Ch 14–15
formula: "b ~ 1 + m + g"
----

Brain size ~ body mass + group size; phylogeny ignored. All variables log-standardized.
"""
function examples(::Val{:primates301_ols})
    d  = dropmissing(load(Val(:primates301)), [:brain, :body, :group_size])
    d.b = (log.(d.brain)      .- mean(log.(d.brain)))      ./ std(log.(d.brain))
    d.m = (log.(d.body)       .- mean(log.(d.body)))        ./ std(log.(d.body))
    d.g = (log.(d.group_size) .- mean(log.(d.group_size))) ./ std(log.(d.group_size))
    return ("b ~ 1 + m + g", d)
end

"""
name: Primates — Phylogenetic Regression
source: https://github.com/rmcelreath/rethinking/blob/master/data/Primates301.csv
example: primates301
dataset: primates301
chapter: SR2 Ch 14–15
formula: "b ~ 1 + m + g + fcor(R)"
----

Same as OLS model plus phylogenetic correlation matrix `R` passed via `data2`.
`R` is built from a phylogenetic tree, e.g. via `ape::vcv.phylo()`.
"""
function examples(::Val{:primates301_phylo})
    d  = dropmissing(load(Val(:primates301)), [:brain, :body, :group_size])
    d.b = (log.(d.brain)      .- mean(log.(d.brain)))      ./ std(log.(d.brain))
    d.m = (log.(d.body)       .- mean(log.(d.body)))        ./ std(log.(d.body))
    d.g = (log.(d.group_size) .- mean(log.(d.group_size))) ./ std(log.(d.group_size))
    return ("b ~ 1 + m + g + fcor(R)", d)   # R = phylogenetic cov matrix (from ape::vcv.phylo)
end

"""
name: :moralizing_gods — Seshat Database of Historical Polities
source: https://github.com/rmcelreath/rethinking/blob/master/data/Moralizing_gods.csv
chapter: SR2 Ch 15
----

Seshat database of historical polities: presence of moralizing gods, writing, and
social complexity across centuries. Key columns: `polity`, `year`, `moralizing_gods`
(0/1/NA), `writing` (0/1/NA), `social_scale` (log10 of social scale).
"""
load(::Val{:moralizing_gods}) =
    CSV.read(Downloads.download(RETHINKING_URL * "Moralizing_gods.csv"), DataFrame)

"""
name: Moralizing Gods — Missing-Data Handling
source: https://github.com/rmcelreath/rethinking/blob/master/data/Moralizing_gods.csv
example: moralizing_gods
dataset: moralizing_gods
chapter: SR2 Ch 15
formula: "moralizing_gods | trials(1) ~ 1 + writing"
----

Do moralizing gods predict / follow the emergence of writing? Rows with missing
`moralizing_gods` or `writing` are dropped.
"""
function examples(::Val{:moralizing_gods})
    d = dropmissing(load(Val(:moralizing_gods)), [:moralizing_gods, :writing])
    return ("moralizing_gods | trials(1) ~ 1 + writing", d)
end

"""
name: :panda_nuts — Chimpanzee Nut-Cracking Proficiency
source: https://github.com/rmcelreath/rethinking/blob/master/data/Panda_nuts.csv
chapter: SR2 Ch 16
----

Chimpanzee nut-cracking proficiency over development. Columns: `name`, `site`,
`group`, `age` (years), `rounds`, `n.pulls` (attempts), `success` (successes),
`seconds` (observation time).
"""
load(::Val{:panda_nuts}) =
    CSV.read(Downloads.download(RETHINKING_URL * "Panda_nuts.csv"), DataFrame)

"""
name: Panda Nuts — Nonlinear Learning-Curve Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/Panda_nuts.csv
example: panda_nuts
dataset: panda_nuts
chapter: SR2 Ch 16
formula: "bf(n_panda ~ seconds * phi * (1 - exp(-k * age_s))^theta, phi + k + theta ~ 1, nl = TRUE)"
----

`poisson(link = identity)`; `n_panda = success`; `phi` = asymptotic rate, `k` = growth
rate, `theta` = shape; `age_s` = standardized age.
"""
function examples(::Val{:panda_nuts})
    d = load(Val(:panda_nuts))
    d.n_panda = d.success
    d.age_s   = (d.age .- mean(d.age)) ./ std(d.age)
    return ("bf(n_panda ~ seconds * phi * (1 - exp(-k * age_s))^theta, phi + k + theta ~ 1, nl = TRUE)", d)
end

"""
name: :lynx_hare — Canadian Lynx and Snowshoe Hare Pelts
source: https://github.com/rmcelreath/rethinking/blob/master/data/Lynx_Hare.csv
chapter: SR2 Ch 16
----

Canadian lynx and snowshoe hare pelts (Hudson's Bay Co.); 21 rows (1900–1920):
`Year`, `Hare` (thousands of pelts), `Lynx` (thousands of pelts).
"""
load(::Val{:lynx_hare}) =
    CSV.read(Downloads.download(RETHINKING_URL * "Lynx_Hare.csv"), DataFrame)

"""
name: Lynx and Hare — AR(1) Autoregressive Model
source: https://github.com/rmcelreath/rethinking/blob/master/data/Lynx_Hare.csv
example: lynx_hare
dataset: lynx_hare
chapter: SR2 Ch 16
formula: "Hare ~ 1 + ar(time = Year, p = 1)"
----

AR(1) autoregressive model for hare population using the full dataset (missings present
in lag variables).
"""
function examples(::Val{:lynx_hare_ar})
    d        = load(Val(:lynx_hare))
    d.Hare_1 = [missing; d.Hare[1:end-1]]   # lag-1 of hare
    d.Lynx_1 = [missing; d.Lynx[1:end-1]]   # lag-1 of lynx
    return ("Hare ~ 1 + ar(time = Year, p = 1)", d)
end

"""
name: Lynx and Hare — Lag-1 Regression
source: https://github.com/rmcelreath/rethinking/blob/master/data/Lynx_Hare.csv
example: lynx_hare
dataset: lynx_hare
chapter: SR2 Ch 16
formula: "Hare ~ 1 + Hare_1"
----

Equivalent lag-1 regression; `Hare_1` = lag of `Hare`. Uses dropmissing dataset
(d2, n=20).
"""
function examples(::Val{:lynx_hare_lag})
    d        = load(Val(:lynx_hare))
    d.Hare_1 = [missing; d.Hare[1:end-1]]
    d.Lynx_1 = [missing; d.Lynx[1:end-1]]
    d2       = dropmissing(d)
    return ("Hare ~ 1 + Hare_1", d2)
end

"""
name: Lynx and Hare — Missing-Data Model for Lagged Hare
source: https://github.com/rmcelreath/rethinking/blob/master/data/Lynx_Hare.csv
example: lynx_hare
dataset: lynx_hare
chapter: SR2 Ch 16
formula: "bf(Hare ~ 1 + mi(Hare_1)) + bf(Hare_1 | mi() ~ 1) + set_rescor(FALSE)"
----

Treating lagged `Hare` as partially observed/missing. Uses the full dataset with
first-row `Hare_1` as missing.
"""
function examples(::Val{:lynx_hare_mi})
    d        = load(Val(:lynx_hare))
    d.Hare_1 = [missing; d.Hare[1:end-1]]
    d.Lynx_1 = [missing; d.Lynx[1:end-1]]
    return ("bf(Hare ~ 1 + mi(Hare_1)) + bf(Hare_1 | mi() ~ 1) + set_rescor(FALSE)", d)
end

"""
name: Lynx and Hare — Bivariate VAR(1)
source: https://github.com/rmcelreath/rethinking/blob/master/data/Lynx_Hare.csv
example: lynx_hare
dataset: lynx_hare
chapter: SR2 Ch 16
formula: "bf(Hare ~ 0 + Intercept + Hare_1 + Lynx_1) + bf(Lynx ~ 0 + Intercept + Lynx_1 + Hare_1) + set_rescor(FALSE)"
----

Bivariate VAR(1); lognormal family. `Lynx` feeds back on `Hare`. Uses dropmissing
dataset (d2, n=20).
"""
function examples(::Val{:lynx_hare_var})
    d        = load(Val(:lynx_hare))
    d.Hare_1 = [missing; d.Hare[1:end-1]]
    d.Lynx_1 = [missing; d.Lynx[1:end-1]]
    d2       = dropmissing(d)
    return ("bf(Hare ~ 0 + Intercept + Hare_1 + Lynx_1) + bf(Lynx ~ 0 + Intercept + Lynx_1 + Hare_1) + set_rescor(FALSE)", d2)
end

const BTDATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv"

"""
name: :btdata — Blue Tit Morphology Data (McElreath)
source: https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv
chapter: SR2 Ch 17
----

BTdata from the MCMCglmm package (also covered in `brms.jl`) — 828 obs of blue tits;
bivariate outcome: `tarsus` (length) + `back` (coloration). Columns: `tarsus`, `back`,
`sex`, `hatchdate`, `fosternest`, `dam`.
"""
load(::Val{:btdata}) = CSV.read(Downloads.download(BTDATA_URL), DataFrame)

"""
name: Blue Tit Morphology — Intercept Only
source: https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv
example: btdata
dataset: btdata
chapter: SR2 Ch 17
formula: "mvbind(tarsus, back) ~ 1"
----

Intercept-only multivariate model; establishes baseline residual correlation between
tarsus length and back coloration.
"""
function examples(::Val{:mcelreath_btdata_intercept})
    d = load(Val(:btdata))
    return ("mvbind(tarsus, back) ~ 1", d)
end

"""
name: Blue Tit Morphology — Full Model
source: https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv
example: btdata
dataset: btdata
chapter: SR2 Ch 17
formula: "mvbind(tarsus, back) ~ sex + hatchdate + (1 | p | fosternest) + (1 | q | dam)"
----

Full model; `p`/`q` labels allow correlated random effects across responses. Sex and
hatch date as fixed effects; foster nest and dam as random effects.
"""
function examples(::Val{:mcelreath_btdata_full})
    d = load(Val(:btdata))
    return ("mvbind(tarsus, back) ~ sex + hatchdate + (1 | p | fosternest) + (1 | q | dam)", d)
end

"""
name: Blue Tit Morphology — Sex × Hatchdate Interaction
source: https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv
example: btdata
dataset: btdata
chapter: SR2 Ch 17
formula: "mvbind(tarsus, back) ~ sex * hatchdate + (1 | p | fosternest) + (1 | q | dam)"
----

Adds `sex` × `hatchdate` interaction to the full model.
"""
function examples(::Val{:mcelreath_btdata_interaction})
    d = load(Val(:btdata))
    return ("mvbind(tarsus, back) ~ sex * hatchdate + (1 | p | fosternest) + (1 | q | dam)", d)
end
