using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: "Doing Bayesian Data Analysis in brms and the tidyverse"
# A. Solomon Kurz (brms translation of John Kruschke's DBDA2, 2nd ed.)
# Source: https://github.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse
# Data:   https://github.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/tree/master/data.R
#
# Chapters without brm() calls (conceptual/prior/power discussions): 1–7, 10–15.

const DBDA_URL = "https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/"

"""
name: z15n50 — 50 Bernoulli Trials, 15 Successes
source: synthetic
chapter: DBDA2 Ch 8
----

50 Bernoulli trials with 15 heads (successes). Single column `y` (0/1 outcome).
"""
load(::Val{:z15n50}) = CSV.read(Downloads.download(DBDA_URL * "z15N50.csv"), DataFrame)

"""
name: z6n8z2n7 — Two-Mint Coin Flip Data
source: synthetic
chapter: DBDA2 Ch 8
----

Two mints: mint 1 gives 6/8 heads, mint 2 gives 2/7. Columns: `y` (0/1 outcome), `s` (mint ID).
"""
load(::Val{:z6n8z2n7}) = CSV.read(Downloads.download(DBDA_URL * "z6N8z2N7.csv"), DataFrame)

"""
name: Coin-Flip — Bernoulli Intercept
source: synthetic
example: z15n50
dataset: z15n50
chapter: DBDA2 Ch 8
formula: "y ~ 1"
----

Intercept-only Bernoulli model; identity link estimates `p` directly from 50 trials with 15 successes.
"""
function examples(::Val{:z15n50})
    d = load(Val(:z15n50))
    return ("y ~ 1", d)
end

"""
name: Coin-Flip — Two-Mint Separate Intercepts
source: synthetic
example: z6n8z2n7
dataset: z6n8z2n7
chapter: DBDA2 Ch 8
formula: "y ~ 0 + s"
----

Separate Bernoulli intercept per mint; no intercept centering. Mint 1: 6/8 heads, mint 2: 2/7.
"""
function examples(::Val{:z6n8z2n7})
    d = load(Val(:z6n8z2n7))
    return ("y ~ 0 + s", d)
end

"""
name: therapeutic_touch — Therapeutic Touch Trial Data
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/TherapeuticTouchData.csv
chapter: DBDA2 Ch 9
----

280 therapeutic-touch trials (Benor et al.); 28 healers (10 trials each). Columns: `y` (1=correct, 0=incorrect), `s` (healer ID 1–28).
"""
load(::Val{:therapeutic_touch}) =
    CSV.read(Downloads.download(DBDA_URL * "TherapeuticTouchData.csv"), DataFrame)

"""
name: batting_average — MLB Batting Average Data
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/BattingAverage.csv
chapter: DBDA2 Ch 9
----

948 MLB player-season records. Columns: `Player`, `PriPos` (primary position), `Hits`, `AtBats`, and other fields.
"""
load(::Val{:batting_average}) =
    CSV.read(Downloads.download(DBDA_URL * "BattingAverage.csv"), DataFrame)

"""
name: Hierarchical Bernoulli — Therapeutic Touch
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/TherapeuticTouchData.csv
example: therapeutic_touch
dataset: therapeutic_touch
chapter: DBDA2 Ch 9
formula: "y ~ 1 + (1 | s)"
----

Hierarchical Bernoulli model with partial pooling across 28 healers; logit link.
"""
function examples(::Val{:therapeutic_touch})
    d = load(Val(:therapeutic_touch))
    return ("y ~ 1 + (1 | s)", d)
end

"""
name: Hierarchical Binomial — MLB Batting Average
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/BattingAverage.csv
example: batting_average
dataset: batting_average
chapter: DBDA2 Ch 9
formula: "Hits | trials(AtBats) ~ 1 + (1 | PriPos) + (1 | PriPos:Player)"
----

Hierarchical binomial model; players nested within positions; estimates batting average with partial pooling.
"""
function examples(::Val{:batting_average})
    d = load(Val(:batting_average))
    return ("Hits | trials(AtBats) ~ 1 + (1 | PriPos) + (1 | PriPos:Player)", d)
end

"""
name: recall — Binomial Memory-Recall Experiment
source: synthetic
chapter: DBDA2 Ch 12
----

6 conditions × 20 subjects, each recalling items from a list of 20. Data generated synthetically
via MersenneTwister(12); columns: `condition` (cond1–cond6), `n_recalled`.
"""
function load(::Val{:recall})
    rng    = MersenneTwister(12)
    conds  = ["cond" * string(i) for i in 1:6]
    p_true = [0.4, 0.5, 0.6, 0.55, 0.45, 0.65]
    rows   = NamedTuple[]
    for (cond, p) in zip(conds, p_true), _ in 1:20
        n = round(Int, 20 * clamp(p + 0.1 * randn(rng), 0.05, 0.95))
        push!(rows, (; condition = cond, n_recalled = n))
    end
    return DataFrame(rows)
end

"""
name: Binomial Recall — Separate Probability per Condition
source: synthetic
example: recall
dataset: recall
chapter: DBDA2 Ch 12
formula: "n_recalled | trials(20) ~ 0 + condition"
----

Separate binomial recall probability per condition; no intercept centering. Six conditions with distinct true probabilities.
"""
function examples(::Val{:recall_conditions})
    d = load(Val(:recall))
    return ("n_recalled | trials(20) ~ 0 + condition", d)
end

"""
name: Binomial Recall — Pooled Single Intercept
source: synthetic
example: recall
dataset: recall
chapter: DBDA2 Ch 12
formula: "n_recalled | trials(20) ~ 1"
----

Pooled single intercept model for memory recall; baseline / comparison model ignoring condition differences.
"""
function examples(::Val{:recall_pooled})
    d = load(Val(:recall))
    return ("n_recalled | trials(20) ~ 1", d)
end

"""
name: two_group_iq — Two-Group IQ Score Data
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/TwoGroupIQ.csv
chapter: DBDA2 Ch 16
----

IQ scores for two treatment groups. Columns: `Group` (Smart Drug / Placebo), `Score` (IQ score).
"""
load(::Val{:two_group_iq}) =
    CSV.read(Downloads.download(DBDA_URL * "TwoGroupIQ.csv"), DataFrame)

"""
name: Two-Group IQ — Single-Group Baseline
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/TwoGroupIQ.csv
example: two_group_iq
dataset: two_group_iq
chapter: DBDA2 Ch 16
formula: "Score ~ 1"
----

Single-group Gaussian model on IQ scores; prior-predictive / baseline before conditioning on group.
"""
function examples(::Val{:two_group_iq_baseline})
    d = load(Val(:two_group_iq))
    return ("Score ~ 1", d)
end

"""
name: Two-Group IQ — Heteroscedastic Groups
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/TwoGroupIQ.csv
example: two_group_iq
dataset: two_group_iq
chapter: DBDA2 Ch 16
formula: "bf(Score ~ 0 + Group, sigma ~ 0 + Group)"
----

Two groups with group-specific mean AND sigma; student-t family for robustness against outliers.
"""
function examples(::Val{:two_group_iq_hetero})
    d = load(Val(:two_group_iq))
    return ("bf(Score ~ 0 + Group, sigma ~ 0 + Group)", d)
end

"""
name: calcium — Calcium Supplementation RCT
source: https://jse.amstat.org/datasets/calcium.dat.txt
chapter: DBDA2 Ch 16
----

Decrease in blood pressure after calcium supplementation (Lyle et al. 1987). Columns: `begin`, `end`, `decrease` (mm Hg; begin − end), `treatment` (Calcium/Placebo).
"""
function load(::Val{:calcium})
    url = "https://jse.amstat.org/datasets/calcium.dat.txt"
    cols = ["begin", "end", "decrease", "treatment"]
    CSV.read(Downloads.download(url), DataFrame;
             header = cols, skipto = 2, delim = ' ', ignorerepeated = true)
end

"""
name: Calcium RCT — Separate Group Means
source: https://jse.amstat.org/datasets/calcium.dat.txt
example: calcium
dataset: calcium
chapter: DBDA2 Ch 16
formula: "decrease ~ 0 + treatment"
----

Separate group means for calcium vs. placebo; Gaussian (or student-t for robustness). Outcome is blood-pressure decrease in mm Hg.
"""
function examples(::Val{:calcium})
    d = load(Val(:calcium))
    return ("decrease ~ 0 + treatment", d)
end

"""
name: htwt30 — Height and Weight Data (n=30)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HtWtData30.csv
chapter: DBDA2 Ch 17
----

Height (cm) and weight (kg) for 30 adults. Columns: `height`, `weight`, `male`.
"""
load(::Val{:htwt30})  = CSV.read(Downloads.download(DBDA_URL * "HtWtData30.csv"),  DataFrame)

"""
name: htwt300 — Height and Weight Data (n=300)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HtWtData300.csv
chapter: DBDA2 Ch 17
----

Height (cm) and weight (kg) for 300 adults. Columns: `height`, `weight`, `male`.
"""
load(::Val{:htwt300}) = CSV.read(Downloads.download(DBDA_URL * "HtWtData300.csv"), DataFrame)

"""
name: htwt — Height and Weight Data (n=300, alias)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HtWtData300.csv
chapter: DBDA2 Ch 17
----

Alias for htwt300. Height (cm) and weight (kg) for 300 adults. Columns: `height`, `weight`, `male`.
"""
load(::Val{:htwt})    = load(Val(:htwt300))

"""
name: hier_linreg — Hierarchical Linear Regression Data
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HierLinRegressData.csv
chapter: DBDA2 Ch 17
----

Multiple subjects each with several observations. Columns: `Subj` (subject ID), `X` (metric predictor), `Y` (metric outcome).
"""
load(::Val{:hier_linreg}) =
    CSV.read(Downloads.download(DBDA_URL * "HierLinRegressData.csv"), DataFrame)

"""
name: income_famsize — Median Household Income by Family Size and State
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/IncomeFamszState3yr.csv
chapter: DBDA2 Ch 17
----

Median household income by family size and U.S. state; three years stacked. Columns: `State`, `family_size`, `median_income`, `se` (standard error).
"""
load(::Val{:income_famsize}) =
    CSV.read(Downloads.download(DBDA_URL * "IncomeFamszState3yr.csv"), DataFrame)

"""
name: Simple Linear Regression — Height Predicts Weight
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HtWtData300.csv
example: htwt
dataset: htwt
chapter: DBDA2 Ch 17
formula: "weight_z ~ 1 + height_z"
----

Simple linear regression with z-standardized predictors; student-t family for robustness. Uses n=300 adults.
"""
function examples(::Val{:htwt})
    d = load(Val(:htwt300))
    d.weight_z = (d.weight .- mean(d.weight)) ./ std(d.weight)
    d.height_z = (d.height .- mean(d.height)) ./ std(d.height)
    return ("weight_z ~ 1 + height_z", d)
end

"""
name: Hierarchical Linear Regression — Varying Intercepts and Slopes
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HierLinRegressData.csv
example: hier_linreg
dataset: hier_linreg
chapter: DBDA2 Ch 17
formula: "y_z ~ 1 + x_z + (1 + x_z || Subj)"
----

Hierarchical model with varying intercepts and slopes per subject; `||` = uncorrelated random effects; z-standardized predictors.
"""
function examples(::Val{:hier_linreg})
    d = load(Val(:hier_linreg))
    d.y_z = (d.Y .- mean(d.Y)) ./ std(d.Y)
    d.x_z = (d.X .- mean(d.X)) ./ std(d.X)
    return ("y_z ~ 1 + x_z + (1 + x_z || Subj)", d)
end

"""
name: Income by Family Size — Measurement-Error Model
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/IncomeFamszState3yr.csv
example: income_famsize
dataset: income_famsize
chapter: DBDA2 Ch 17
formula: "median_income_z | se(se_z, sigma = TRUE) ~ 1 + family_size_z + I(family_size_z^2) + (1 + family_size_z + I(family_size_z^2) || State)"
----

Measurement-error model via `| se()`; quadratic trend in family size; state-level random slopes. Accounts for known standard error in the outcome.
"""
function examples(::Val{:income_famsize})
    d = load(Val(:income_famsize))
    d.median_income_z = (d.median_income .- mean(d.median_income)) ./ std(d.median_income)
    d.family_size_z   = (d.family_size   .- mean(d.family_size))   ./ std(d.family_size)
    d.se_z            = d.se ./ std(d.median_income)
    return ("median_income_z | se(se_z, sigma = TRUE) ~ 1 + family_size_z + I(family_size_z^2) + (1 + family_size_z + I(family_size_z^2) || State)", d)
end

"""
name: guber1999 — U.S. State-Level SAT Score Data (Guber 1999)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/Guber1999data.csv
chapter: DBDA2 Ch 18
----

50 U.S. states, 7 variables: `State`, `Sat` (mean SAT score), `Spend` (expenditure per pupil, \$1,000s), `PctSAT` (% students taking SAT), `TeachPay`, `StuTchRatio`, `SalaryScale`, `PupilExp`.
"""
load(::Val{:guber1999}) =
    CSV.read(Downloads.download(DBDA_URL * "Guber1999data.csv"), DataFrame)

"""
name: Guber 1999 — Base Multiple Regression
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/Guber1999data.csv
example: guber1999
dataset: guber1999
chapter: DBDA2 Ch 18
formula: "satt_z ~ 1 + spend_z + prcnt_take_z"
----

Multiple regression of SAT score on spending and percent taking SAT; percent taking is a suppressor variable.
"""
function examples(::Val{:guber1999_base})
    d = load(Val(:guber1999))
    for col in [:Sat, :Spend, :PctSAT]
        d[!, Symbol(lowercase(string(col)) * "_z")] =
            (d[!, col] .- mean(d[!, col])) ./ std(d[!, col])
    end
    d.satt_z       = d.sat_z
    d.spend_z      = d.spend_z
    d.prcnt_take_z = d.pctsat_z
    d.prop_not_take_z = (1.0 .- d.PctSAT ./ 100 .- mean(1.0 .- d.PctSAT ./ 100)) ./
                        std(1.0 .- d.PctSAT ./ 100)
    d.interaction_z = (d.spend_z .* d.prcnt_take_z .- mean(d.spend_z .* d.prcnt_take_z)) ./
                       std(d.spend_z .* d.prcnt_take_z)
    return ("satt_z ~ 1 + spend_z + prcnt_take_z", d)
end

"""
name: Guber 1999 — Complement Proportion Predictor
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/Guber1999data.csv
example: guber1999
dataset: guber1999
chapter: DBDA2 Ch 18
formula: "satt_z ~ 0 + Intercept + spend_z + prcnt_take_z + prop_not_take_z"
----

Adds complementary proportion (prop_not_take_z = 1 - prcnt_take); explicit centering via `0 + Intercept`.
"""
function examples(::Val{:guber1999_complement})
    d = load(Val(:guber1999))
    for col in [:Sat, :Spend, :PctSAT]
        d[!, Symbol(lowercase(string(col)) * "_z")] =
            (d[!, col] .- mean(d[!, col])) ./ std(d[!, col])
    end
    d.satt_z       = d.sat_z
    d.spend_z      = d.spend_z
    d.prcnt_take_z = d.pctsat_z
    d.prop_not_take_z = (1.0 .- d.PctSAT ./ 100 .- mean(1.0 .- d.PctSAT ./ 100)) ./
                        std(1.0 .- d.PctSAT ./ 100)
    d.interaction_z = (d.spend_z .* d.prcnt_take_z .- mean(d.spend_z .* d.prcnt_take_z)) ./
                       std(d.spend_z .* d.prcnt_take_z)
    return ("satt_z ~ 0 + Intercept + spend_z + prcnt_take_z + prop_not_take_z", d)
end

"""
name: Guber 1999 — Interaction Term
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/Guber1999data.csv
example: guber1999
dataset: guber1999
chapter: DBDA2 Ch 18
formula: "satt_z ~ 1 + spend_z + prcnt_take_z + interaction_z"
----

Adds interaction_z = spend_z * prcnt_take_z (standardized product); student-t for robustness.
"""
function examples(::Val{:guber1999_interaction})
    d = load(Val(:guber1999))
    for col in [:Sat, :Spend, :PctSAT]
        d[!, Symbol(lowercase(string(col)) * "_z")] =
            (d[!, col] .- mean(d[!, col])) ./ std(d[!, col])
    end
    d.satt_z       = d.sat_z
    d.spend_z      = d.spend_z
    d.prcnt_take_z = d.pctsat_z
    d.prop_not_take_z = (1.0 .- d.PctSAT ./ 100 .- mean(1.0 .- d.PctSAT ./ 100)) ./
                        std(1.0 .- d.PctSAT ./ 100)
    d.interaction_z = (d.spend_z .* d.prcnt_take_z .- mean(d.spend_z .* d.prcnt_take_z)) ./
                       std(d.spend_z .* d.prcnt_take_z)
    return ("satt_z ~ 1 + spend_z + prcnt_take_z + interaction_z", d)
end

"""
name: fruitfly — Drosophila Longevity Data (Partridge & Farquhar 1981)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/FruitflyDataReduced.csv
chapter: DBDA2 Ch 19
----

125 male Drosophila melanogaster. Columns: `CompanionNumber` (1–5 companion types: Alone, 1 Virgin, 8 Virgin, 1 Pregnant, 8 Pregnant), `Longevity` (days), `Thorax` (mm).
"""
load(::Val{:fruitfly}) =
    CSV.read(Downloads.download(DBDA_URL * "FruitflyDataReduced.csv"), DataFrame)

"""
name: Fruitfly — Hierarchical One-Way ANOVA
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/FruitflyDataReduced.csv
example: fruitfly
dataset: fruitfly
chapter: DBDA2 Ch 19
formula: "Longevity ~ 1 + (1 | CompanionNumber)"
----

ANOVA as hierarchical model; partial pooling across companion types for longevity.
"""
function examples(::Val{:fruitfly_hierarchical})
    d = load(Val(:fruitfly))
    d.thorax_c = d.Thorax .- mean(d.Thorax)
    return ("Longevity ~ 1 + (1 | CompanionNumber)", d)
end

"""
name: Fruitfly — Pooled Baseline
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/FruitflyDataReduced.csv
example: fruitfly
dataset: fruitfly
chapter: DBDA2 Ch 19
formula: "Longevity ~ 1"
----

Pooled / no-group baseline model for longevity; comparison model with no companion-type structure.
"""
function examples(::Val{:fruitfly_pooled})
    d = load(Val(:fruitfly))
    d.thorax_c = d.Thorax .- mean(d.Thorax)
    return ("Longevity ~ 1", d)
end

"""
name: Fruitfly — Robust ANOVA with Heterogeneous Sigma
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/FruitflyDataReduced.csv
example: fruitfly
dataset: fruitfly
chapter: DBDA2 Ch 19
formula: "bf(Longevity ~ 0 + CompanionNumber, sigma ~ 0 + CompanionNumber)"
----

Robust ANOVA with group-specific means AND sigmas; student-t family for outlier robustness.
"""
function examples(::Val{:fruitfly_robust})
    d = load(Val(:fruitfly))
    d.thorax_c = d.Thorax .- mean(d.Thorax)
    return ("bf(Longevity ~ 0 + CompanionNumber, sigma ~ 0 + CompanionNumber)", d)
end

"""
name: Fruitfly — ANCOVA with Thorax Covariate
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/FruitflyDataReduced.csv
example: fruitfly
dataset: fruitfly
chapter: DBDA2 Ch 19
formula: "Longevity ~ 1 + thorax_c + (1 | CompanionNumber)"
----

ANCOVA controlling for thorax length; `thorax_c` = centered Thorax; common slope across companion groups.
"""
function examples(::Val{:fruitfly_ancova})
    d = load(Val(:fruitfly))
    d.thorax_c = d.Thorax .- mean(d.Thorax)
    return ("Longevity ~ 1 + thorax_c + (1 | CompanionNumber)", d)
end

"""
name: Fruitfly — ANHECOVA with Random Thorax Slope
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/FruitflyDataReduced.csv
example: fruitfly
dataset: fruitfly
chapter: DBDA2 Ch 19
formula: "Longevity ~ 1 + thorax_c + (1 + thorax_c | CompanionNumber)"
----

ANHECOVA with random slope for thorax length per companion group; allows group-specific relationship between body size and longevity.
"""
function examples(::Val{:fruitfly_anhecova})
    d = load(Val(:fruitfly))
    d.thorax_c = d.Thorax .- mean(d.Thorax)
    return ("Longevity ~ 1 + thorax_c + (1 + thorax_c | CompanionNumber)", d)
end

"""
name: salary — Fictional Salary Data by Position and Organization
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/Salary.csv
chapter: DBDA2 Ch 20
----

Fictional salary data. Columns: `Salary`, `Pos` (position), `Org` (organization).
"""
load(::Val{:salary})    = CSV.read(Downloads.download(DBDA_URL * "Salary.csv"),          DataFrame)

"""
name: splitplot — Split-Plot Agricultural Experiment
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/SplitPlotAgriData.csv
chapter: DBDA2 Ch 20
----

Split-plot agronomy experiment. Columns: `Yield`, `Till` (tillage method), `Fert` (fertilizer), `Field` (blocking factor).
"""
load(::Val{:splitplot}) = CSV.read(Downloads.download(DBDA_URL * "SplitPlotAgriData.csv"), DataFrame)

"""
name: Salary — Two-Way Random Effects ANOVA
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/Salary.csv
example: salary
dataset: salary
chapter: DBDA2 Ch 20
formula: "Salary ~ 1 + (1 | Pos) + (1 | Org) + (1 | Pos:Org)"
----

Two-way random effects ANOVA with interaction; Gaussian family; partial pooling across positions and organizations.
"""
function examples(::Val{:salary_anova})
    d = load(Val(:salary))
    return ("Salary ~ 1 + (1 | Pos) + (1 | Org) + (1 | Pos:Org)", d)
end

"""
name: Salary — Robust ANOVA with Heterogeneous Residual Variance
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/Salary.csv
example: salary
dataset: salary
chapter: DBDA2 Ch 20
formula: "bf(Salary ~ 1 + (1 | Pos) + (1 | Org) + (1 | Pos:Org), sigma ~ 1 + (1 | Pos:Org))"
----

Robust variant of salary ANOVA with heterogeneous residual variance per cell (Pos:Org); student-t family.
"""
function examples(::Val{:salary_robust})
    d = load(Val(:salary))
    return ("bf(Salary ~ 1 + (1 | Pos) + (1 | Org) + (1 | Pos:Org), sigma ~ 1 + (1 | Pos:Org))", d)
end

"""
name: Split-Plot ANOVA — With Field Blocking
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/SplitPlotAgriData.csv
example: splitplot
dataset: splitplot
chapter: DBDA2 Ch 20
formula: "Yield ~ 1 + (1 | Till) + (1 | Fert) + (1 | Field) + (1 | Till:Fert)"
----

Split-plot ANOVA with tillage × fertilizer interaction plus blocking on `Field`.
"""
function examples(::Val{:splitplot_field})
    d = load(Val(:splitplot))
    return ("Yield ~ 1 + (1 | Till) + (1 | Fert) + (1 | Field) + (1 | Till:Fert)", d)
end

"""
name: Split-Plot ANOVA — Without Field Blocking
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/SplitPlotAgriData.csv
example: splitplot
dataset: splitplot
chapter: DBDA2 Ch 20
formula: "Yield ~ 1 + (1 | Till) + (1 | Fert) + (1 | Till:Fert)"
----

Split-plot ANOVA without explicit field blocking; comparison model to assess the contribution of the field random effect.
"""
function examples(::Val{:splitplot_nofield})
    d = load(Val(:splitplot))
    return ("Yield ~ 1 + (1 | Till) + (1 | Fert) + (1 | Till:Fert)", d)
end

"""
name: htwt110 — Height, Weight and Sex Data (n=110)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HtWtData110.csv
chapter: DBDA2 Ch 21
----

110 adults from Appleton, French & Vanderpump (1996). Columns: `height` (cm), `weight` (kg), `male` (0/1).
"""
load(::Val{:htwt110}) =
    CSV.read(Downloads.download(DBDA_URL * "HtWtData110.csv"), DataFrame)

"""
name: Logistic Regression — Single Predictor (Weight)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HtWtData110.csv
example: htwt110
dataset: htwt110
chapter: DBDA2 Ch 21
formula: "male ~ 1 + weight_z"
----

Logistic regression predicting sex from standardized weight; single metric predictor.
"""
function examples(::Val{:htwt110_single})
    d = load(Val(:htwt110))
    d.weight_z = (d.weight .- mean(d.weight)) ./ std(d.weight)
    d.height_z = (d.height .- mean(d.height)) ./ std(d.height)
    return ("male ~ 1 + weight_z", d)
end

"""
name: Logistic Regression — Two Predictors (Weight and Height)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HtWtData110.csv
example: htwt110
dataset: htwt110
chapter: DBDA2 Ch 21
formula: "male ~ 1 + weight_z + height_z"
----

Logistic regression with two metric predictors; collinear but jointly informative for predicting sex.
"""
function examples(::Val{:htwt110_two_pred})
    d = load(Val(:htwt110))
    d.weight_z = (d.weight .- mean(d.weight)) ./ std(d.weight)
    d.height_z = (d.height .- mean(d.height)) ./ std(d.height)
    return ("male ~ 1 + weight_z + height_z", d)
end

"""
name: Robust Logistic Regression — Mixture with Guessing
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HtWtData110.csv
example: htwt110
dataset: htwt110
chapter: DBDA2 Ch 21
formula: "bf(male ~ a * 0.5 + (1 - a) * 1 / (1 + exp(-(b0 + b1 * weight_z))), a + b0 + b1 ~ 1, nl = TRUE)"
----

Robust nonlinear logistic model; `a` = probability of guessing (mixture weight); identity link; estimated via `nl = TRUE`.
"""
function examples(::Val{:htwt110_robust})
    d = load(Val(:htwt110))
    d.weight_z = (d.weight .- mean(d.weight)) ./ std(d.weight)
    d.height_z = (d.height .- mean(d.height)) ./ std(d.height)
    return ("bf(male ~ a * 0.5 + (1 - a) * 1 / (1 + exp(-(b0 + b1 * weight_z))), a + b0 + b1 ~ 1, nl = TRUE)", d)
end

"""
name: softmax1 — Synthetic Categorical Outcome Data (Set 1)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/SoftmaxRegData1.csv
chapter: DBDA2 Ch 22
----

Synthetic categorical outcome data. Columns: `Y` (category 1–3), `X1` (metric), `X2` (metric).
"""
load(::Val{:softmax1}) = CSV.read(Downloads.download(DBDA_URL * "SoftmaxRegData1.csv"),    DataFrame)

"""
name: softmax2 — Synthetic Categorical Outcome Data (Set 2)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/SoftmaxRegData2.csv
chapter: DBDA2 Ch 22
----

Second synthetic categorical outcome dataset. Columns: `Y` (category 1–3), `X1` (metric), `X2` (metric).
"""
load(::Val{:softmax2}) = CSV.read(Downloads.download(DBDA_URL * "SoftmaxRegData2.csv"),    DataFrame)

"""
name: condlog1 — Synthetic Conditional Logistic Data (Set 1)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/CondLogistRegData1.csv
chapter: DBDA2 Ch 22
----

Synthetic conditional logistic regression data. Columns: `Y` (ordered category), `Y_ord`, `X1` (metric), `X2` (metric).
"""
load(::Val{:condlog1}) = CSV.read(Downloads.download(DBDA_URL * "CondLogistRegData1.csv"), DataFrame)

"""
name: condlog2 — Synthetic Conditional Logistic Data (Set 2)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/CondLogistRegData2.csv
chapter: DBDA2 Ch 22
----

Second synthetic conditional logistic regression dataset. Columns: `Y` (ordered category), `Y_ord`, `X1` (metric), `X2` (metric).
"""
load(::Val{:condlog2}) = CSV.read(Downloads.download(DBDA_URL * "CondLogistRegData2.csv"), DataFrame)

"""
name: softmax — Alias for Softmax Data Set 1
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/SoftmaxRegData1.csv
chapter: DBDA2 Ch 22
----

Alias for softmax1. Synthetic categorical outcome data. Columns: `Y` (category 1–3), `X1` (metric), `X2` (metric).
"""
load(::Val{:softmax})  = load(Val(:softmax1))

"""
name: Softmax — Categorical Regression with Two Predictors
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/SoftmaxRegData1.csv
example: softmax
dataset: softmax1
chapter: DBDA2 Ch 22
formula: "Y ~ 0 + Intercept + X1 + X2"
----

Categorical/softmax regression with separate intercept per category; `categorical` family.
"""
function examples(::Val{:softmax_categorical})
    d1 = load(Val(:softmax1))
    return ("Y ~ 0 + Intercept + X1 + X2", d1)
end

"""
name: Softmax — Intercepts-Only Baseline
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/SoftmaxRegData1.csv
example: softmax
dataset: softmax1
chapter: DBDA2 Ch 22
formula: "Y ~ 1"
----

Intercepts-only softmax model; estimates baseline probability per category with no predictors.
"""
function examples(::Val{:softmax_baseline})
    d1 = load(Val(:softmax1))
    return ("Y ~ 1", d1)
end

"""
name: Conditional Logistic — Ordinal Regression (Data Set 1)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/CondLogistRegData1.csv
example: softmax
dataset: condlog1
chapter: DBDA2 Ch 22
formula: "Y_ord ~ 1 + cs(X1) + cs(X2)"
----

Sequential ordinal regression with category-specific slopes via `cs()`; `sratio` family. First conditional logistic dataset.
"""
function examples(::Val{:condlog1_ordinal})
    d3 = load(Val(:condlog1))
    return ("Y_ord ~ 1 + cs(X1) + cs(X2)", d3)
end

"""
name: Conditional Logistic — Ordinal Regression (Data Set 2)
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/CondLogistRegData2.csv
example: softmax
dataset: condlog2
chapter: DBDA2 Ch 22
formula: "Y_ord ~ 1 + cs(X1) + cs(X2)"
----

Sequential ordinal regression with category-specific slopes via `cs()`; `sratio` family. Second conditional logistic dataset.
"""
function examples(::Val{:condlog2_ordinal})
    d4 = load(Val(:condlog2))
    return ("Y_ord ~ 1 + cs(X1) + cs(X2)", d4)
end

"""
name: ordinal_probit — Synthetic Ordinal Probit Data
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/OrdinalProbitData1.csv
chapter: DBDA2 Ch 23
----

Synthetic ordinal outcome with a metric predictor. Columns: `Y` (ordered category 1–7), `X` (metric predictor).
"""
load(::Val{:ordinal_probit}) =
    CSV.read(Downloads.download(DBDA_URL * "OrdinalProbitData1.csv"), DataFrame)

"""
name: happiness_assets — Happiness and Financial Assets Data
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HappinessAssetsDebt.csv
chapter: DBDA2 Ch 23
----

Subjective happiness on a financial assets scale. Columns: `Happiness` (1–5), `Assets` (dollars), `Debt` (dollars), and other financial variables.
"""
load(::Val{:happiness_assets}) =
    CSV.read(Downloads.download(DBDA_URL * "HappinessAssetsDebt.csv"), DataFrame)

"""
name: movies — Rotten Tomatoes Movie Ratings
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/Movies.csv
chapter: DBDA2 Ch 23
----

Rotten Tomatoes movie ratings and metadata. Columns: `Title`, `Year`, `Length` (min), `Rating` (1–5 ordinal).
"""
load(::Val{:movies}) =
    CSV.read(Downloads.download(DBDA_URL * "Movies.csv"), DataFrame)

"""
name: Ordinal Probit — Intercept Only
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/OrdinalProbitData1.csv
example: ordinal_probit
dataset: ordinal_probit
chapter: DBDA2 Ch 23
formula: "Y ~ 1"
----

Intercept-only ordinal probit model; thresholds estimated with `cumulative` family and no predictors.
"""
function examples(::Val{:ordinal_probit_intercept})
    d = load(Val(:ordinal_probit))
    return ("Y ~ 1", d)
end

"""
name: Ordinal Probit — Discrimination Parameter
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/OrdinalProbitData1.csv
example: ordinal_probit
dataset: ordinal_probit
chapter: DBDA2 Ch 23
formula: "bf(Y ~ 1 + X) + lf(disc ~ 0 + X, cmc = FALSE)"
----

Ordinal model with discrimination parameter; `disc` controls category spacing and varies with X; `lf()` submodel.
"""
function examples(::Val{:ordinal_probit_disc})
    d = load(Val(:ordinal_probit))
    return ("bf(Y ~ 1 + X) + lf(disc ~ 0 + X, cmc = FALSE)", d)
end

"""
name: Ordinal Probit — Heteroscedastic Gaussian Comparison
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/OrdinalProbitData1.csv
example: ordinal_probit
dataset: ordinal_probit
chapter: DBDA2 Ch 23
formula: "bf(Y ~ 0 + X, sigma ~ 0 + X)"
----

Heteroscedastic Gaussian comparison model; `sigma` varies with X; alternative to ordinal family.
"""
function examples(::Val{:ordinal_probit_hetero})
    d = load(Val(:ordinal_probit))
    return ("bf(Y ~ 0 + X, sigma ~ 0 + X)", d)
end

"""
name: Happiness and Assets — Cumulative Ordinal Probit
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HappinessAssetsDebt.csv
example: happiness_assets
dataset: happiness_assets
chapter: DBDA2 Ch 23
formula: "Happiness ~ 1 + Assets_s"
----

Cumulative ordinal probit with standardized assets as predictor; `Assets_s` = (Assets - mean) / sd.
"""
function examples(::Val{:happiness_assets})
    d = load(Val(:happiness_assets))
    d.Assets_s = (d.Assets .- mean(d.Assets)) ./ std(d.Assets)
    return ("Happiness ~ 1 + Assets_s", d)
end

"""
name: Movie Ratings — Cumulative Ordinal Regression
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/Movies.csv
example: movies
dataset: movies
chapter: DBDA2 Ch 23
formula: "Rating ~ 1 + Year_s + Length_s"
----

Cumulative ordinal regression; `Year` and `Length` standardized to `Year_s` and `Length_s`.
"""
function examples(::Val{:movies})
    d = load(Val(:movies))
    d.Year_s   = (d.Year   .- mean(d.Year))   ./ std(d.Year)
    d.Length_s = (d.Length .- mean(d.Length)) ./ std(d.Length)
    return ("Rating ~ 1 + Year_s + Length_s", d)
end

"""
name: haireye — Hair and Eye Color Contingency Table
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HairEyeColor.csv
chapter: DBDA2 Ch 24
----

Hair and eye color counts; classic contingency-table example. Columns: `Hair` (Black/Brown/Red/Blond), `Eye` (Brown/Blue/Hazel/Green), `Sex`, `Count`.
"""
load(::Val{:haireye}) =
    CSV.read(Downloads.download(DBDA_URL * "HairEyeColor.csv"), DataFrame)

"""
name: Hair-Eye Color — Poisson Log-Linear
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HairEyeColor.csv
example: haireye
dataset: haireye
chapter: DBDA2 Ch 24
formula: "Count ~ 1 + (1 | Hair) + (1 | Eye) + (1 | Hair:Eye)"
----

Poisson log-linear model with main effects and interaction as random effects; models 4×4 contingency table.
"""
function examples(::Val{:haireye_poisson})
    d = load(Val(:haireye))
    return ("Count ~ 1 + (1 | Hair) + (1 | Eye) + (1 | Hair:Eye)", d)
end

"""
name: Hair-Eye Color — Binomial Version
source: https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/HairEyeColor.csv
example: haireye
dataset: haireye
chapter: DBDA2 Ch 24
formula: "Count | trials(264) ~ 1 + (1 | Hair) + (1 | Eye) + (1 | Hair:Eye)"
----

Binomial version of the 4×4 hair-eye contingency table; `trials(264)` = total N; same random-effects structure as Poisson variant.
"""
function examples(::Val{:haireye_binomial})
    d = load(Val(:haireye))
    return ("Count | trials(264) ~ 1 + (1 | Hair) + (1 | Eye) + (1 | Hair:Eye)", d)
end

"""
name: censored — Truncated and Censored Gaussian Data
source: synthetic
chapter: DBDA2 Ch 25
----

500 draws from Normal(100, 15) with censoring applied. Columns: `y` (original), `y_na` (with 50 MCAR missings), `y1` (left/right censored at 85/115), `cen1` (censoring indicator), `y2`/`y3` (interval bounds), `cen2` (= "interval").
"""
function load(::Val{:censored})
    rng = MersenneTwister(25)
    n   = 500
    y   = randn(rng, n) .* 15 .+ 100
    # Apply left censoring at 85 and right censoring at 115
    cen1 = ifelse.(y .< 85, "left", ifelse.(y .> 115, "right", "none"))
    y1   = clamp.(y, 85, 115)
    # Interval censoring: round to nearest 5
    y2   = floor.(y ./ 5) .* 5
    y3   = y2 .+ 5
    cen2 = fill("interval", n)
    # Missing values (MCAR)
    y_na = Vector{Union{Float64,Missing}}(y)
    y_na[shuffle(rng, 1:n)[1:50]] .= missing
    return DataFrame(; y, y_na, y1, cen1, y2, y3, cen2)
end

"""
name: Censored Data — Missing Values Model
source: synthetic
example: censored
dataset: censored
chapter: DBDA2 Ch 25
formula: "y_na ~ 1"
----

Gaussian model with missing values; brms uses listwise deletion by default. 50 MCAR missing observations out of 500.
"""
function examples(::Val{:censored_missing})
    d = load(Val(:censored))
    return ("y_na ~ 1", d)
end

"""
name: Censored Data — Left/Right Censoring
source: synthetic
example: censored
dataset: censored
chapter: DBDA2 Ch 25
formula: "y1 | cens(cen1) ~ 1"
----

Left/right censored Gaussian model; `cen1` ∈ {"left", "none", "right"}; censored at 85 and 115.
"""
function examples(::Val{:censored_lr})
    d = load(Val(:censored))
    return ("y1 | cens(cen1) ~ 1", d)
end

"""
name: Censored Data — Interval Censoring
source: synthetic
example: censored
dataset: censored
chapter: DBDA2 Ch 25
formula: "y2 | cens(cen2, y3) ~ 1"
----

Interval censored Gaussian model; `y3` = upper bound of each interval; values rounded to nearest 5.
"""
function examples(::Val{:censored_interval})
    d = load(Val(:censored))
    return ("y2 | cens(cen2, y3) ~ 1", d)
end
