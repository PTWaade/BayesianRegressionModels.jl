using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: "Doing Bayesian Data Analysis in brms and the tidyverse"
# A. Solomon Kurz (brms translation of John Kruschke's DBDA2, 2nd ed.)
# Source: https://github.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse
# Data:   https://github.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/tree/master/data.R
#
# Chapters without brm() calls (conceptual/prior/power discussions): 1–7, 10–15.

const DBDA_URL = "https://raw.githubusercontent.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse/master/data.R/"

##############################################################################
# DBDA2 Ch 8: Markov Chain Monte Carlo — Bernoulli/Binomial coin-flip models
#
# Dataset 1: z15N50 — 50 Bernoulli trials, 15 successes (heads).
# Dataset 2: z6N8z2N7 — two mints: mint 1 gives 6/8 heads, mint 2 gives 2/7.
#
# brms model formulas:
#   "y ~ 1"
#     (intercept-only Bernoulli; identity link estimates p directly)
#   "y ~ 0 + s"
#     (s = mint indicator; separate p per group, no intercept centering)
##############################################################################

load(::Val{:z15n50})   = CSV.read(Downloads.download(DBDA_URL * "z15N50.csv"),   DataFrame)
load(::Val{:z6n8z2n7}) = CSV.read(Downloads.download(DBDA_URL * "z6N8z2N7.csv"), DataFrame)

function examples(::Val{:z15n50})
    d = load(Val(:z15n50))
    return [("y ~ 1", d)]
end

function examples(::Val{:z6n8z2n7})
    d = load(Val(:z6n8z2n7))
    return [("y ~ 0 + s", d)]
end

##############################################################################
# DBDA2 Ch 9: Hierarchical models
#
# Dataset 1: TherapeuticTouchData — 280 therapeutic-touch trials (Benor et al.)
#   28 healers (10 trials each): y (1=correct, 0=incorrect), s (healer ID 1–28).
#
# Dataset 2: BattingAverage — 948 MLB player-season records.
#   Player, PriPos (primary position), Hits, AtBats, other fields.
#
# brms model formulas:
#   "y ~ 1 + (1 | s)"
#     (hierarchical Bernoulli; partial pooling across healers; logit link)
#   "Hits | trials(AtBats) ~ 1 + (1 | PriPos) + (1 | PriPos:Player)"
#     (hierarchical binomial; players nested within positions; batting average)
##############################################################################

load(::Val{:therapeutic_touch}) =
    CSV.read(Downloads.download(DBDA_URL * "TherapeuticTouchData.csv"), DataFrame)

load(::Val{:batting_average}) =
    CSV.read(Downloads.download(DBDA_URL * "BattingAverage.csv"), DataFrame)

function examples(::Val{:therapeutic_touch})
    d = load(Val(:therapeutic_touch))
    return [("y ~ 1 + (1 | s)", d)]
end

function examples(::Val{:batting_average})
    d = load(Val(:batting_average))
    return [("Hits | trials(AtBats) ~ 1 + (1 | PriPos) + (1 | PriPos:Player)", d)]
end

##############################################################################
# DBDA2 Ch 12: Binomial memory-recall experiment (simulated data)
# 6 conditions × n subjects, each recalling items from a list of 20.
# Data generated in R via expand_grid + rbinom; reproduced synthetically here.
#
# brms model formulas:
#   "n_recalled | trials(20) ~ 0 + condition"
#     (separate binomial probability per condition; no intercept)
#   "n_recalled | trials(20) ~ 1"
#     (pooled single intercept; comparison model)
##############################################################################

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

function examples(::Val{:recall})
    d = load(Val(:recall))
    return [
        ("n_recalled | trials(20) ~ 0 + condition", d),
        ("n_recalled | trials(20) ~ 1", d),
    ]
end

##############################################################################
# DBDA2 Ch 16: Metric outcome with one or two groups
# Source: https://jse.amstat.org/datasets/calcium.dat.txt (calcium supplementation RCT)
#
# Dataset 1: TwoGroupIQ — IQ scores for two treatment groups.
#   Columns: Group (Smart Drug / Placebo), Score (IQ score).
#
# Dataset 2: calcium — decrease in blood pressure after calcium supplementation.
#   Columns: treatment (Calcium/Placebo), decrease (mm Hg).
#   From: Lyle et al. (1987).
#
# brms model formulas:
#   "Score ~ 1"
#     (single-group Gaussian; prior-predictive / baseline)
#   "bf(Score ~ 0 + Group, sigma ~ 0 + Group)"
#     (two groups with group-specific mean AND sigma; student-t family for robustness)
#   "decrease ~ 0 + treatment"
#     (calcium RCT; separate group means; Gaussian or student)
##############################################################################

load(::Val{:two_group_iq}) =
    CSV.read(Downloads.download(DBDA_URL * "TwoGroupIQ.csv"), DataFrame)

function load(::Val{:calcium})
    url = "https://jse.amstat.org/datasets/calcium.dat.txt"
    cols = ["begin", "end", "decrease", "treatment"]
    CSV.read(Downloads.download(url), DataFrame;
             header = cols, skipto = 2, delim = ' ', ignorerepeated = true)
end

function examples(::Val{:two_group_iq})
    d = load(Val(:two_group_iq))
    return [
        ("Score ~ 1", d),
        ("bf(Score ~ 0 + Group, sigma ~ 0 + Group)", d),
    ]
end

function examples(::Val{:calcium})
    d = load(Val(:calcium))
    return [("decrease ~ 0 + treatment", d)]
end

##############################################################################
# DBDA2 Ch 17: Metric outcome with metric predictor(s)
#
# Dataset 1: HtWtData — height (cm) and weight (kg) for adults.
#   HtWtData30.csv (n=30), HtWtData300.csv (n=300). Columns: height, weight, male.
#
# Dataset 2: HierLinRegressData — hierarchical linear regression.
#   Multiple subjects (Subj), each with several (x, y) observations.
#
# Dataset 3: IncomeFamszState3yr — median household income by family size and U.S. state.
#   Columns: State, family_size, median_income, se (standard error).
#   Three years stacked.
#
# brms model formulas:
#   "weight_z ~ 1 + height_z"
#     (ch17: simple linear regression; z-standardized; student-t for robustness)
#   "y_z ~ 1 + x_z + (1 + x_z || Subj)"
#     (ch17: hierarchical with varying intercepts and slopes; || = uncorrelated)
#   "median_income_z | se(se_z, sigma = TRUE) ~ 1 + family_size_z + I(family_size_z^2) + (1 + family_size_z + I(family_size_z^2) || State)"
#     (ch17: measurement-error model via | se(); quadratic trend; state-level random slopes)
##############################################################################

load(::Val{:htwt30})  = CSV.read(Downloads.download(DBDA_URL * "HtWtData30.csv"),  DataFrame)
load(::Val{:htwt300}) = CSV.read(Downloads.download(DBDA_URL * "HtWtData300.csv"), DataFrame)
load(::Val{:htwt})    = load(Val(:htwt300))
load(::Val{:hier_linreg}) =
    CSV.read(Downloads.download(DBDA_URL * "HierLinRegressData.csv"), DataFrame)
load(::Val{:income_famsize}) =
    CSV.read(Downloads.download(DBDA_URL * "IncomeFamszState3yr.csv"), DataFrame)

function examples(::Val{:htwt})
    d = load(Val(:htwt300))
    d.weight_z = (d.weight .- mean(d.weight)) ./ std(d.weight)
    d.height_z = (d.height .- mean(d.height)) ./ std(d.height)
    return [
        ("weight_z ~ 1 + height_z", d),
    ]
end

function examples(::Val{:hier_linreg})
    d = load(Val(:hier_linreg))
    d.y_z = (d.Y .- mean(d.Y)) ./ std(d.Y)
    d.x_z = (d.X .- mean(d.X)) ./ std(d.X)
    return [
        ("y_z ~ 1 + x_z + (1 + x_z || Subj)", d),
    ]
end

function examples(::Val{:income_famsize})
    d = load(Val(:income_famsize))
    d.median_income_z = (d.median_income .- mean(d.median_income)) ./ std(d.median_income)
    d.family_size_z   = (d.family_size   .- mean(d.family_size))   ./ std(d.family_size)
    d.se_z            = d.se ./ std(d.median_income)
    return [
        ("median_income_z | se(se_z, sigma = TRUE) ~ 1 + family_size_z + I(family_size_z^2) + (1 + family_size_z + I(family_size_z^2) || State)", d),
    ]
end

##############################################################################
# DBDA2 Ch 18: Metric outcome with multiple metric predictors
# Source: Guber (1999) — U.S. state-level SAT scores.
#
# Dataset: Guber1999data — 50 states, 7 variables.
#   State, Sat (mean SAT score), Spend (expenditure per pupil, $1 000s),
#   PctSAT (% students taking SAT), TeachPay, StuTchRatio, SalaryScale, PupilExp.
#   Key variables standardized: satt_z, spend_z, prcnt_take_z, interaction_z.
#
# brms model formulas:
#   "satt_z ~ 1 + spend_z + prcnt_take_z"
#     (multiple regression; percent taking SAT is a suppressor variable)
#   "satt_z ~ 0 + Intercept + spend_z + prcnt_take_z + prop_not_take_z"
#     (adding complementary proportion; explicit centering via 0 + Intercept)
#   "satt_z ~ 1 + spend_z + prcnt_take_z + interaction_z"
#     (interaction_z = spend_z * prcnt_take_z; all student-t for robustness)
##############################################################################

load(::Val{:guber1999}) =
    CSV.read(Downloads.download(DBDA_URL * "Guber1999data.csv"), DataFrame)

function examples(::Val{:guber1999})
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
    return [
        ("satt_z ~ 1 + spend_z + prcnt_take_z", d),
        ("satt_z ~ 0 + Intercept + spend_z + prcnt_take_z + prop_not_take_z", d),
        ("satt_z ~ 1 + spend_z + prcnt_take_z + interaction_z", d),
    ]
end

##############################################################################
# DBDA2 Ch 19: Metric outcome with one nominal predictor (one-way ANOVA)
# Source: Partridge & Farquhar (1981) — fruit fly longevity experiment.
#
# Dataset: FruitflyDataReduced — 125 male Drosophila melanogaster.
#   CompanionNumber (1–5 companion types), Longevity (days), Thorax (mm).
#   CompanionNumber levels: Alone, 1 Virgin, 8 Virgin, 1 Pregnant, 8 Pregnant.
#
# brms model formulas:
#   "Longevity ~ 1 + (1 | CompanionNumber)"
#     (ANOVA as hierarchical model; partial pooling across companion types)
#   "Longevity ~ 1"
#     (pooled / no-group baseline for comparison)
#   "bf(Longevity ~ 0 + CompanionNumber, sigma ~ 0 + CompanionNumber)"
#     (robust ANOVA: group-specific means AND sigmas; student-t)
#   "Longevity ~ 1 + thorax_c + (1 | CompanionNumber)"
#     (ANCOVA: control for thorax length; thorax_c = centered thorax)
#   "Longevity ~ 1 + thorax_c + (1 + thorax_c | CompanionNumber)"
#     (ANHECOVA: random slope for thorax per companion group)
##############################################################################

load(::Val{:fruitfly}) =
    CSV.read(Downloads.download(DBDA_URL * "FruitflyDataReduced.csv"), DataFrame)

function examples(::Val{:fruitfly})
    d = load(Val(:fruitfly))
    d.thorax_c = d.Thorax .- mean(d.Thorax)
    return [
        ("Longevity ~ 1 + (1 | CompanionNumber)", d),
        ("Longevity ~ 1", d),
        ("bf(Longevity ~ 0 + CompanionNumber, sigma ~ 0 + CompanionNumber)", d),
        ("Longevity ~ 1 + thorax_c + (1 | CompanionNumber)", d),
        ("Longevity ~ 1 + thorax_c + (1 + thorax_c | CompanionNumber)", d),
    ]
end

##############################################################################
# DBDA2 Ch 20: Metric outcome with multiple nominal predictors (two-way ANOVA)
#
# Dataset 1: Salary — fictional salary data with position and organization.
#   Columns: Salary, Pos (position), Org (organization).
#
# Dataset 2: SplitPlotAgriData — split-plot agronomy experiment.
#   Columns: Yield, Till (tillage method), Fert (fertilizer), Field (blocking factor).
#
# brms model formulas:
#   "Salary ~ 1 + (1 | Pos) + (1 | Org) + (1 | Pos:Org)"
#     (two-way random effects ANOVA + interaction; Gaussian)
#   "bf(Salary ~ 1 + (1 | Pos) + (1 | Org) + (1 | Pos:Org), sigma ~ 1 + (1 | Pos:Org))"
#     (robust variant: heterogeneous residual variance per cell; student-t)
#   "Yield ~ 1 + (1 | Till) + (1 | Fert) + (1 | Field) + (1 | Till:Fert)"
#     (split-plot ANOVA: tillage × fertilizer + blocking on Field)
#   "Yield ~ 1 + (1 | Till) + (1 | Fert) + (1 | Till:Fert)"
#     (same without explicit field blocking)
##############################################################################

load(::Val{:salary})    = CSV.read(Downloads.download(DBDA_URL * "Salary.csv"),          DataFrame)
load(::Val{:splitplot}) = CSV.read(Downloads.download(DBDA_URL * "SplitPlotAgriData.csv"), DataFrame)

function examples(::Val{:salary})
    d = load(Val(:salary))
    return [
        ("Salary ~ 1 + (1 | Pos) + (1 | Org) + (1 | Pos:Org)", d),
        ("bf(Salary ~ 1 + (1 | Pos) + (1 | Org) + (1 | Pos:Org), sigma ~ 1 + (1 | Pos:Org))", d),
    ]
end

function examples(::Val{:splitplot})
    d = load(Val(:splitplot))
    return [
        ("Yield ~ 1 + (1 | Till) + (1 | Fert) + (1 | Field) + (1 | Till:Fert)", d),
        ("Yield ~ 1 + (1 | Till) + (1 | Fert) + (1 | Till:Fert)", d),
    ]
end

##############################################################################
# DBDA2 Ch 21: Dichotomous predicted variable (logistic regression)
# Source: Appleton, French & Vanderpump (1996) — height/weight/sex data.
#
# Dataset: HtWtData110 — 110 adults with height, weight, and sex.
#   Columns: height (cm), weight (kg), male (0/1).
#   Also re-uses BattingAverage.csv for hierarchical logistic.
#
# brms model formulas:
#   "male ~ 1 + weight_z"
#     (logistic regression; single metric predictor)
#   "male ~ 1 + weight_z + height_z"
#     (two metric predictors; collinear but informative)
#   "y ~ 1 + x1 + x2 + x1:x2"
#     (interaction; synthetic data)
#   "bf(male ~ a * 0.5 + (1 - a) * 1 / (1 + exp(-(b0 + b1 * weight_z))), a + b0 + b1 ~ 1, nl = TRUE)"
#     (robust logistic: a = probability of guessing; identity link; nl = TRUE)
#   "Hits | trials(AtBats) ~ 1 + (1 | PriPos) + (1 | PriPos:Player)"
#     (hierarchical logistic/binomial; same as ch9 but in logistic regression context)
##############################################################################

load(::Val{:htwt110}) =
    CSV.read(Downloads.download(DBDA_URL * "HtWtData110.csv"), DataFrame)

function examples(::Val{:htwt110})
    d = load(Val(:htwt110))
    d.weight_z = (d.weight .- mean(d.weight)) ./ std(d.weight)
    d.height_z = (d.height .- mean(d.height)) ./ std(d.height)
    return [
        ("male ~ 1 + weight_z", d),
        ("male ~ 1 + weight_z + height_z", d),
        ("bf(male ~ a * 0.5 + (1 - a) * 1 / (1 + exp(-(b0 + b1 * weight_z))), a + b0 + b1 ~ 1, nl = TRUE)", d),
    ]
end

##############################################################################
# DBDA2 Ch 22: Nominal predicted variable (softmax / categorical regression)
#
# Dataset 1: SoftmaxRegData1 / SoftmaxRegData2 — synthetic categorical outcome.
#   Columns: Y (category 1–3), X1 (metric), X2 (metric).
#
# Dataset 2: CondLogistRegData1 / CondLogistRegData2 — conditional logistic.
#   Columns: Y (ordered category), Y_ord, X1, X2.
#
# brms model formulas:
#   "Y ~ 0 + Intercept + X1 + X2"
#     (categorical/softmax regression; separate intercept per category)
#   "Y ~ 1"
#     (intercepts-only softmax; baseline probability per category)
#   "Y_ord ~ 1 + cs(X1) + cs(X2)"
#     (sequential ordinal regression with category-specific slopes; sratio family)
##############################################################################

load(::Val{:softmax1}) = CSV.read(Downloads.download(DBDA_URL * "SoftmaxRegData1.csv"),    DataFrame)
load(::Val{:softmax2}) = CSV.read(Downloads.download(DBDA_URL * "SoftmaxRegData2.csv"),    DataFrame)
load(::Val{:condlog1}) = CSV.read(Downloads.download(DBDA_URL * "CondLogistRegData1.csv"), DataFrame)
load(::Val{:condlog2}) = CSV.read(Downloads.download(DBDA_URL * "CondLogistRegData2.csv"), DataFrame)
load(::Val{:softmax})  = load(Val(:softmax1))

function examples(::Val{:softmax})
    d1 = load(Val(:softmax1))
    d3 = load(Val(:condlog1))
    d4 = load(Val(:condlog2))
    return [
        ("Y ~ 0 + Intercept + X1 + X2", d1),
        ("Y ~ 1", d1),
        ("Y_ord ~ 1 + cs(X1) + cs(X2)", d3),   # sequential ordinal (sratio)
        ("Y_ord ~ 1 + cs(X1) + cs(X2)", d4),
    ]
end

##############################################################################
# DBDA2 Ch 23: Ordinal predicted variable (ordered probit / cumulative)
#
# Dataset 1: OrdinalProbitData1 — synthetic ordinal Y (1–7) with metric X.
# Dataset 2: HappinessAssetsDebt — subjective happiness on financial assets scale.
#   Columns: Happiness (1–5), Assets (dollars), Debt (dollars), etc.
# Dataset 3: Movies — Rotten Tomatoes movie ratings and metadata.
#   Columns: Title, Year, Length (min), Rating (1–5 ordinal).
#
# brms model formulas:
#   "Y ~ 1"
#     (intercept-only ordinal probit; thresholds estimated; cumulative family)
#   "bf(Y ~ 1 + X) + lf(disc ~ 0 + X, cmc = FALSE)"
#     (ordinal with discrimination parameter; disc controls category spacing)
#   "bf(Y ~ 0 + X, sigma ~ 0 + X)"
#     (heteroscedastic Gaussian comparison model; sigma varies with X)
#   "Happiness ~ 1 + Assets_s"
#     (ordinal probit; cumulative(probit); Assets_s = standardized assets)
#   "Rating ~ 1 + Year_s + Length_s"
#     (movie rating as cumulative ordinal; Year and Length standardized)
##############################################################################

load(::Val{:ordinal_probit}) =
    CSV.read(Downloads.download(DBDA_URL * "OrdinalProbitData1.csv"), DataFrame)
load(::Val{:happiness_assets}) =
    CSV.read(Downloads.download(DBDA_URL * "HappinessAssetsDebt.csv"), DataFrame)
load(::Val{:movies}) =
    CSV.read(Downloads.download(DBDA_URL * "Movies.csv"), DataFrame)

function examples(::Val{:ordinal_probit})
    d = load(Val(:ordinal_probit))
    return [
        ("Y ~ 1", d),
        ("bf(Y ~ 1 + X) + lf(disc ~ 0 + X, cmc = FALSE)", d),
        ("bf(Y ~ 0 + X, sigma ~ 0 + X)", d),
    ]
end

function examples(::Val{:happiness_assets})
    d = load(Val(:happiness_assets))
    d.Assets_s = (d.Assets .- mean(d.Assets)) ./ std(d.Assets)
    return [("Happiness ~ 1 + Assets_s", d)]
end

function examples(::Val{:movies})
    d = load(Val(:movies))
    d.Year_s   = (d.Year   .- mean(d.Year))   ./ std(d.Year)
    d.Length_s = (d.Length .- mean(d.Length)) ./ std(d.Length)
    return [("Rating ~ 1 + Year_s + Length_s", d)]
end

##############################################################################
# DBDA2 Ch 24: Count predicted variable (Poisson / log-linear)
#
# Dataset: HairEyeColor — hair and eye color counts.
#   Columns: Hair (Black/Brown/Red/Blond), Eye (Brown/Blue/Hazel/Green), Sex, Count.
#   Classic contingency-table example.
#
# brms model formulas:
#   "Count ~ 1 + (1 | Hair) + (1 | Eye) + (1 | Hair:Eye)"
#     (Poisson log-linear; main effects + interaction as random effects)
#   "count | trials(264) ~ 1 + (1 | a) + (1 | b) + (1 | a:b)"
#     (binomial version of the same 4×4 contingency table; trials = total N)
##############################################################################

load(::Val{:haireye}) =
    CSV.read(Downloads.download(DBDA_URL * "HairEyeColor.csv"), DataFrame)

function examples(::Val{:haireye})
    d = load(Val(:haireye))
    return [
        ("Count ~ 1 + (1 | Hair) + (1 | Eye) + (1 | Hair:Eye)", d),
        ("Count | trials(264) ~ 1 + (1 | Hair) + (1 | Eye) + (1 | Hair:Eye)", d),
    ]
end

##############################################################################
# DBDA2 Ch 25: Tools for robust modeling — truncated and censored data
# Synthetic data: 500 draws from Normal(100, 15) with truncation/censoring applied.
#
# brms model formulas:
#   "y_na ~ 1"
#     (Gaussian with missing values; brms uses listwise deletion by default)
#   "y1 | cens(cen1) ~ 1"
#     (left/right censored; cen1 in {\"left\", \"none\", \"right\"})
#   "y2 | cens(cen2, y3) ~ 1"
#     (interval censored; y3 = upper bound of interval)
##############################################################################

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

function examples(::Val{:censored})
    d = load(Val(:censored))
    return [
        ("y_na ~ 1", d),
        ("y1 | cens(cen1) ~ 1", d),
        ("y2 | cens(cen2, y3) ~ 1", d),
    ]
end
