using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: "Statistical Rethinking with brms, ggplot2, and the tidyverse: 2nd ed."
# A. Solomon Kurz (brms translation of Richard McElreath's Statistical Rethinking, 2nd ed.)
# Source: https://github.com/ASKurz/Statistical_Rethinking_with_brms_ggplot2_and_the_tidyverse_2_ed
# Data:   https://github.com/rmcelreath/rethinking/tree/master/data

const RETHINKING_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/"

##############################################################################
# SR2 Ch 2–3: Globe tossing
# Binomial likelihood for estimating p(water) from globe-tossing experiment.
# Data: 24 water observations out of 36 tosses (ch2); 6 out of 9 (ch3 recap).
#
# brms model formula:
#   "w | trials(n) ~ 0 + Intercept"   (family = binomial(link = "identity"))
#     (0 + Intercept fixes the link to identity, estimating p directly)
##############################################################################

load(::Val{:globe}) = DataFrame(w = [24], n = [36])

function examples(::Val{:globe})
    data = load(Val(:globe))
    return [
        ("w | trials(n) ~ 0 + Intercept", data),
    ]
end

##############################################################################
# SR2 Ch 4: Heights and weights — Gaussian regression
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv
#
# Dataset: Howell1 — !Kung San census data (Nancy Howell)
#   544 individuals: height (cm), weight (kg), age (years), male (0/1).
#   Adults only (age ≥ 18, n=352) used for the linear height ~ weight models.
#   Full dataset (including children) used for polynomial models.
#
# brms model formulas:
#   "height ~ 1"
#     (intercept-only Gaussian; prior predictive / baseline)
#   "height ~ 1 + weight_c"
#     (simple linear regression; weight_c = weight − mean(weight))
#   "height ~ 1 + weight_s + weight_s2"
#     (quadratic; weight_s = standardized weight, weight_s2 = weight_s^2)
#   "height ~ 1 + weight_s + weight_s2 + weight_s3"
#     (cubic polynomial)
#   "bf(height ~ a + exp(lb) * weight_c, a ~ 1, lb ~ 1, nl = TRUE)"
#     (nonlinear: exponential growth model for height on weight; nl=TRUE)
#
# SR2 Ch 16 revisits Howell1 for a nonlinear volume model:
#   "bf(w ~ log(pi * k * p^2 * h^3), k + p ~ 1, nl = TRUE)"
#     (lognormal; models body weight as a scaled cylinder; w=weight, h=height)
##############################################################################

load(::Val{:howell1}) = CSV.read(Downloads.download(RETHINKING_URL * "Howell1.csv"), DataFrame)

function examples(::Val{:howell1})
    d  = load(Val(:howell1))
    d2 = filter(r -> r.age >= 18, d)          # adults only
    d2.weight_c = d2.weight .- mean(d2.weight)
    d.weight_s  = (d.weight .- mean(d.weight)) ./ std(d.weight)
    d.weight_s2 = d.weight_s .^ 2
    d.weight_s3 = d.weight_s .^ 3
    return [
        ("height ~ 1", d2),
        ("height ~ 1 + weight_c", d2),
        ("height ~ 1 + weight_s + weight_s2", d),
        ("height ~ 1 + weight_s + weight_s2 + weight_s3", d),
        ("bf(height ~ a + exp(lb) * weight_c, a ~ 1, lb ~ 1, nl = TRUE)", d2),
        ("bf(w ~ log(3.141593 * k * p^2 * h^3), k + p ~ 1, nl = TRUE)", DataFrame(w = d.weight, h = d.height)),
    ]
end

##############################################################################
# SR2 Ch 4: Cherry blossoms — B-spline regression
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/cherry_blossoms.csv
#
# Dataset: Cherry blossoms bloom dates in Japan, years 801–2015.
#   827 observations after dropping rows with missing `doy` (day of year).
#
# brms model formula:
#   "doy ~ 1 + B"
#     (B is a B-spline basis matrix; each column is a basis function)
#     (constructed via bs(year, knots=iknots) in R; ~ 17 basis columns)
#
# Note: also covered in bambi.jl (:cherry_blossoms) with Bambi/Patsy spline syntax.
##############################################################################

load(::Val{:cherry_blossoms}) =
    CSV.read(Downloads.download(RETHINKING_URL * "cherry_blossoms.csv"), DataFrame)

function examples(::Val{:cherry_blossoms})
    d = dropmissing(load(Val(:cherry_blossoms)), :doy)
    # B is a B-spline design matrix; here we note the formula symbolically.
    # In practice: B = R's bs(d$year, knots=quantile(d$year, probs=seq(0.1,0.9,by=0.1)))
    return [
        ("doy ~ 1 + B", d),
    ]
end

##############################################################################
# SR2 Ch 5: Spurious and masked associations — multiple regression
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/WaffleDivorce.csv
#
# Dataset: WaffleDivorce — U.S. state-level divorce and marriage statistics.
#   50 states: Divorce rate, Marriage rate, MedianAgeMarriage, WaffleHouses, South, etc.
#   Variables standardized in examples: D (divorce), A (median age at marriage), M (marriage rate).
#
# brms model formulas:
#   "D ~ 1 + A"     (ch5: simple regression; A predicts D)
#   "D ~ 1 + M"     (ch5: simple regression)
#   "D ~ 1 + M + A" (ch5: multiple regression — M's effect vanishes after conditioning on A)
#   "M ~ 1 + A"     (ch5: mediator regression)
#   "bf(D ~ 1 + M + A) + bf(M ~ 1 + A) + set_rescor(FALSE)"
#     (ch5: multivariate regression; simultaneous causal model for D and M)
#   "D ~ 1 + A + M" (ch6: collider / pipe demonstration with WaffleHouses)
#   "D_obs | mi(D_sd) ~ 1 + A + M"
#     (ch15: measurement error model; D_sd = standard error of Divorce estimate)
##############################################################################

load(::Val{:waffle_divorce}) =
    CSV.read(Downloads.download(RETHINKING_URL * "WaffleDivorce.csv"), DataFrame)

function examples(::Val{:waffle_divorce})
    d = load(Val(:waffle_divorce))
    d.D = (d.Divorce         .- mean(d.Divorce))         ./ std(d.Divorce)
    d.A = (d.MedianAgeMarriage .- mean(d.MedianAgeMarriage)) ./ std(d.MedianAgeMarriage)
    d.M = (d.Marriage        .- mean(d.Marriage))        ./ std(d.Marriage)
    # ch15: D_sd is an illustration; McElreath treats Divorce SE as known (≈ 0.1 * |D|)
    d.D_sd  = 0.1 .* abs.(d.D) .+ 0.01
    d.D_obs = d.D .+ d.D_sd .* randn(MersenneTwister(15), nrow(d))
    return [
        ("D ~ 1 + A", d),
        ("D ~ 1 + M", d),
        ("D ~ 1 + M + A", d),
        ("M ~ 1 + A", d),
        ("bf(D ~ 1 + M + A) + bf(M ~ 1 + A) + set_rescor(FALSE)", d),
        ("D_obs | mi(D_sd) ~ 1 + A + M", d),
    ]
end

##############################################################################
# SR2 Ch 5–6: Milk energy and primate neocortex — masked association
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv
#
# Dataset: milk — primate milk composition and life history.
#   29 species; 12 with complete neocortex.perc data.
#   Columns: clade, species, kcal.per.g, perc.fat, perc.protein,
#            perc.lactose, mass (kg), neocortex.perc (% brain that is neocortex).
#
# brms model formulas:
#   "kcal.per.g_s ~ 1 + neocortex.perc_s"
#     (ch5: positive association before controlling for body mass)
#   "kcal.per.g_s ~ 1 + log_mass_s"
#     (ch5: negative association; larger species produce less calorie-dense milk)
#   "kcal.per.g_s ~ 1 + neocortex.perc_s + log_mass_s"
#     (ch5: both effects revealed simultaneously — masked association)
#   "height ~ 0 + sex"
#     (ch5: using Howell1 — sex as index variable with 0-suppressed intercept)
#   "kcal.per.g_s ~ 0 + clade"
#     (ch5: clade index model)
#   "kcal.per.g_s ~ 0 + clade + house"
#     (ch5: clade + arbitrary 'house' variable added for demonstration)
#   "bf(kcal.per.g_s ~ 0 + a + h, a ~ 0 + clade, h ~ 0 + house, nl = TRUE)"
#     (ch5: nonlinear reformulation of the index model)
#   "bf(k ~ 1 + mi(b) + m) + bf(b | mi() ~ 1) + set_rescor(FALSE)"
#     (ch15: joint missing-data model; b=neocortex.perc with missing values)
##############################################################################

load(::Val{:milk}) =
    CSV.read(Downloads.download(RETHINKING_URL * "milk.csv"), DataFrame)

function examples(::Val{:milk})
    d = load(Val(:milk))
    # Add underscore-name aliases so Julia can access dot-named columns
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
    return [
        ("kcal_s ~ 1 + neo_s", dcc),
        ("kcal_s ~ 1 + logmass_s", dcc),
        ("kcal_s ~ 1 + neo_s + logmass_s", dcc),
        ("kcal_s ~ 0 + clade", dcc),
        ("bf(k ~ 1 + mi(b) + m) + bf(b | mi() ~ 1) + set_rescor(FALSE)", d),
    ]
end

##############################################################################
# SR2 Ch 6: Plant growth experiment — post-treatment bias / collider
# Synthetic data generated by McElreath (no external CSV).
# 200 plants: h0 (initial height), h1 (final height), treatment (0/1), fungus (0/1).
# Fungus is a mediator: treatment → fungus → h1. Including fungus blocks the
# treatment effect (post-treatment bias). Excluding fungus gives causal estimate.
#
# brms model formulas:
#   "h1 ~ 0 + h0"
#     (baseline: no treatment effect)
#   "bf(h1 ~ h0 * (a + t * treatment + f * fungus), a + t + f ~ 1, nl = TRUE)"
#     (post-treatment bias: treatment effect absorbed by conditioning on fungus)
#   "bf(h1 ~ h0 * (a + t * treatment), a + t ~ 1, nl = TRUE)"
#     (causal model: treatment effect recovered by excluding fungus)
##############################################################################

function load(::Val{:plant_growth})
    rng = MersenneTwister(6)
    n = 200
    h0        = rand(rng, n) .+ 10.0
    treatment = repeat([0, 1], n ÷ 2)
    fungus    = [rand(rng) < (treatment[i] == 1 ? 0.1 : 0.5) ? 1 : 0 for i in 1:n]
    h1        = h0 .* (1.5 .- 0.2 .* fungus .+ 0.05 .* randn(rng, n))
    return DataFrame(; h0, h1, treatment = float.(treatment), fungus = float.(fungus))
end

function examples(::Val{:plant_growth})
    d = load(Val(:plant_growth))
    return [
        ("h1 ~ 0 + h0", d),
        ("bf(h1 ~ h0 * (a + t * treatment + f * fungus), a + t + f ~ 1, nl = TRUE)", d),
        ("bf(h1 ~ h0 * (a + t * treatment), a + t ~ 1, nl = TRUE)", d),
    ]
end

##############################################################################
# SR2 Ch 6: Happiness and marriage — collider bias
# Synthetic data: 1 000 individuals followed over 10 simulated years.
# Columns: age (1–65), happiness (−2 to 2), married (0/1), mid (married+1 as index).
# Marriage is a collider between age and happiness; conditioning on it induces
# a spurious negative association between age and happiness.
#
# brms model formulas:
#   "happiness ~ 0 + mid + a"
#     (mid = marriage indicator as index; collider model — spurious age effect)
#   "happiness ~ 0 + Intercept + a"
#     (no conditioning on marriage — age has no effect on happiness)
##############################################################################

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

function examples(::Val{:happiness})
    d = load(Val(:happiness))
    return [
        ("happiness ~ 0 + mid + a", d),     # collider — spurious age effect
        ("happiness ~ 0 + Intercept + a", d), # causal — no age effect
    ]
end

##############################################################################
# SR2 Ch 8–9: Terrain ruggedness and GDP — continent interaction
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/rugged.csv
#
# Dataset: rugged — cross-national terrain ruggedness and economic data.
#   234 countries: rugged (ruggedness index), log_gdp (log GDP per capita 2000),
#   cont_africa (1 if African country), isocode.
#   Subset to rows with non-missing log_gdp.
#
# brms model formulas:
#   "log_gdp_std ~ 1 + rugged_std_c"
#     (ch8: pooled model, ignoring continent)
#   "log_gdp_std ~ 0 + cid + rugged_std_c"
#     (ch8: continent intercepts via index variable cid; ruggedness same slope)
#   "bf(log_gdp_std ~ 0 + a + b * rugged_std_c, a ~ 0 + cid, b ~ 0 + cid, nl = TRUE)"
#     (ch8: continent-specific intercepts AND slopes; ruggedness hurts outside Africa)
#   "bf(log_gdp_std ~ 0 + a + b * rugged_std_c, a ~ 0 + cid, b ~ 0 + cid, nl = TRUE)"
#     (ch8: same with student family for robustness)
##############################################################################

load(::Val{:rugged}) =
    CSV.read(Downloads.download(RETHINKING_URL * "rugged.csv"), DataFrame)

function examples(::Val{:rugged})
    d  = dropmissing(load(Val(:rugged)), :rgdppc_2000)
    d.log_gdp     = log.(d.rgdppc_2000)
    d.log_gdp_std = d.log_gdp ./ mean(d.log_gdp)
    d.rugged_std  = d.rugged ./ maximum(d.rugged)
    d.rugged_std_c = d.rugged_std .- mean(d.rugged_std)
    d.cid         = ifelse.(d.cont_africa .== 1, "Africa", "Other")
    return [
        ("log_gdp_std ~ 1 + rugged_std_c", d),
        ("log_gdp_std ~ 0 + cid + rugged_std_c", d),
        ("bf(log_gdp_std ~ 0 + a + b * rugged_std_c, a ~ 0 + cid, b ~ 0 + cid, nl = TRUE)", d),
    ]
end

##############################################################################
# SR2 Ch 8: Tulips — continuous interaction
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/tulips.csv
#
# Dataset: tulips — greenhouse tulip experiment.
#   27 plants: blooms (flower size), water (1–3), shade (1–3), bed (a/b/c).
#
# brms model formulas:
#   "blooms_std ~ 1 + water_cent + shade_cent"
#     (additive model; water_cent and shade_cent are mean-centered)
#   "blooms_std ~ 1 + water_cent + shade_cent + water_cent:shade_cent"
#     (interaction: water and shade interact; effect of water depends on shade level)
##############################################################################

load(::Val{:tulips}) =
    CSV.read(Downloads.download(RETHINKING_URL * "tulips.csv"), DataFrame)

function examples(::Val{:tulips})
    d = load(Val(:tulips))
    d.blooms_std  = d.blooms ./ maximum(d.blooms)
    d.water_cent  = d.water .- mean(d.water)
    d.shade_cent  = d.shade .- mean(d.shade)
    return [
        ("blooms_std ~ 1 + water_cent + shade_cent", d),
        ("blooms_std ~ 1 + water_cent + shade_cent + water_cent:shade_cent", d),
    ]
end

##############################################################################
# SR2 Ch 10: Distributional model — heteroscedastic Gaussian
# Synthetic data; sigma varies with a predictor x.
#
# brms model formula:
#   "bf(y ~ 1, sigma ~ 1 + x)"
#     (distributional model: both mean and log(sigma) have their own sub-models)
##############################################################################

function load(::Val{:hetero})
    rng = MersenneTwister(10)
    n = 100
    x = randn(rng, n)
    y = randn(rng, n) .* exp.(0.5 .* x)
    return DataFrame(; y, x)
end

function examples(::Val{:hetero})
    d = load(Val(:hetero))
    return [("bf(y ~ 1, sigma ~ 1 + x)", d)]
end

##############################################################################
# SR2 Ch 11: Chimpanzees — prosocial choice (binomial GLM + actor effects)
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/chimpanzees.csv
#
# Dataset: chimpanzees — lever-pulling experiment across 7 actors and 6 blocks.
#   504 trials: pulled_left (0/1), prosoc_left (0/1), condition (0/1),
#   actor (1–7), block (1–6). treatment = 1 + prosoc_left + 2*condition.
#
# brms model formulas:
#   "pulled_left | trials(1) ~ 1"
#     (ch11: intercept-only binomial; baseline probability of pulling left)
#   "bf(pulled_left | trials(1) ~ a + b, a ~ 0 + actor, b ~ 0 + treatment, nl = TRUE)"
#     (ch11: actor-indexed intercepts + treatment effects; no pooling)
#   "bf(pulled_left | trials(1) ~ a + b, a ~ 1 + (1 | actor) + (1 | block), b ~ 0 + treatment, nl = TRUE)"
#     (ch13: multilevel — partial pooling across actors and blocks)
#   "pulled_left | trials(1) ~ 0 + treatment + (0 + treatment | actor) + (0 + treatment | block)"
#     (ch14: varying slopes — each actor/block has its own treatment-effect vector)
##############################################################################

load(::Val{:chimpanzees}) =
    CSV.read(Downloads.download(RETHINKING_URL * "chimpanzees.csv"), DataFrame)

function examples(::Val{:chimpanzees})
    d = load(Val(:chimpanzees))
    d.treatment = string.(1 .+ d.prosoc_left .+ 2 .* d.condition)
    d.actor     = string.(d.actor)
    d.block     = string.(d.block)
    return [
        ("pulled_left | trials(1) ~ 1", d),
        ("bf(pulled_left | trials(1) ~ a + b, a ~ 0 + actor, b ~ 0 + treatment, nl = TRUE)", d),
        ("bf(pulled_left | trials(1) ~ a + b, a ~ 1 + (1 | actor) + (1 | block), b ~ 0 + treatment, nl = TRUE)", d),
        ("pulled_left | trials(1) ~ 0 + treatment + (0 + treatment | actor) + (0 + treatment | block)", d),
    ]
end

##############################################################################
# SR2 Ch 11: UC Berkeley admissions — Simpson's paradox (binomial)
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/UCBadmit.csv
#
# Dataset: UCBadmit — graduate admissions by department and gender (1973).
#   12 rows: dept (A–F), applicant.gender (male/female), admit, reject, applications.
#
# brms model formulas:
#   "admit | trials(applications) ~ 0 + gid"
#     (ch11: gid = gender index; apparent gender gap)
#   "bf(admit | trials(applications) ~ a + d, a ~ 0 + gid, d ~ 0 + dept, nl = TRUE)"
#     (ch11: gender + department; gap disappears after conditioning on dept)
#   "admit | vint(applications) ~ 0 + gid"
#     (ch12: custom beta-binomial family; vint passes integer auxiliary data)
##############################################################################

load(::Val{:ucbadmit}) =
    CSV.read(Downloads.download(RETHINKING_URL * "UCBadmit.csv"), DataFrame)

function examples(::Val{:ucbadmit})
    d = load(Val(:ucbadmit))
    d.gid = string.(d[!, "applicant.gender"])
    return [
        ("admit | trials(applications) ~ 0 + gid", d),
        ("bf(admit | trials(applications) ~ a + d, a ~ 0 + gid, d ~ 0 + dept, nl = TRUE)", d),
        ("admit | vint(applications) ~ 0 + gid", d),
    ]
end

##############################################################################
# SR2 Ch 11–12: Kline island tool counts — Poisson and negative binomial GLM
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/Kline.csv
#         https://github.com/rmcelreath/rethinking/blob/master/data/Kline2.csv
#
# Dataset: Kline — tool counts for 10 Pacific island societies.
#   Columns: culture, population, contact (high/low), total.tools, mean.TU.
#   Kline2 adds lat (latitude) and lon2 (longitude on [0,2π]) for the GP model.
#
# brms model formulas:
#   "total_tools ~ 1"
#     (ch11: intercept-only Poisson; log link)
#   "bf(total_tools ~ a + b * log_pop_std, a + b ~ 0 + cid, nl = TRUE)"
#     (ch11: contact-index intercepts and slopes on log-population)
#   "bf(total_tools ~ exp(a) * population^b / g, a + b ~ 0 + cid, g ~ 1, nl = TRUE)"
#     (ch11: scientific (power-law) model with identity link)
#   "total_tools ~ 1"
#     (ch12: negative binomial variant)
#   "bf(total_tools ~ exp(a) * population^b / g, a ~ 1 + gp(lat_adj, lon2_adj, scale=FALSE), b + g ~ 1, nl = TRUE)"
#     (ch14: adds Gaussian process over geographic coordinates; Kline2 required)
##############################################################################

load(::Val{:kline})  = CSV.read(Downloads.download(RETHINKING_URL * "Kline.csv"),  DataFrame)
load(::Val{:kline2}) = CSV.read(Downloads.download(RETHINKING_URL * "Kline2.csv"), DataFrame)

function examples(::Val{:kline})
    d = load(Val(:kline))
    d.log_pop     = log.(d.population)
    d.log_pop_std = (d.log_pop .- mean(d.log_pop)) ./ std(d.log_pop)
    d.cid         = string.(d.contact)
    return [
        ("total_tools ~ 1", d),
        ("bf(total_tools ~ a + b * log_pop_std, a + b ~ 0 + cid, nl = TRUE)", d),
        ("bf(total_tools ~ exp(a) * population^b / g, a + b ~ 0 + cid, g ~ 1, nl = TRUE)", d),
    ]
end

function examples(::Val{:kline2})
    d = load(Val(:kline2))
    d.log_pop     = log.(d.population)
    d.log_pop_std = (d.log_pop .- mean(d.log_pop)) ./ std(d.log_pop)
    d.cid         = string.(d.contact)
    d.lat_adj     = d.lat .- mean(d.lat)
    return [
        ("bf(total_tools ~ exp(a) * population^b / g, a ~ 1 + gp(lat_adj, lon2_adj, scale=FALSE), b + g ~ 1, nl = TRUE)", d),
    ]
end

##############################################################################
# SR2 Ch 12: Trolley — ordered logistic regression + monotonic education
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/Trolley.csv
#
# Dataset: Trolley — trolley-problem moral intuitions (McElreath & Turpin 2022).
#   9 930 rows: response (1–7), action (0/1), intention (0/1), contact (0/1),
#   edu (8-level ordered education), age, male, id, story.
#
# brms model formulas:
#   "response ~ 1"
#     (ch12: intercept-only cumulative ordinal; 6 thresholds estimated)
#   "response ~ 1 + action + contact + intention + intention:action + intention:contact"
#     (ch12: action/contact/intention effects on moral acceptability)
#   "response ~ 1 + action + contact + intention + mo(edu_new)"
#     (ch12: monotonic effect of ordered education; edu_new = integer-coded edu)
##############################################################################

load(::Val{:trolley}) =
    CSV.read(Downloads.download(RETHINKING_URL * "Trolley.csv"), DataFrame)

function examples(::Val{:trolley})
    d = load(Val(:trolley))
    edu_order = ["Elementary School", "Middle School", "Some High School",
                 "High School Graduate", "Some College", "Bachelor's Degree",
                 "Master's Degree", "Graduate Degree"]
    edu_map   = Dict(e => i for (i, e) in enumerate(edu_order))
    d.edu_new = [get(edu_map, e, missing) for e in d.edu]
    return [
        ("response ~ 1", d),
        ("response ~ 1 + action + contact + intention + intention:action + intention:contact", d),
        ("response ~ 1 + action + contact + intention + mo(edu_new)", d),
    ]
end

##############################################################################
# SR2 Ch 12–13: Reed frogs — survival experiment, varying intercepts
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/reedfrogs.csv
#
# Dataset: reedfrogs — tadpole tank survival experiment.
#   48 tanks: density (initial tadpoles), pred (predation: no/pred),
#   size (small/big), surv (survivors), propsurv (proportion surviving).
#
# brms model formulas:
#   "surv | trials(density) ~ 0 + factor(tank)"
#     (ch13: no-pooling model; one parameter per tank)
#   "surv | trials(density) ~ 1 + (1 | tank)"
#     (ch13: partial pooling — adaptive regularization via multilevel)
##############################################################################

load(::Val{:reedfrogs}) =
    CSV.read(Downloads.download(RETHINKING_URL * "reedfrogs.csv"), DataFrame)

function examples(::Val{:reedfrogs})
    d = load(Val(:reedfrogs))
    d.tank = string.(1:nrow(d))
    return [
        ("surv | trials(density) ~ 0 + factor(tank)", d),
        ("surv | trials(density) ~ 1 + (1 | tank)", d),
    ]
end

##############################################################################
# SR2 Ch 14: Café visit times — varying slopes (Gaussian simulation)
# Synthetic data: 20 cafés, 10 morning and 10 afternoon visits each.
# afternoon (0/1) effect on wait time; both intercept and slope vary by café.
#
# brms model formula:
#   "wait ~ 1 + afternoon + (1 + afternoon | cafe)"
#     (varying intercepts AND slopes; café-level covariance between them)
##############################################################################

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

function examples(::Val{:cafe})
    d = load(Val(:cafe))
    return [("wait ~ 1 + afternoon + (1 + afternoon | cafe)", d)]
end

##############################################################################
# SR2 Ch 14–15: Primates — phylogenetic regression
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/Primates301.csv
#
# Dataset: Primates301 — life history and social learning in 301 primate species.
#   Columns include: brain (brain size), body (body mass), group_size,
#   social_learning, research_effort, name (species name).
#
# brms model formulas:
#   "b ~ 1 + m + g"
#     (ch14: brain ~ body mass + group size; phylogeny ignored)
#   "b ~ 1 + m + g + fcor(R)"
#     (ch14: same + phylogenetic correlation matrix R passed via data2;
#      R must be built from a phylogenetic tree, e.g. via ape::vcv.phylo())
##############################################################################

load(::Val{:primates301}) =
    CSV.read(Downloads.download(RETHINKING_URL * "Primates301.csv"), DataFrame)

function examples(::Val{:primates301})
    d  = dropmissing(load(Val(:primates301)), [:brain, :body, :group_size])
    d.b = (log.(d.brain)      .- mean(log.(d.brain)))      ./ std(log.(d.brain))
    d.m = (log.(d.body)       .- mean(log.(d.body)))        ./ std(log.(d.body))
    d.g = (log.(d.group_size) .- mean(log.(d.group_size))) ./ std(log.(d.group_size))
    return [
        ("b ~ 1 + m + g", d),
        ("b ~ 1 + m + g + fcor(R)", d),   # R = phylogenetic cov matrix (from ape::vcv.phylo)
    ]
end

##############################################################################
# SR2 Ch 15: Moralizing gods — missing-data handling (binomial)
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/Moralizing_gods.csv
#
# Dataset: Moralizing_gods — Seshat database of historical polities.
#   Presence of moralizing gods, writing, and social complexity across centuries.
#   Key columns: polity, year, moralizing_gods (0/1/NA), writing (0/1/NA),
#   social_scale (log10 of social scale).
#
# brms model formula:
#   "moralizing_gods | trials(1) ~ 1 + writing"
#     (ch15: do moralizing gods predict / follow emergence of writing?)
##############################################################################

load(::Val{:moralizing_gods}) =
    CSV.read(Downloads.download(RETHINKING_URL * "Moralizing_gods.csv"), DataFrame)

function examples(::Val{:moralizing_gods})
    d = dropmissing(load(Val(:moralizing_gods)), [:moralizing_gods, :writing])
    return [
        ("moralizing_gods | trials(1) ~ 1 + writing", d),
    ]
end

##############################################################################
# SR2 Ch 16: Panda nuts — nonlinear learning-curve model
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/Panda_nuts.csv
#
# Dataset: Panda_nuts — chimpanzee nut-cracking proficiency over development.
#   Columns: name, site, group, age (years), rounds, n.pulls (attempts),
#            success (successes), seconds (observation time).
#
# brms model formula:
#   "bf(n_panda ~ seconds * phi * (1 - exp(-k * age_s))^theta, phi + k + theta ~ 1, nl = TRUE)"
#     (poisson(link = identity); n_panda = success; phi = asymptotic rate,
#      k = growth rate, theta = shape; age_s = standardized age)
##############################################################################

load(::Val{:panda_nuts}) =
    CSV.read(Downloads.download(RETHINKING_URL * "Panda_nuts.csv"), DataFrame)

function examples(::Val{:panda_nuts})
    d = load(Val(:panda_nuts))
    d.n_panda = d.success
    d.age_s   = (d.age .- mean(d.age)) ./ std(d.age)
    return [
        ("bf(n_panda ~ seconds * phi * (1 - exp(-k * age_s))^theta, phi + k + theta ~ 1, nl = TRUE)", d),
    ]
end

##############################################################################
# SR2 Ch 16: Lynx and Hare — time series and VAR models
# Source: https://github.com/rmcelreath/rethinking/blob/master/data/Lynx_Hare.csv
#
# Dataset: Lynx_Hare — Canadian lynx and snowshoe hare pelts (Hudson's Bay Co.)
#   21 rows (1900–1920): Year, Hare (thousands of pelts), Lynx (thousands of pelts).
#
# brms model formulas:
#   "Hare ~ 1 + ar(time = Year, p = 1)"
#     (ch16: AR(1) autoregressive model for hare population)
#   "Hare ~ 1 + Hare_1"
#     (ch16: equivalent lag-1 regression; Hare_1 = lag of Hare)
#   "bf(Hare ~ 1 + mi(Hare_1)) + bf(Hare_1 | mi() ~ 1) + set_rescor(FALSE)"
#     (ch16: treating lagged Hare as partially observed/missing)
#   "bf(Hare ~ 0 + Intercept + Hare_1 + Lynx_1) + bf(Lynx ~ 0 + Intercept + Lynx_1 + Hare_1) + set_rescor(FALSE)"
#     (ch16: bivariate VAR(1); lognormal family; Lynx feeds back on Hare)
##############################################################################

load(::Val{:lynx_hare}) =
    CSV.read(Downloads.download(RETHINKING_URL * "Lynx_Hare.csv"), DataFrame)

function examples(::Val{:lynx_hare})
    d        = load(Val(:lynx_hare))
    d.Hare_1 = [missing; d.Hare[1:end-1]]   # lag-1 of hare
    d.Lynx_1 = [missing; d.Lynx[1:end-1]]   # lag-1 of lynx
    d2       = dropmissing(d)
    return [
        ("Hare ~ 1 + ar(time = Year, p = 1)", d),
        ("Hare ~ 1 + Hare_1", d2),
        ("bf(Hare ~ 1 + mi(Hare_1)) + bf(Hare_1 | mi() ~ 1) + set_rescor(FALSE)", d),
        ("bf(Hare ~ 0 + Intercept + Hare_1 + Lynx_1) + bf(Lynx ~ 0 + Intercept + Lynx_1 + Hare_1) + set_rescor(FALSE)", d2),
    ]
end

##############################################################################
# SR2 Ch 17: Blue tit morphology — multivariate Gaussian response
# Dataset: BTdata from MCMCglmm package (also covered in scripts/examples/brms.jl)
# Source: https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv
#
# brms model formulas:
#   "mvbind(tarsus, back) ~ 1"
#     (ch17: intercept-only multivariate; baseline residual correlation)
#   "mvbind(tarsus, back) ~ sex + hatchdate + (1 | p | fosternest) + (1 | q | dam)"
#     (ch17: full model; p/q labels allow correlated random effects across responses)
#   "mvbind(tarsus, back) ~ sex * hatchdate + (1 | p | fosternest) + (1 | q | dam)"
#     (ch17: adds sex × hatchdate interaction)
##############################################################################

const BTDATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MCMCglmm/BTdata.csv"

load(::Val{:btdata}) = CSV.read(Downloads.download(BTDATA_URL), DataFrame)

function examples(::Val{:btdata})
    d = load(Val(:btdata))
    return [
        ("mvbind(tarsus, back) ~ 1", d),
        ("mvbind(tarsus, back) ~ sex + hatchdate + (1 | p | fosternest) + (1 | q | dam)", d),
        ("mvbind(tarsus, back) ~ sex * hatchdate + (1 | p | fosternest) + (1 | q | dam)", d),
    ]
end
