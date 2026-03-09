using CSV, DataFrames, Downloads, Random, Statistics, Dates

# Formulas from: epinowcast and baselinenowcast (Epinowcast Community)
# Sources:
#   epinowcast:      https://package.epinowcast.org
#   baselinenowcast: https://baselinenowcast.epinowcast.org
#
# epinowcast uses a MODULAR formula interface, NOT brms syntax.
# A nowcast model is composed of up to three formula-bearing sub-modules:
#
#   enw_expectation(r = <formula>, observation = <formula>)
#     — expected final counts via geometric random walk or renewal process
#   enw_reference(parametric = <formula>, non_parametric = <formula>, distribution = ...)
#     — parametric or non-parametric reporting delay distribution
#   enw_report(non_parametric = <formula>, structural = ...)
#     — systematic effects on report date (e.g. day-of-week)
#   enw_missing(formula = <formula>)
#     — probability of missing reference dates
#
# Formula strings here concatenate the sub-module specs with '+' for readability.
# baselinenowcast has no formula interface; its specification is represented by
# its key procedural arguments (scale_factor, prop_delay).

# ── Synthetic hospitalization reporting triangle ───────────────────────────────

"""
name: hosp_triangle — Synthetic COVID-19 Hospitalization Reporting Triangle
source: synthetic
----

Synthetic reporting triangle mimicking the epinowcast and baselinenowcast vignettes
(Germany COVID-19 hospitalization data from the Robert Koch Institute). 60 reference
dates with lognormal reporting delays (meanlog=1.8, sdlog=0.5; typical mean ~6-day delay),
max delay 40 days, Poisson true counts with sinusoidal trend (~30–70 cases/day).

Columns: `reference_date` (Date of positive test), `report_date` (Date reported),
`confirm` (cumulative count reported by `report_date` for this `reference_date`).
"""
function load(::Val{:hosp_triangle})
    rng      = MersenneTwister(2024)
    n_ref    = 60
    max_del  = 40
    meanlog  = 1.8
    sdlog    = 0.5
    t0       = Date(2021, 5, 1)
    true_counts = [
        30 + floor(Int, 40 * (0.5 + 0.5 * sin(2π * i / 30))) + rand(rng, 0:10)
        for i in 1:n_ref
    ]
    rows = NamedTuple[]
    for (i, n_true) in enumerate(true_counts)
        ref = t0 + Day(i - 1)
        delays = clamp.(
            floor.(Int, exp.(meanlog .+ sdlog .* randn(rng, n_true))),
            0, max_del,
        )
        cum = 0
        for d in 0:max_del
            cum += count(==(d), delays)
            push!(rows, (; reference_date = ref, report_date = ref + Day(d), confirm = cum))
        end
    end
    return DataFrame(rows)
end

"""
name: hosp_triangle_age — Synthetic Age-Stratified Hospitalization Reporting Triangle
source: synthetic
----

Age-stratified extension of `hosp_triangle` with three age groups (00-17, 18-59, 60+),
each with different true case rates and slightly different mean reporting delays.
Used for hierarchical nowcasting models that allow delay parameters to vary by group.

Columns: `reference_date`, `report_date`, `confirm`, `age_group`
("00-17", "18-59", "60+").
"""
function load(::Val{:hosp_triangle_age})
    rng   = MersenneTwister(2025)
    n_ref = 60
    t0    = Date(2021, 5, 1)
    groups = [
        ("00-17", 10, 1.5, 0.5),   # (label, base_count, meanlog, sdlog)
        ("18-59", 30, 1.8, 0.5),
        ("60+",   20, 2.0, 0.6),
    ]
    max_del = 40
    rows = NamedTuple[]
    for (label, base, meanlog, sdlog) in groups
        for i in 1:n_ref
            n_true = base + floor(Int, base * 0.5 * sin(2π * i / 30)) + rand(rng, 0:5)
            ref    = t0 + Day(i - 1)
            delays = clamp.(
                floor.(Int, exp.(meanlog .+ sdlog .* randn(rng, max(n_true, 0)))),
                0, max_del,
            )
            cum = 0
            for d in 0:max_del
                cum += count(==(d), delays)
                push!(rows, (;
                    reference_date = ref,
                    report_date    = ref + Day(d),
                    confirm        = cum,
                    age_group      = label,
                ))
            end
        end
    end
    return DataFrame(rows)
end

# ── epinowcast models ─────────────────────────────────────────────────────────

"""
name: 'Epinowcast — Basic: Geometric Random Walk + Lognormal Delay'
source: https://package.epinowcast.org/articles/epinowcast.html
example: epinowcast_basic
dataset: hosp_triangle
formula: "enw_expectation(~0 + (1|day)) + enw_reference(~1, dist='lognormal')"
----

Default epinowcast model for nowcasting right-truncated hospitalization counts.
`enw_expectation(~0 + (1|day))` models expected final counts via a geometric random walk
(daily random intercepts on the log scale). `enw_reference(~1, dist='lognormal')` fits a
static lognormal delay distribution shared across all reference dates. Negative-binomial
observation model with overdispersion parameter φ.
"""
function examples(::Val{:enw_basic})
    data = load(Val(:hosp_triangle))
    return ("enw_expectation(~0 + (1|day)) + enw_reference(~1, dist='lognormal')", data)
end

"""
name: Epinowcast — Report Day-of-Week Effects
source: https://package.epinowcast.org/articles/epinowcast.html
example: epinowcast_basic
dataset: hosp_triangle
formula: "enw_expectation(~0 + (1|day)) + enw_reference(~1, dist='lognormal') + enw_report(~(1|day_of_week))"
----

Extends the basic model with a non-parametric report date effect: `enw_report(~(1|day_of_week))`
adds random intercepts for day of the week of the report date, capturing systematic
under-reporting on weekends. All other components identical to `:enw_basic`.
"""
function examples(::Val{:enw_report_dow})
    data = load(Val(:hosp_triangle))
    return (
        "enw_expectation(~0 + (1|day)) + enw_reference(~1, dist='lognormal') + enw_report(~(1|day_of_week))",
        data,
    )
end

"""
name: Epinowcast — Non-parametric Hazard Delay Model
source: https://package.epinowcast.org/articles/model.html
example: epinowcast_basic
dataset: hosp_triangle
formula: "enw_expectation(~0 + (1|day)) + enw_reference(parametric=~0, non_parametric=~0+delay)"
----

Replaces the parametric lognormal delay with a fully non-parametric discrete-time hazard
model. `enw_reference(parametric=~0, non_parametric=~0+delay)` estimates a separate
baseline hazard for each delay value using a Cox proportional hazards formulation.
More flexible than lognormal; useful when the delay distribution is multi-modal or irregular.
"""
function examples(::Val{:enw_np_reference})
    data = load(Val(:hosp_triangle))
    return (
        "enw_expectation(~0 + (1|day)) + enw_reference(parametric=~0, non_parametric=~0+delay)",
        data,
    )
end

"""
name: Epinowcast — Age-Stratified Hierarchical Delays
source: https://package.epinowcast.org/articles/germany-age-stratified-nowcasting.html
example: epinowcast_age
dataset: hosp_triangle_age
formula: "enw_expectation(~0 + (1|day_of_week) + (1|day:.group)) + enw_reference(~1 + (1|age_group)) + enw_report(~(1|day_of_week))"
----

Hierarchical nowcast jointly modeling three age strata. `enw_expectation(...)` includes
day-of-week and daily random effects per group (`.group` notation). `enw_reference(~1 + (1|age_group))`
allows lognormal delay parameters to vary by age group via partial pooling. `enw_report(...)`
accounts for weekend reporting dips shared across groups.
"""
function examples(::Val{:enw_age_reference})
    data = load(Val(:hosp_triangle_age))
    return (
        "enw_expectation(~0 + (1|day_of_week) + (1|day:.group)) + enw_reference(~1 + (1|age_group)) + enw_report(~(1|day_of_week))",
        data,
    )
end

"""
name: Epinowcast — Time-Varying Delay with Weekly Random Walk by Age Group
source: https://package.epinowcast.org/articles/germany-age-stratified-nowcasting.html
example: epinowcast_age
dataset: hosp_triangle_age
formula: "enw_expectation(~0 + (1|day_of_week) + (1|day:.group)) + enw_reference(~1 + (1|age_group) + rw(week, by=age_group)) + enw_report(~(1|day_of_week))"
----

Extends `:enw_age_reference` with time-varying delay: `rw(week, by=age_group)` adds a
weekly random walk on delay parameters independently per age group, capturing temporal
changes in testing and reporting practices (e.g., shifts in test positivity or lab capacity)
that affect different age groups differently.
"""
function examples(::Val{:enw_age_week_reference})
    data = load(Val(:hosp_triangle_age))
    return (
        "enw_expectation(~0 + (1|day_of_week) + (1|day:.group)) + enw_reference(~1 + (1|age_group) + rw(week, by=age_group)) + enw_report(~(1|day_of_week))",
        data,
    )
end

"""
name: Epinowcast — Renewal Process with Rt Estimation
source: https://package.epinowcast.org/articles/single-timeseries-rt-estimation.html
example: epinowcast_rt
dataset: hosp_triangle
formula: "enw_expectation(r=~1+rw(week), generation_time=gt_pmf, observation=~1+(1|day_of_week)) + enw_reference(~1, dist='lognormal')"
----

Mechanistic renewal-process expectation model that jointly nowcasts and estimates the
instantaneous reproduction number Rₜ. `r=~1+rw(week)` specifies a weekly random walk
on log(Rₜ). `generation_time=gt_pmf` provides a fixed discretised generation time PMF
(Gamma-derived). `observation=~1+(1|day_of_week)` models day-of-week ascertainment
variation. `enw_reference(~1, dist='lognormal')` fits a static lognormal reporting delay.
"""
function examples(::Val{:enw_rt_renewal})
    data = load(Val(:hosp_triangle))
    return (
        "enw_expectation(r=~1+rw(week), generation_time=gt_pmf, observation=~1+(1|day_of_week)) + enw_reference(~1, dist='lognormal')",
        data,
    )
end

# ── baselinenowcast models ─────────────────────────────────────────────────────

"""
name: Baseline Nowcast — Empirical Delay Estimation
source: https://baselinenowcast.epinowcast.org/articles/baselinenowcast.html
example: baselinenowcast
dataset: hosp_triangle
formula: "baselinenowcast(scale_factor = 3, prop_delay = 0.5)"
hidden: true
----

Non-Bayesian baseline nowcasting method using empirical delay distributions estimated
from historical data. Not a regression formula — specified by two procedural arguments:
`scale_factor = 3` allocates 3× max-delay time points as training data;
`prop_delay = 0.5` splits training data equally between delay estimation and uncertainty
quantification (retrospective error). Outputs probabilistic draws over final counts.
`baselinenowcast` is designed as a reference implementation for benchmarking Bayesian
nowcasters like `epinowcast`.
"""
function examples(::Val{:bnc_empirical})
    data = load(Val(:hosp_triangle))
    return ("baselinenowcast(scale_factor = 3, prop_delay = 0.5)", data)
end
