using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: epidist (Epinowcast Community)
# Source: https://epidist.epinowcast.org
# Package: https://github.com/epinowcast/epidist
#
# epidist is built on brms and estimates epidemiological delay distributions
# (e.g. symptom-onset-to-case-notification) while correcting for observational
# biases: interval censoring of primary/secondary events and right truncation.
# The formula syntax is brms-compatible; the marginal model uses a custom
# likelihood family (`marginal_lognormal`) from the primarycensored package.

# ── Synthetic outbreak linelist ───────────────────────────────────────────────

"""
name: outbreak_linelist — Simulated Epidemiological Delay Linelist
source: synthetic
----

Synthetic outbreak linelist mimicking the epidist vignette (Epinowcast Community).
Approximately 200 cases with daily-interval-censored primary (symptom onset) and
secondary (case notification) event times over a 25-day observation window.
Delay distribution: Lognormal(meanlog=1.6, sdlog=0.5); exponential epidemic growth.

Columns: `ptime_lwr`, `ptime_upr` (primary event window, days from outbreak start),
`stime_lwr`, `stime_upr` (secondary event window), `obs_time` (observation cutoff = 25),
`delay_lwr` = `stime_lwr - ptime_upr`, `delay_upr` = `stime_upr - ptime_lwr`.
"""
function load(::Val{:outbreak_linelist})
    rng      = MersenneTwister(2024)
    obs_time = 25.0
    meanlog  = 1.6
    sdlog    = 0.5
    rows     = NamedTuple[]
    attempts = 0
    while length(rows) < 200 && attempts < 5_000
        attempts += 1
        ptime     = rand(rng) * obs_time
        delay     = exp(meanlog + sdlog * randn(rng))
        stime     = ptime + delay
        stime > obs_time && continue
        ptime_lwr = floor(ptime)
        ptime_upr = ptime_lwr + 1.0
        stime_lwr = floor(stime)
        stime_upr = stime_lwr + 1.0
        push!(rows, (;
            ptime_lwr,
            ptime_upr,
            stime_lwr,
            stime_upr,
            obs_time,
            delay_lwr = max(0.0, stime_lwr - ptime_upr),
            delay_upr = stime_upr - ptime_lwr,
        ))
    end
    return DataFrame(rows)
end

"""
name: outbreak_aggregated — Aggregated Delay Data for epidist Models
source: synthetic
----

Aggregated form of `outbreak_linelist`: unique (delay_lwr, delay_upr, pwindow, swindow,
relative_obs_time) combinations with case counts `n`. This is the format consumed by
epidist's naive and marginal lognormal model families.

Columns: `delay_lwr`, `delay_upr`, `pwindow` (=1, primary censoring window),
`swindow` (=1, secondary censoring window), `relative_obs_time` (= obs_time − delay_lwr),
`n` (case count per unique delay cell).
"""
function load(::Val{:outbreak_aggregated})
    ll = load(Val(:outbreak_linelist))
    ll.pwindow              .= 1.0
    ll.swindow              .= 1.0
    ll.relative_obs_time    = ll.obs_time .- ll.delay_lwr
    agg = combine(
        groupby(ll, [:delay_lwr, :delay_upr, :relative_obs_time, :pwindow, :swindow]),
        nrow => :n,
    )
    return agg
end

# ── epidist models ─────────────────────────────────────────────────────────────

"""
name: Delay Distribution — Naive Lognormal (No Bias Correction)
source: https://epidist.epinowcast.org/articles/epidist.html
example: outbreak_delays
dataset: outbreak_aggregated
formula: "delay | weights(n) ~ 1"
----

Naive lognormal regression for delay distribution estimation (e.g. symptom onset to
case notification). Does not account for interval censoring or right truncation of
secondary events, producing estimates biased toward shorter delays. The `sigma`
sub-model (`sigma ~ 1`) estimates the lognormal standard deviation on the log scale.
Baseline model; compare with `:delay_marginal`.
"""
function examples(::Val{:delay_naive})
    data = load(Val(:outbreak_aggregated))
    return ("delay | weights(n) ~ 1", data)
end

"""
name: Delay Distribution — Marginal Lognormal (Censoring + Truncation Corrected)
source: https://epidist.epinowcast.org/articles/epidist.html
example: outbreak_delays
dataset: outbreak_aggregated
formula: "delay_lwr | weights(n) + vreal(relative_obs_time, pwindow, swindow, delay_upr) ~ 1"
----

Marginal lognormal model correcting for double interval censoring (primary and secondary
event times observed only within daily windows) and right truncation (only cases with
secondary event before `obs_time` are observable). Uses a custom `marginal_lognormal`
brms family implemented via the `primarycensored` Stan library.

`vreal()` passes auxiliary real-valued data to the custom likelihood: observation cutoff,
primary censoring window, secondary censoring window, and upper delay bound. The likelihood
integrates over primary event time within the censoring window.
"""
function examples(::Val{:delay_marginal})
    data = load(Val(:outbreak_aggregated))
    return (
        "delay_lwr | weights(n) + vreal(relative_obs_time, pwindow, swindow, delay_upr) ~ 1",
        data,
    )
end
