using CSV, DataFrames, Downloads, Random, Statistics

# Formulas from: flocker — Flexible Occupancy Estimation
# Source: https://jsocolar.github.io/flocker/
# Package: https://github.com/jsocolar/flocker
#
# flocker uses brms as a backend to fit Bayesian occupancy models. Formulas are
# split across sub-models using named arguments to flock():
#   f_occ  — log-odds of site occupancy
#   f_det  — log-odds of detection given presence
#   f_col  — log-odds of colonisation (multi-season)
#   f_ex   — log-odds of extinction (multi-season)
#   f_auto — autologistic persistence offset (autologistic multi-season)
#
# Formula notation here: semicolon-separated labelled sub-models, e.g.
#   "occ: ~ uc1 + (1 | sp); det: ~ uc1 + ec1 + (1 | sp)"
#
# uc1 = unit-level (site) covariate, constant across visits
# ec1 = event-level (visit) covariate, varying across visits

# ── Utility ───────────────────────────────────────────────────────────────────

_logistic(x) = 1 / (1 + exp(-x))

# ── Single-season rep-varying data ───────────────────────────────────────────

"""
name: fd_rep_varying — Simulated Single-Season Multi-Species Detection Data
source: synthetic
----

Simulated single-season occupancy data: 100 sites × 4 visits × 10 species,
with a unit-level covariate `uc1` (site characteristic, drawn N(0,1)) and an
event-level covariate `ec1` (visit characteristic, drawn N(0,1)).

True occupancy log-odds: `0.5 + 0.8 * uc1 + species_re` (species RE ~ N(0, 0.8²)).
True detection log-odds: `0.2 + 0.5 * uc1 + 0.7 * ec1 + species_re_det`.

Each row is one visit (site × visit × species combination).
Columns: `site` (Int), `visit` (Int 1–4), `species` (String "sp01"–"sp10"),
`y` (detection, 0/1), `uc1` (site covariate), `ec1` (visit covariate).
"""
function load(::Val{:fd_rep_varying})
    rng        = MersenneTwister(4242)
    n_sites    = 100
    n_visits   = 4
    n_species  = 10
    uc1        = randn(rng, n_sites)                         # site-level covariate
    sp_occ_re  = 0.8 * randn(rng, n_species)                # species occupancy RE
    sp_det_re  = 0.6 * randn(rng, n_species)                # species detection RE
    rows = NamedTuple[]
    for sp in 1:n_species
        for site in 1:n_sites
            psi = _logistic(0.5 + 0.8 * uc1[site] + sp_occ_re[sp])
            z   = rand(rng) < psi ? 1 : 0               # true occupancy
            for visit in 1:n_visits
                ec1 = randn(rng)
                p   = z == 1 ? _logistic(0.2 + 0.5 * uc1[site] + 0.7 * ec1 + sp_det_re[sp]) : 0.0
                push!(rows, (;
                    site,
                    visit,
                    species = lpad(sp, 2, "0") |> s -> "sp$s",
                    y       = rand(rng) < p ? 1 : 0,
                    uc1     = uc1[site],
                    ec1,
                ))
            end
        end
    end
    return DataFrame(rows)
end

"""
name: Single-Season Occupancy — Multi-Species with Event Covariates
source: https://jsocolar.github.io/flocker/articles/flocker_tutorial.html
example: single_season
dataset: fd_rep_varying
formula: "occ: ~ uc1 + (1 + uc1 | species); det: ~ uc1 + ec1 + (1 + uc1 + ec1 | species)"
----

Single-season multi-species occupancy model with site-level (`uc1`) and
visit-level (`ec1`) covariates.

- `occ` sub-model: log-odds of occupancy at a site; `uc1` as fixed slope and
  random slope by species.
- `det` sub-model: log-odds of detection given presence; both `uc1` and the
  event-varying `ec1` with random slopes by species.

Species-level random effects pool information across the 10 species. The model
returns a `brmsfit` object; all brms post-processing tools apply.

flocker call: `flock(f_occ = ~ uc1 + (1 + uc1 | species), f_det = ~ uc1 + ec1 + (1 + uc1 + ec1 | species), flocker_data = fd)`
"""
function examples(::Val{:single_season_repvarying})
    data = load(Val(:fd_rep_varying))
    return (
        "occ: ~ uc1 + (1 + uc1 | species); det: ~ uc1 + ec1 + (1 + uc1 + ec1 | species)",
        data,
    )
end

"""
name: Single-Season Occupancy — Multi-Species, No Event Covariates
source: https://jsocolar.github.io/flocker/articles/flocker_tutorial.html
example: single_season
dataset: fd_rep_varying
formula: "occ: ~ uc1 + (1 + uc1 | species); det: ~ uc1 + (1 + uc1 | species)"
----

Single-season multi-species occupancy model with only site-level (`uc1`)
covariates in both occupancy and detection sub-models (no event-varying
covariate). Detection probability is constant across visits within a site,
enabling an efficient "rep-constant" parameterisation in flocker.

flocker call: `flock(f_occ = ~ uc1 + (1 + uc1 | species), f_det = ~ uc1 + (1 + uc1 | species), flocker_data = fd, rep_constant = TRUE)`
"""
function examples(::Val{:single_season_repconstant})
    data = load(Val(:fd_rep_varying))
    return (
        "occ: ~ uc1 + (1 + uc1 | species); det: ~ uc1 + (1 + uc1 | species)",
        data,
    )
end

# ── Multi-season data ─────────────────────────────────────────────────────────

"""
name: fd_multi — Simulated Multi-Season Occupancy Data
source: synthetic
----

Simulated dynamic occupancy data: 100 sites × 3 seasons × 4 visits, single
species. Site-level covariate `uc1` ~ N(0,1); visit-level `ec1` ~ N(0,1).

True initial occupancy: logit(ψ₁) = 0.3 + 0.8 * uc1.
True colonisation:      logit(γ)  = −1.0 + 0.6 * uc1.
True extinction:        logit(ε)  = −0.5 − 0.4 * uc1.
True detection:         logit(p)  = 0.4 + 0.5 * uc1 + 0.6 * ec1.

Each row is one visit (site × season × visit).
Columns: `site` (Int), `season` (Int 1–3), `visit` (Int 1–4), `y` (detection),
`uc1` (site covariate), `ec1` (visit covariate).
"""
function load(::Val{:fd_multi})
    rng      = MersenneTwister(1234)
    n_sites  = 100
    n_season = 3
    n_visits = 4
    uc1      = randn(rng, n_sites)
    rows     = NamedTuple[]
    for site in 1:n_sites
        z = rand(rng) < _logistic(0.3 + 0.8 * uc1[site]) ? 1 : 0
        for season in 1:n_season
            if season > 1
                if z == 0
                    z = rand(rng) < _logistic(-1.0 + 0.6 * uc1[site]) ? 1 : 0  # colonise
                else
                    z = rand(rng) < _logistic(-0.5 - 0.4 * uc1[site]) ? 0 : 1  # persist
                end
            end
            for visit in 1:n_visits
                ec1 = randn(rng)
                p   = z == 1 ? _logistic(0.4 + 0.5 * uc1[site] + 0.6 * ec1) : 0.0
                push!(rows, (; site, season, visit,
                    y   = rand(rng) < p ? 1 : 0,
                    uc1 = uc1[site],
                    ec1,
                ))
            end
        end
    end
    return DataFrame(rows)
end

"""
name: Colonisation-Extinction — Explicit Initial Occupancy
source: https://jsocolar.github.io/flocker/articles/flocker_tutorial.html
example: colex_multiseason
dataset: fd_multi
formula: "occ: ~ uc1; det: ~ uc1 + ec1; col: ~ uc1; ex: ~ uc1"
----

Dynamic (multi-season) colonisation-extinction occupancy model with explicit
estimation of initial occupancy. Four sub-models:
- `occ` — initial occupancy at season 1 (log-odds)
- `det` — detection given presence (log-odds; with event covariate)
- `col` — colonisation probability (log-odds): unoccupied sites becoming occupied
- `ex`  — extinction probability (log-odds): occupied sites becoming unoccupied

`multi_init = "explicit"` in flocker: initial occupancy estimated freely.

flocker call: `flock(f_occ=~uc1, f_det=~uc1+ec1, f_col=~uc1, f_ex=~uc1, flocker_data=fd, multiseason="colex", multi_init="explicit")`
"""
function examples(::Val{:colex_explicit})
    data = load(Val(:fd_multi))
    return ("occ: ~ uc1; det: ~ uc1 + ec1; col: ~ uc1; ex: ~ uc1", data)
end

"""
name: Colonisation-Extinction — Equilibrium Initial Occupancy
source: https://jsocolar.github.io/flocker/articles/flocker_tutorial.html
example: colex_multiseason
dataset: fd_multi
formula: "det: ~ uc1 + ec1; col: ~ uc1; ex: ~ uc1"
----

Dynamic colonisation-extinction occupancy model where initial occupancy is
derived from the colonisation/extinction equilibrium rather than estimated as
a free parameter. No `occ` sub-model; initial occupancy = col / (col + ex).

`multi_init = "equilibrium"` in flocker. Reduces parameters and enforces
stationarity of the occupancy process.

flocker call: `flock(f_det=~uc1+ec1, f_col=~uc1, f_ex=~uc1, flocker_data=fd, multiseason="colex", multi_init="equilibrium")`
"""
function examples(::Val{:colex_equilibrium})
    data = load(Val(:fd_multi))
    return ("det: ~ uc1 + ec1; col: ~ uc1; ex: ~ uc1", data)
end

"""
name: Autologistic Occupancy — Equilibrium Initial Occupancy
source: https://jsocolar.github.io/flocker/articles/flocker_tutorial.html
example: autologistic_multiseason
dataset: fd_multi
formula: "det: ~ uc1 + ec1; col: ~ uc1; auto: ~ 1"
----

Autologistic dynamic occupancy model: colonisation and persistence share
predictor structure; a constant logit-scale offset (`auto: ~ 1`) is added to
persistence probability relative to colonisation. Initial occupancy at
equilibrium.

`multiseason = "autologistic"`, `multi_init = "equilibrium"` in flocker.
The autologistic formulation is more parsimonious than explicit colex when
persistence and colonisation share the same predictors.

flocker call: `flock(f_det=~uc1+ec1, f_col=~uc1, f_auto=~1, flocker_data=fd, multiseason="autologistic", multi_init="equilibrium")`
"""
function examples(::Val{:autologistic_equilibrium})
    data = load(Val(:fd_multi))
    return ("det: ~ uc1 + ec1; col: ~ uc1; auto: ~ 1", data)
end

# ── Data-augmented multi-species data ─────────────────────────────────────────

"""
name: fd_augmented — Simulated Data-Augmented Species Richness Data
source: synthetic
----

Simulated data-augmented occupancy dataset for estimating total species
richness. 50 real species observed at 50 sites × 4 visits; 50 additional
"pseudospecies" (all zeros) appended for the data augmentation approach.

`ff_species` is flocker's reserved keyword for the species variable in
augmented models.

Columns: `site` (Int), `visit` (Int 1–4), `ff_species` (String "sp01"–"sp50"
and "ps01"–"ps50"), `y` (detection 0/1; always 0 for pseudospecies), `uc1`
(site covariate), `ec1` (visit covariate).
"""
function load(::Val{:fd_augmented})
    rng           = MersenneTwister(5050)
    n_sites       = 50
    n_visits      = 4
    n_real_sp     = 50
    n_pseudo_sp   = 50
    uc1           = randn(rng, n_sites)
    sp_occ_re     = 0.7 * randn(rng, n_real_sp)
    sp_det_re     = 0.5 * randn(rng, n_real_sp)
    rows = NamedTuple[]
    # Real species
    for sp in 1:n_real_sp
        for site in 1:n_sites
            psi = _logistic(0.0 + 0.6 * uc1[site] + sp_occ_re[sp])
            z   = rand(rng) < psi ? 1 : 0
            for visit in 1:n_visits
                ec1 = randn(rng)
                p   = z == 1 ? _logistic(0.0 + 0.4 * uc1[site] + 0.5 * ec1 + sp_det_re[sp]) : 0.0
                push!(rows, (;
                    site,
                    visit,
                    ff_species = "sp$(lpad(sp, 2, '0'))",
                    y  = rand(rng) < p ? 1 : 0,
                    uc1 = uc1[site],
                    ec1,
                ))
            end
        end
    end
    # Pseudo-species (all undetected)
    for sp in 1:n_pseudo_sp
        for site in 1:n_sites
            for visit in 1:n_visits
                ec1 = randn(rng)
                push!(rows, (;
                    site,
                    visit,
                    ff_species = "ps$(lpad(sp, 2, '0'))",
                    y  = 0,
                    uc1 = uc1[site],
                    ec1,
                ))
            end
        end
    end
    return DataFrame(rows)
end

"""
name: Data-Augmented Occupancy — Species Richness Estimation
source: https://jsocolar.github.io/flocker/articles/flocker_tutorial.html
example: augmented_richness
dataset: fd_augmented
formula: "occ: ~ (1 | ff_species); det: ~ uc1 + ec1 + (1 + uc1 + ec1 | ff_species)"
----

Data augmentation approach for estimating total species richness from
incomplete sampling. The dataset includes 50 observed species plus 50
pseudospecies (all-zero detection histories); the occupancy probability of
pseudospecies reflects the probability that an unobserved species is actually
present.

`ff_species` is flocker's reserved grouping keyword for species in augmented
models. Occupancy random intercept by species captures species-level variation
in commonness; detection random slopes by species capture variation in
detectability.

flocker call: `flock(f_occ=~(1|ff_species), f_det=~uc1+ec1+(1+uc1+ec1|ff_species), flocker_data=fd, augmented=TRUE)`
"""
function examples(::Val{:augmented_multispecies})
    data = load(Val(:fd_augmented))
    return (
        "occ: ~ (1 | ff_species); det: ~ uc1 + ec1 + (1 + uc1 + ec1 | ff_species)",
        data,
    )
end
