# scripts/examples/all.jl
#
# Entry point for all regression examples.
#
# Each entry in the CATALOG corresponds to exactly one (source, example, dataset, formula)
# combination, matching the docstring structure on every `examples()` function.
#
# Usage:
#   include("scripts/examples/all.jl")
#   list_examples()                        # print full catalog
#   list_examples(:brms)                   # filter by source
#   formula, data = get_examples(:brms, :cbpp_binomial)
#   doc            = describe(:brms, :cbpp_binomial)
#   for (source, key, formula, data) in all_examples()
#       println(source, " / ", key, " : ", formula)
#   end
#
# Sources:
#   :bambi          — Bambi (Python) model gallery
#   :brms           — brms vignettes (Bürkner)
#   :mcelreath      — Statistical Rethinking 2e (Kurz brms translation)
#   :kruschke       — DBDA2 (Kurz brms translation)
#   :burkner_papers — Bürkner JSS 2017 + R Journal 2018
#   :vasishth       — Bayesian Data Analysis for Cognitive Science

# ── Each source in its own module to avoid name clashes ─────────────────────

module _Bambi;         include("bambi.jl");          end
module _Brms;          include("brms.jl");           end
module _McElreath;     include("mcelreath.jl");      end
module _Kruschke;      include("kruschke.jl");       end
module _BurknerPapers; include("burkner_papers.jl"); end
module _Vasishth;      include("vasishth.jl");       end

const _SOURCE_MODULES = Dict{Symbol, Module}(
    :bambi          => _Bambi,
    :brms           => _Brms,
    :mcelreath      => _McElreath,
    :kruschke       => _Kruschke,
    :burkner_papers => _BurknerPapers,
    :vasishth       => _Vasishth,
)

# ── Catalog ───────────────────────────────────────────────────────────────────
# Each entry is one (source, key) pair mapping to exactly one examples() function,
# which returns a single (formula, data) tuple.
# Keys follow the convention: <dataset>_<variant> or <dataset> for single-formula datasets.
# The docstring on each examples() function carries structured metadata:
#   name, source, example, dataset, formula (and optionally chapter, verified).

const CATALOG = [

    # ── bambi ─────────────────────────────────────────────────────────────────

    # Multiple Linear Regression
    (source=:bambi, key=:escs),

    # Regression Splines
    (source=:bambi, key=:cherry_blossoms_explicit),
    (source=:bambi, key=:cherry_blossoms_absorbed),

    # Hierarchical Linear Regression
    (source=:bambi, key=:dietox),
    (source=:bambi, key=:sleepstudy),

    # Radon Multilevel Models
    (source=:bambi, key=:radon_pooled),
    (source=:bambi, key=:radon_nopooling),
    (source=:bambi, key=:radon_partial),
    (source=:bambi, key=:radon_floor),
    (source=:bambi, key=:radon_slopes),
    (source=:bambi, key=:radon_county_pred),

    # Strack RRR Reanalysis
    (source=:bambi, key=:strack_rrr_simple),
    (source=:bambi, key=:strack_rrr_full),

    # Shooter Task
    (source=:bambi, key=:shooter_rate_simple),
    (source=:bambi, key=:shooter_rate_target),
    (source=:bambi, key=:shooter_binary),

    # Fixed vs Random Effects
    (source=:bambi, key=:fixed_random_naive),
    (source=:bambi, key=:fixed_random_fe),
    (source=:bambi, key=:fixed_random_re),
    (source=:bambi, key=:fixed_random_mundlak),

    # Robust Regression
    (source=:bambi, key=:t_regression_gaussian),
    (source=:bambi, key=:t_regression_t),

    # Predict New Groups
    (source=:bambi, key=:predict_new_groups),

    # Polynomial Regression
    (source=:bambi, key=:poly_falling_explicit),
    (source=:bambi, key=:poly_falling_alt),
    (source=:bambi, key=:poly_falling_precomp),
    (source=:bambi, key=:poly_projectile_power),
    (source=:bambi, key=:poly_projectile_poly),
    (source=:bambi, key=:poly_planets),

    # Logistic Regression
    (source=:bambi, key=:logistic_anes),

    # Model Comparison
    (source=:bambi, key=:model_comparison_linear),
    (source=:bambi, key=:model_comparison_quad),
    (source=:bambi, key=:model_comparison_cubic),

    # Hierarchical Binomial
    (source=:bambi, key=:hierarchical_binomial_nopooling),
    (source=:bambi, key=:hierarchical_binomial_partial),

    # Alternative Link Functions
    (source=:bambi, key=:alternative_links_logit),
    (source=:bambi, key=:alternative_links_probit),
    (source=:bambi, key=:alternative_links_cloglog),

    # Wald and Gamma
    (source=:bambi, key=:wald_gamma_wald),
    (source=:bambi, key=:wald_gamma_gamma),

    # Negative Binomial
    (source=:bambi, key=:negative_binomial_main),
    (source=:bambi, key=:negative_binomial_interaction),

    # Count / Roaches
    (source=:bambi, key=:count_roaches_poisson),
    (source=:bambi, key=:count_roaches_nb),

    # Beta Regression
    (source=:bambi, key=:beta_probs_intercept),
    (source=:bambi, key=:beta_coin),
    (source=:bambi, key=:beta_batting_intercept),
    (source=:bambi, key=:beta_batting_shift),

    # Categorical Regression
    (source=:bambi, key=:categorical_toy),
    (source=:bambi, key=:categorical_iris),
    (source=:bambi, key=:categorical_alligator),

    # Circular Regression
    (source=:bambi, key=:circular_vonmises),
    (source=:bambi, key=:circular_gaussian),

    # Quantile Regression
    (source=:bambi, key=:quantile_p10),
    (source=:bambi, key=:quantile_p50),
    (source=:bambi, key=:quantile_p90),
    (source=:bambi, key=:quantile_gaussian),

    # MrP
    (source=:bambi, key=:mister_p),

    # Zero-Inflated / Hurdle
    (source=:bambi, key=:zip_mu),
    (source=:bambi, key=:zip_psi),
    (source=:bambi, key=:hurdle_mu),
    (source=:bambi, key=:hurdle_psi),

    # Ordinal Regression
    (source=:bambi, key=:ordinal_trolley_intercept),
    (source=:bambi, key=:ordinal_trolley_effects),
    (source=:bambi, key=:ordinal_hr_years),

    # Distributional Models
    (source=:bambi, key=:distributional_const_alpha),
    (source=:bambi, key=:distributional_var_alpha),
    (source=:bambi, key=:distributional_bikes),

    # Hilbert-Space GP (1D)
    (source=:bambi, key=:hsgp_1d_basic),
    (source=:bambi, key=:hsgp_1d_centered),
    (source=:bambi, key=:hsgp_1d_by_group),
    (source=:bambi, key=:hsgp_1d_nocov),

    # Hilbert-Space GP (2D)
    (source=:bambi, key=:hsgp_2d_iso),
    (source=:bambi, key=:hsgp_2d_by_group),
    (source=:bambi, key=:hsgp_2d_nocov),
    (source=:bambi, key=:hsgp_2d_aniso),
    (source=:bambi, key=:hsgp_2d_poisson),

    # Survival: Accelerated Failure Time
    (source=:bambi, key=:survival_intercept),
    (source=:bambi, key=:survival_color),

    # Survival: Discrete Time
    (source=:bambi, key=:survival_disc_sim),
    (source=:bambi, key=:survival_disc_binomial),
    (source=:bambi, key=:survival_disc_spline),
    (source=:bambi, key=:survival_disc_poisson),

    # Survival: Continuous Time
    (source=:bambi, key=:survival_cont_weibull),
    (source=:bambi, key=:survival_cont_retention_fe),
    (source=:bambi, key=:survival_cont_retention_re),

    # Orthogonal Polynomial
    (source=:bambi, key=:orthopoly_explicit),
    (source=:bambi, key=:orthopoly_poly),
    (source=:bambi, key=:orthopoly_linear),
    (source=:bambi, key=:orthopoly_quad),
    (source=:bambi, key=:orthopoly_compare),

    # Plot Predictions
    (source=:bambi, key=:plot_pred_linear),
    (source=:bambi, key=:plot_pred_nb),
    (source=:bambi, key=:plot_pred_logistic),
    (source=:bambi, key=:plot_pred_distr),

    # Plot Comparisons
    (source=:bambi, key=:plot_comp_zip),
    (source=:bambi, key=:plot_comp_logistic),

    # Plot Slopes
    (source=:bambi, key=:plot_slopes_main),
    (source=:bambi, key=:plot_slopes_interaction),

    # Alternative Samplers
    (source=:bambi, key=:alternative_samplers),

    # ── brms vignettes ────────────────────────────────────────────────────────

    # Custom Families
    (source=:brms, key=:cbpp_binomial),
    (source=:brms, key=:cbpp_beta_binomial),

    # Distributional Regression
    (source=:brms, key=:distreg_normal),
    (source=:brms, key=:distreg_nb),
    (source=:brms, key=:distreg_zip),
    (source=:brms, key=:distreg_gam),

    # Handle Missing Values
    (source=:brms, key=:nhanes_imputed),
    (source=:brms, key=:nhanes_joint),
    (source=:brms, key=:nhanes_error),

    # Monotonic Effects
    (source=:brms, key=:income_mo),
    (source=:brms, key=:income_num),
    (source=:brms, key=:income_nominal),
    (source=:brms, key=:income_mo_age),
    (source=:brms, key=:income_mo_city),

    # Multivariate Models
    (source=:brms, key=:btdata_compact),
    (source=:brms, key=:btdata_explicit),
    (source=:brms, key=:btdata_spline),

    # Nonlinear Models
    (source=:brms, key=:nonlinear_exp),
    (source=:brms, key=:nonlinear_linear),
    (source=:brms, key=:nonlinear_loss),

    # Phylogenetic Models
    (source=:brms, key=:phylo_simple_re),
    (source=:brms, key=:phylo_simple_resid),
    (source=:brms, key=:phylo_repeat_re),
    (source=:brms, key=:phylo_effect),
    (source=:brms, key=:phylo_pois),

    # ── McElreath / Kurz SR2 ──────────────────────────────────────────────────

    # Globe Tossing
    (source=:mcelreath, key=:globe),

    # Heights and Weights (Howell)
    (source=:mcelreath, key=:howell1_intercept),
    (source=:mcelreath, key=:howell1_linear),
    (source=:mcelreath, key=:howell1_quad),
    (source=:mcelreath, key=:howell1_cubic),
    (source=:mcelreath, key=:howell1_nl),
    (source=:mcelreath, key=:howell1_volume),

    # Cherry Blossoms
    (source=:mcelreath, key=:cherry_blossoms),

    # Waffle Divorce
    (source=:mcelreath, key=:waffle_divorce_a),
    (source=:mcelreath, key=:waffle_divorce_m),
    (source=:mcelreath, key=:waffle_divorce_am),
    (source=:mcelreath, key=:waffle_divorce_mediator),
    (source=:mcelreath, key=:waffle_divorce_multivariate),
    (source=:mcelreath, key=:waffle_divorce_meas_err),

    # Milk Energy
    (source=:mcelreath, key=:milk_neo),
    (source=:mcelreath, key=:milk_mass),
    (source=:mcelreath, key=:milk_both),
    (source=:mcelreath, key=:milk_clade),
    (source=:mcelreath, key=:milk_mi),

    # Plant Growth
    (source=:mcelreath, key=:plant_growth_baseline),
    (source=:mcelreath, key=:plant_growth_biased),
    (source=:mcelreath, key=:plant_growth_causal),

    # Happiness
    (source=:mcelreath, key=:happiness_collider),
    (source=:mcelreath, key=:happiness_causal),

    # Rugged Terrain
    (source=:mcelreath, key=:rugged_pooled),
    (source=:mcelreath, key=:rugged_continent),
    (source=:mcelreath, key=:rugged_nl),

    # Tulips
    (source=:mcelreath, key=:tulips_additive),
    (source=:mcelreath, key=:tulips_interaction),

    # Heteroscedastic
    (source=:mcelreath, key=:hetero),

    # Chimpanzees
    (source=:mcelreath, key=:chimpanzees_intercept),
    (source=:mcelreath, key=:chimpanzees_actors),
    (source=:mcelreath, key=:chimpanzees_multilevel),
    (source=:mcelreath, key=:chimpanzees_slopes),

    # UCB Admissions
    (source=:mcelreath, key=:ucbadmit_gender),
    (source=:mcelreath, key=:ucbadmit_dept),
    (source=:mcelreath, key=:ucbadmit_beta_binomial),

    # Kline Tool Use
    (source=:mcelreath, key=:kline_intercept),
    (source=:mcelreath, key=:kline_contact),
    (source=:mcelreath, key=:kline_power),

    # Kline with Phylogeny
    (source=:mcelreath, key=:kline2),

    # Trolley
    (source=:mcelreath, key=:trolley_intercept),
    (source=:mcelreath, key=:trolley_effects),
    (source=:mcelreath, key=:trolley_edu),

    # Reed Frogs
    (source=:mcelreath, key=:reedfrogs_nopooling),
    (source=:mcelreath, key=:reedfrogs_partial),

    # Café
    (source=:mcelreath, key=:cafe),

    # Primates 301
    (source=:mcelreath, key=:primates301_ols),
    (source=:mcelreath, key=:primates301_phylo),

    # Moralizing Gods
    (source=:mcelreath, key=:moralizing_gods),

    # Panda Nuts
    (source=:mcelreath, key=:panda_nuts),

    # Lynx-Hare
    (source=:mcelreath, key=:lynx_hare_ar),
    (source=:mcelreath, key=:lynx_hare_lag),
    (source=:mcelreath, key=:lynx_hare_mi),
    (source=:mcelreath, key=:lynx_hare_var),

    # Blue Tit (McElreath variant)
    (source=:mcelreath, key=:mcelreath_btdata_intercept),
    (source=:mcelreath, key=:mcelreath_btdata_full),
    (source=:mcelreath, key=:mcelreath_btdata_interaction),

    # ── Kruschke / Kurz DBDA2 ─────────────────────────────────────────────────

    # Coin Flips
    (source=:kruschke, key=:z15n50),
    (source=:kruschke, key=:z6n8z2n7),

    # Therapeutic Touch
    (source=:kruschke, key=:therapeutic_touch),

    # MLB Batting Average
    (source=:kruschke, key=:batting_average),

    # Memory Recall
    (source=:kruschke, key=:recall_conditions),
    (source=:kruschke, key=:recall_pooled),

    # Two-Group IQ
    (source=:kruschke, key=:two_group_iq_baseline),
    (source=:kruschke, key=:two_group_iq_hetero),

    # Calcium RCT
    (source=:kruschke, key=:calcium),

    # Height and Weight
    (source=:kruschke, key=:htwt),
    (source=:kruschke, key=:hier_linreg),
    (source=:kruschke, key=:income_famsize),

    # Guber 1999 SAT
    (source=:kruschke, key=:guber1999_base),
    (source=:kruschke, key=:guber1999_complement),
    (source=:kruschke, key=:guber1999_interaction),

    # Fruitfly
    (source=:kruschke, key=:fruitfly_hierarchical),
    (source=:kruschke, key=:fruitfly_pooled),
    (source=:kruschke, key=:fruitfly_robust),
    (source=:kruschke, key=:fruitfly_ancova),
    (source=:kruschke, key=:fruitfly_anhecova),

    # Salary
    (source=:kruschke, key=:salary_anova),
    (source=:kruschke, key=:salary_robust),

    # Split-Plot
    (source=:kruschke, key=:splitplot_field),
    (source=:kruschke, key=:splitplot_nofield),

    # Height/Weight n=110
    (source=:kruschke, key=:htwt110_single),
    (source=:kruschke, key=:htwt110_two_pred),
    (source=:kruschke, key=:htwt110_robust),

    # Softmax / Multinomial
    (source=:kruschke, key=:softmax_categorical),
    (source=:kruschke, key=:softmax_baseline),
    (source=:kruschke, key=:condlog1_ordinal),
    (source=:kruschke, key=:condlog2_ordinal),

    # Ordinal Probit
    (source=:kruschke, key=:ordinal_probit_intercept),
    (source=:kruschke, key=:ordinal_probit_disc),
    (source=:kruschke, key=:ordinal_probit_hetero),

    # Happiness and Assets
    (source=:kruschke, key=:happiness_assets),

    # Movies
    (source=:kruschke, key=:movies),

    # Hair-Eye Contingency
    (source=:kruschke, key=:haireye_poisson),
    (source=:kruschke, key=:haireye_binomial),

    # Censored Data
    (source=:kruschke, key=:censored_missing),
    (source=:kruschke, key=:censored_lr),
    (source=:kruschke, key=:censored_interval),

    # ── Bürkner papers ────────────────────────────────────────────────────────

    # Epilepsy
    (source=:burkner_papers, key=:epilepsy_base),
    (source=:burkner_papers, key=:epilepsy_trunc),
    (source=:burkner_papers, key=:epilepsy_simple),

    # Inhaler
    (source=:burkner_papers, key=:inhaler),

    # Kidney
    (source=:burkner_papers, key=:kidney),

    # Heteroscedastic
    (source=:burkner_papers, key=:hetero_jss_sigma),
    (source=:burkner_papers, key=:hetero_jss_quantile),

    # Fish Counts
    (source=:burkner_papers, key=:fish_rj_poisson),
    (source=:burkner_papers, key=:fish_rj_zip),

    # Munich Rent
    (source=:burkner_papers, key=:rent99_spline),
    (source=:burkner_papers, key=:rent99_distr),

    # Loss Development
    (source=:burkner_papers, key=:loss_rj),

    # Multiple Membership
    (source=:burkner_papers, key=:multi_member_equal),
    (source=:burkner_papers, key=:multi_member_weighted),

    # ── Vasishth / Nicenboim / Schad ──────────────────────────────────────────

    # Pupil Dilation
    (source=:vasishth, key=:pupil),

    # Spacebar RT
    (source=:vasishth, key=:spacebar),

    # Working Memory Recall
    (source=:vasishth, key=:recall_wm),

    # N400 ERP
    (source=:vasishth, key=:n400_uncorr),
    (source=:vasishth, key=:n400_corr),
    (source=:vasishth, key=:n400_crossed),
    (source=:vasishth, key=:n400_distr),

    # Stroop
    (source=:vasishth, key=:stroop),

    # Pooling
    (source=:vasishth, key=:pooling_partial),
    (source=:vasishth, key=:pooling_complete),
    (source=:vasishth, key=:pooling_none),

    # Contrasts (1 factor)
    (source=:vasishth, key=:contrasts1_treatment),
    (source=:vasishth, key=:contrasts1_cellmeans),
    (source=:vasishth, key=:contrasts1_monotonic),

    # Contrasts (2×2)
    (source=:vasishth, key=:contrasts2x2_factorial),
    (source=:vasishth, key=:contrasts2x2_nested),
    (source=:vasishth, key=:contrasts2x2_logistic),

    # Meta-Analysis
    (source=:vasishth, key=:meta_sbi),

    # Individual Differences
    (source=:vasishth, key=:indiv_diff_naive),
    (source=:vasishth, key=:indiv_diff_me),
]

# ── Register each entry as a documented get_examples method ──────────────────
#
# Docstrings are forwarded from each source module's examples() function, so:
#   @doc get_examples(Val(:mcelreath), Val(:howell1_linear))
# shows the same docstring as the one in mcelreath.jl.

for _entry in CATALOG
    _src, _k = _entry.source, _entry.key
    _mod = _SOURCE_MODULES[_src]
    _sig = Core.apply_type(Tuple, Core.apply_type(Val, _k))
    _doc = Base.Docs.doc(Base.Docs.Binding(_mod, :examples), _sig)
    @eval @doc $_doc function get_examples(::Val{$(QuoteNode(_src))}, ::Val{$(QuoteNode(_k))})
        _SOURCE_MODULES[$(QuoteNode(_src))].examples(Val($(QuoteNode(_k))))
    end
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    get_examples(source, key) → (formula::String, data::DataFrame)
    get_examples(Val(source), Val(key)) → (formula::String, data::DataFrame)

Return the `(formula, data)` pair for the given source and formula key.

Each `(source, key)` pair corresponds to exactly one formula and one dataset.
The docstring mirrors the one in the source file's `examples()` function and is
shown by `@doc`.

    formula, data = get_examples(:brms, :cbpp_binomial)
    println(formula)
"""
get_examples(source::Symbol, key::Symbol) = get_examples(Val(source), Val(key))

"""
    load_data(source, key) → DataFrame

Load the raw dataset for the given source and dataset key.
Not all entries have an explicit `load` function; for synthetic-only entries
this will throw a `MethodError`.
"""
function load_data(source::Symbol, key::Symbol)
    mod = get(_SOURCE_MODULES, source, nothing)
    mod === nothing && error("Unknown source :$source.")
    return mod.load(Val(key))
end

"""
    describe(source, key) → Markdown.MD

Return the full docstring for the given `(source, key)` pair.
Equivalent to `@doc get_examples(Val(source), Val(key))` in the REPL.

    describe(:mcelreath, :howell1_linear)

# Docstring format

Every `examples()` function in the source files has a docstring with this structure:

```
\"\"\"
name: Human-readable title
source: https://example.com/notebook  # or "synthetic"
example: example_group_name           # groups related formulas
dataset: dataset_name                 # the load() key for the dataset
formula: "formula string"             # the model formula
chapter: SR2 Ch 4                     # optional; for book-based files only
verified: true                        # optional; default false
----

Markdown description.
\"\"\"
```

**YAML fields** (before the `----` line):

| Field      | Type   | Required | Notes |
|------------|--------|----------|-------|
| `name`     | string | yes      | Display title |
| `source`   | string | yes      | URL or `"synthetic"` |
| `example`  | string | yes      | Example group (e.g. `cbpp`, `radon`) |
| `dataset`  | string | yes      | Dataset key matching a `load()` overload |
| `formula`  | string | yes      | The model formula string |
| `chapter`  | string | no       | Chapter reference for book-based files |
| `verified` | bool   | no       | `true` once human-reviewed; default `false` |
"""
function describe(source::Symbol, key::Symbol)
    mod = get(_SOURCE_MODULES, source, nothing)
    mod === nothing && error("Unknown source :$source.")
    sig = Core.apply_type(Tuple, Core.apply_type(Val, key))
    return Base.Docs.doc(Base.Docs.Binding(mod, :examples), sig)
end

# Extract the first non-empty line of a docstring as a one-line summary.
function _summary(source::Symbol, key::Symbol)
    text = sprint(show, MIME("text/plain"), describe(source, key))
    for line in split(text, "\n")
        s = strip(line)
        !isempty(s) && return s
    end
    return string(key)
end

"""
    list_examples([source]) → nothing

Print a formatted catalog of all available examples, optionally filtered
to a single source symbol.

    list_examples()              # all sources
    list_examples(:brms)        # only brms vignette examples
"""
function list_examples(source::Union{Symbol,Nothing}=nothing)
    entries = source === nothing ? CATALOG : filter(e -> e.source == source, CATALOG)
    isempty(entries) && (println("No entries for source :$source."); return)
    W_src, W_key = 16, 40
    println(rpad("Source", W_src), rpad("Key", W_key), "Summary")
    println("─"^(W_src + W_key + 52))
    prev = nothing
    for e in entries
        e.source !== prev && (prev !== nothing && println(); prev = e.source)
        println(rpad(string(e.source), W_src), rpad(string(e.key), W_key), _summary(e.source, e.key))
    end
    n_src = length(unique(e.source for e in entries))
    println("\nTotal: $(length(entries)) formulas across $n_src source(s).")
end

"""
    all_examples() → Channel

Lazy iterator over every `(source, key, formula, data)` tuple in the catalog.

    for (source, key, formula, data) in all_examples()
        println(source, " / ", key, " : ", formula)
    end
"""
function all_examples()
    Channel() do ch
        for e in CATALOG
            formula, data = get_examples(e.source, e.key)
            put!(ch, (e.source, e.key, formula, data))
        end
    end
end

# ── Run as script ─────────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    list_examples()
end
