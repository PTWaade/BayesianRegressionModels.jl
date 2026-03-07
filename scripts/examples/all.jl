# scripts/examples/all.jl
#
# Entry point for all regression examples.
#
# Each (source, key) pair has a documented `get_examples` method, so
# descriptions live in the docstring system and are queryable via:
#   @doc get_examples(Val(:mcelreath), Val(:howell1))
#   describe(:mcelreath, :howell1)
#
# Usage:
#   include("scripts/examples/all.jl")
#   list_examples()                             # print full catalog
#   list_examples(:mcelreath)                   # filter by source
#   pairs = get_examples(:mcelreath, :milk)     # → Vector{Tuple{String, DataFrame}}
#   df    = load_data(:mcelreath, :howell1)     # → DataFrame (where load() exists)
#   md    = describe(:mcelreath, :howell1)      # → Markdown.MD docstring
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
# Descriptions are the ground truth: they are both stored here for listing
# and registered as docstrings on specific get_examples(Val, Val) methods
# (see the @eval loop below) so they are queryable via @doc and describe().

const CATALOG = [

    # ── bambi ─────────────────────────────────────────────────────────────────
    (source=:bambi, key=:escs,                    description="PISA ESCS — educational attainment (linear)"),
    (source=:bambi, key=:cherry_blossoms,          description="Cherry blossom bloom dates — spline regression"),
    (source=:bambi, key=:dietox,                   description="Dietary experiment — longitudinal linear mixed"),
    (source=:bambi, key=:sleepstudy,               description="Sleep deprivation — varying intercepts and slopes"),
    (source=:bambi, key=:radon,                    description="Radon measurements — hierarchical normal"),
    (source=:bambi, key=:strack_rrr,               description="Strack facial-feedback replication — linear mixed"),
    (source=:bambi, key=:shooter,                  description="Police shooting bias — categorical predictor"),
    (source=:bambi, key=:fixed_random,             description="Fixed vs random effects illustration"),
    (source=:bambi, key=:t_regression,             description="Student-t robust regression"),
    (source=:bambi, key=:predict_new_groups,       description="Prediction for new (unseen) groups"),
    (source=:bambi, key=:polynomial_regression,    description="Polynomial regression demo"),
    (source=:bambi, key=:logistic_anes,            description="ANES logistic regression"),
    (source=:bambi, key=:model_comparison,         description="Model comparison (LOO/WAIC)"),
    (source=:bambi, key=:hierarchical_binomial,    description="Hierarchical binomial model"),
    (source=:bambi, key=:alternative_links,        description="GLM link function comparison"),
    (source=:bambi, key=:wald_gamma,               description="Wald / Gamma regression"),
    (source=:bambi, key=:negative_binomial,        description="Negative binomial count regression"),
    (source=:bambi, key=:count_roaches,            description="Cockroach count data — Poisson/NB"),
    (source=:bambi, key=:beta_regression,          description="Beta regression — batting averages"),
    (source=:bambi, key=:categorical_regression,   description="Categorical (softmax) regression — iris"),
    (source=:bambi, key=:circular_regression,      description="Circular / von Mises regression"),
    (source=:bambi, key=:quantile_regression,      description="Bayesian quantile regression — BMI"),
    (source=:bambi, key=:mister_p,                 description="Multilevel regression & poststratification"),
    (source=:bambi, key=:zero_inflated,            description="Zero-inflated Poisson / NB"),
    (source=:bambi, key=:ordinal_regression,       description="Ordinal / cumulative regression"),
    (source=:bambi, key=:distributional_models,    description="Distributional models — bikes dataset"),
    (source=:bambi, key=:hsgp_1d,                  description="Hilbert-space GP — 1D smooth"),
    (source=:bambi, key=:hsgp_2d,                  description="Hilbert-space GP — 2D smooth"),
    (source=:bambi, key=:survival_model,           description="Survival model (general)"),
    (source=:bambi, key=:survival_discrete_time,   description="Discrete-time survival"),
    (source=:bambi, key=:survival_continuous_time, description="Continuous-time survival"),
    (source=:bambi, key=:orthogonal_polynomial,    description="Orthogonal polynomial regression — MPG"),
    (source=:bambi, key=:plot_predictions,         description="Marginal predictions plotting — mtcars"),
    (source=:bambi, key=:plot_comparisons,         description="Marginal comparisons plotting"),
    (source=:bambi, key=:plot_slopes,              description="Marginal slopes plotting"),
    (source=:bambi, key=:alternative_samplers,     description="Alternative MCMC samplers demo"),

    # ── brms vignettes ────────────────────────────────────────────────────────
    (source=:brms, key=:cbpp,          description="CBPP cattle disease — custom beta-binomial family"),
    (source=:brms, key=:distreg,       description="Distributional regression — location-scale & GAM"),
    (source=:brms, key=:nhanes,        description="NHANES — multiple imputation / joint missing-data"),
    (source=:brms, key=:income,        description="Life satisfaction — monotonic ordinal income effect"),
    (source=:brms, key=:btdata,        description="Blue tit bivariate — correlated random effects"),
    (source=:brms, key=:nonlinear,     description="Nonlinear models — exponential growth & loss triangle"),
    (source=:brms, key=:phylogenetics, description="Phylogenetic models — gr() covariance random effect"),

    # ── McElreath / Kurz SR2 ──────────────────────────────────────────────────
    (source=:mcelreath, key=:globe,           description="Globe tossing — binomial coin-flip (ch2)"),
    (source=:mcelreath, key=:howell1,         description="!Kung height/weight — linear & nonlinear (ch4)"),
    (source=:mcelreath, key=:cherry_blossoms, description="Cherry blossom bloom dates — B-spline (ch4)"),
    (source=:mcelreath, key=:waffle_divorce,  description="Waffle divorce — DAG / multiple regression (ch5)"),
    (source=:mcelreath, key=:milk,            description="Primate milk — missing data / index model (ch5,15)"),
    (source=:mcelreath, key=:plant_growth,    description="Plant growth — post-treatment bias (ch6)"),
    (source=:mcelreath, key=:happiness,       description="Age × happiness — collider bias (ch6)"),
    (source=:mcelreath, key=:rugged,          description="Terrain ruggedness × GDP — interaction (ch8)"),
    (source=:mcelreath, key=:tulips,          description="Tulip blooms — water × shade interaction (ch8)"),
    (source=:mcelreath, key=:hetero,          description="Heteroscedastic regression — sigma submodel (ch9)"),
    (source=:mcelreath, key=:chimpanzees,     description="Prosocial chimp experiment — GLMM (ch11)"),
    (source=:mcelreath, key=:ucbadmit,        description="UC Berkeley admissions — aggregated binomial (ch11)"),
    (source=:mcelreath, key=:kline,           description="Kline island tools — Poisson / GLMM (ch11)"),
    (source=:mcelreath, key=:kline2,          description="Kline2 island tools — GP distance effect (ch14)"),
    (source=:mcelreath, key=:trolley,         description="Trolley moral dilemma — ordered logit (ch12)"),
    (source=:mcelreath, key=:reedfrogs,       description="Reed frog survival — varying intercepts (ch13)"),
    (source=:mcelreath, key=:cafe,            description="Café waiting times — varying slopes (ch14)"),
    (source=:mcelreath, key=:primates301,     description="Primate brain/body — phylogenetic GP (ch14)"),
    (source=:mcelreath, key=:moralizing_gods, description="Moralizing gods — historical panel data (ch14)"),
    (source=:mcelreath, key=:panda_nuts,      description="Panda nut cracking — nonlinear growth (ch16)"),
    (source=:mcelreath, key=:lynx_hare,       description="Lynx-hare population dynamics — ODE (ch16)"),
    (source=:mcelreath, key=:btdata,          description="Blue tit — multivariate mixed model (ch15)"),

    # ── Kruschke / Kurz DBDA2 ─────────────────────────────────────────────────
    (source=:kruschke, key=:z15n50,            description="Bernoulli 15/50 — MCMC intro (ch8)"),
    (source=:kruschke, key=:z6n8z2n7,          description="Two-mint Bernoulli 6/8 + 2/7 (ch8)"),
    (source=:kruschke, key=:therapeutic_touch,  description="Therapeutic touch — hierarchical Bernoulli (ch9)"),
    (source=:kruschke, key=:batting_average,    description="MLB batting averages — hierarchical binomial (ch9)"),
    (source=:kruschke, key=:recall,             description="Memory recall — binomial conditions (ch12)"),
    (source=:kruschke, key=:two_group_iq,       description="Two-group IQ — group-specific mu/sigma (ch16)"),
    (source=:kruschke, key=:calcium,            description="Calcium supplementation RCT — two-group (ch16)"),
    (source=:kruschke, key=:htwt,               description="Height/weight regression — metric predictor (ch17)"),
    (source=:kruschke, key=:hier_linreg,        description="Hierarchical linear regression (ch17)"),
    (source=:kruschke, key=:income_famsize,     description="Income by family size/state — measurement error (ch17)"),
    (source=:kruschke, key=:guber1999,          description="US SAT scores — multiple regression (ch18)"),
    (source=:kruschke, key=:fruitfly,           description="Fruitfly longevity — one-way ANOVA/ANCOVA (ch19)"),
    (source=:kruschke, key=:salary,             description="Fictional salary — two-way random effects (ch20)"),
    (source=:kruschke, key=:splitplot,          description="Split-plot agronomy — two-way ANOVA (ch20)"),
    (source=:kruschke, key=:htwt110,            description="Height/weight/sex — logistic regression (ch21)"),
    (source=:kruschke, key=:softmax,            description="Softmax / conditional logistic regression (ch22)"),
    (source=:kruschke, key=:ordinal_probit,     description="Ordered probit with discrimination (ch23)"),
    (source=:kruschke, key=:happiness_assets,   description="Happiness vs assets — cumulative ordinal (ch23)"),
    (source=:kruschke, key=:movies,             description="Movie ratings — ordinal regression (ch23)"),
    (source=:kruschke, key=:haireye,            description="Hair/eye color — Poisson log-linear (ch24)"),
    (source=:kruschke, key=:censored,           description="Truncated and censored data (ch25)"),

    # ── Bürkner papers ────────────────────────────────────────────────────────
    (source=:burkner_papers, key=:epilepsy,     description="Epileptic seizure counts — Poisson GLMM (JSS 2017)"),
    (source=:burkner_papers, key=:inhaler,      description="Inhaler ordinal ratings — sequential ratio (JSS 2017)"),
    (source=:burkner_papers, key=:kidney,       description="Kidney recurrence — lognormal survival (JSS 2017)"),
    (source=:burkner_papers, key=:hetero_jss,   description="Heteroscedastic & quantile regression (JSS 2017)"),
    (source=:burkner_papers, key=:fish_rj,      description="Fishing trips — zero-inflated Poisson (RJ 2018)"),
    (source=:burkner_papers, key=:rent99,       description="Munich rents — smooth distributional (RJ 2018)"),
    (source=:burkner_papers, key=:loss_rj,      description="Actuarial loss triangle — nonlinear Weibull (RJ 2018)"),
    (source=:burkner_papers, key=:multi_member, description="Multiple membership — mm() syntax (RJ 2018)"),

    # ── Vasishth / Nicenboim / Schad ──────────────────────────────────────────
    (source=:vasishth, key=:pupil,        description="Pupil dilation — cognitive load effect (ch4)"),
    (source=:vasishth, key=:spacebar,     description="Spacebar RT — practice effects, lognormal (ch4)"),
    (source=:vasishth, key=:recall_wm,    description="Working memory recall — set size (ch4)"),
    (source=:vasishth, key=:n400,         description="N400 EEG — word predictability, crossed RE (ch5)"),
    (source=:vasishth, key=:stroop,       description="Stroop task — lognormal varying slopes (ch5)"),
    (source=:vasishth, key=:pooling,      description="Hierarchical pooling illustration (ch5)"),
    (source=:vasishth, key=:contrasts1,   description="Contrast coding — single 4-level factor (ch6)"),
    (source=:vasishth, key=:contrasts2x2, description="Contrast coding — 2×2 factorial design (ch7)"),
    (source=:vasishth, key=:meta_sbi,     description="Meta-analysis — brain injury effect sizes (ch11)"),
    (source=:vasishth, key=:indiv_diff,   description="Individual differences — measurement error (ch11)"),
]

# ── Register each entry as a documented get_examples method ──────────────────
#
# This generates one method per (source, key) pair:
#   get_examples(::Val{:mcelreath}, ::Val{:howell1})  →  examples from mcelreath.jl
#
# Docstrings are attached to each method, so descriptions are queryable via:
#   @doc get_examples(Val(:mcelreath), Val(:howell1))
#   describe(:mcelreath, :howell1)
#   methods(get_examples)

for _entry in CATALOG
    _src, _k, _desc = _entry.source, _entry.key, _entry.description
    @eval @doc $_desc function get_examples(::Val{$(QuoteNode(_src))}, ::Val{$(QuoteNode(_k))})
        _SOURCE_MODULES[$(QuoteNode(_src))].examples(Val($(QuoteNode(_k))))
    end
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    get_examples(source, key) → Vector{Tuple{String, DataFrame}}
    get_examples(Val(source), Val(key)) → Vector{Tuple{String, DataFrame}}

Return all `(formula, data)` pairs for the given source and dataset key.

Each specific `(source, key)` pair has its own documented method, so:
- `@doc get_examples(Val(:mcelreath), Val(:howell1))` shows the dataset description.
- `methods(get_examples)` lists all registered examples.

Example:
    for (formula, df) in get_examples(:mcelreath, :milk)
        println(formula)
    end
"""
get_examples(source::Symbol, key::Symbol) = get_examples(Val(source), Val(key))

"""
    load_data(source, key) → DataFrame

Load the raw dataset for the given source and key.
Not all entries have an explicit `load` function; for synthetic-only entries
this will throw a `MethodError`.
"""
function load_data(source::Symbol, key::Symbol)
    mod = get(_SOURCE_MODULES, source, nothing)
    mod === nothing && error("Unknown source :$source.")
    return mod.load(Val(key))
end

const _DESCRIPTIONS = Dict((e.source, e.key) => e.description for e in CATALOG)

"""
    describe(source, key) → String

Return the one-line description for the given `(source, key)` pair.

The same text is also accessible as a method docstring via:
    @doc get_examples(Val(:mcelreath), Val(:howell1))

Example:
    describe(:mcelreath, :howell1)
"""
function describe(source::Symbol, key::Symbol)
    get(_DESCRIPTIONS, (source, key), "No description found for ($source, $key)")
end

"""
    list_examples([source]) → nothing

Print a formatted catalog of all available examples, optionally filtered
to a single source symbol.

    list_examples()              # all sources
    list_examples(:mcelreath)   # only McElreath/SR2 examples
"""
function list_examples(source::Union{Symbol,Nothing}=nothing)
    entries = source === nothing ? CATALOG : filter(e -> e.source == source, CATALOG)
    isempty(entries) && (println("No entries for source :$source."); return)
    W_src, W_key = 16, 24
    println(rpad("Source", W_src), rpad("Key", W_key), "Description")
    println("─"^(W_src + W_key + 52))
    prev = nothing
    for e in entries
        e.source !== prev && (prev !== nothing && println(); prev = e.source)
        println(rpad(string(e.source), W_src), rpad(string(e.key), W_key), e.description)
    end
    n_src = length(unique(e.source for e in entries))
    println("\nTotal: $(length(entries)) datasets across $n_src source(s).")
end

"""
    all_examples() → Channel

Lazy iterator over every `(source, key, pairs)` triple in the catalog,
where `pairs = get_examples(source, key)`.

    for (source, key, pairs) in all_examples()
        for (formula, df) in pairs
            # process formula and dataset
        end
    end
"""
function all_examples()
    Channel() do ch
        for e in CATALOG
            put!(ch, (e.source, e.key, get_examples(e.source, e.key)))
        end
    end
end

# ── Run as script ─────────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    list_examples()
end
