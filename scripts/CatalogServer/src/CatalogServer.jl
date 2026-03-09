module CatalogServer

using Markdown
using YAML
using Oxygen: @get, html, serve as oxygen_serve

export render_page, serve

# ── Source metadata ───────────────────────────────────────────────────────────

const SOURCE_META = Dict(
    :bambi          => (label="Bambi model gallery",                              color="#f59e0b"),
    :brms           => (label="brms vignettes (Bürkner)",                         color="#3b82f6"),
    :mcelreath      => (label="Statistical Rethinking 2e (Kurz / McElreath)",     color="#10b981"),
    :kruschke       => (label="DBDA2 (Kurz / Kruschke)",                          color="#8b5cf6"),
    :burkner_papers => (label="Bürkner JSS 2017 + R Journal 2018",                color="#06b6d4"),
    :vasishth       => (label="Bayesian Data Analysis for Cognitive Science",      color="#ef4444"),
    :action_models  => (label="ActionModels.jl (ComputationalPsychiatry)",         color="#f97316"),
    :epidist        => (label="epidist (Epinowcast Community)",                    color="#ec4899"),
    :epinowcast     => (label="epinowcast + baselinenowcast",                      color="#14b8a6"),
    :rstanarm       => (label="rstanarm vignettes",                                color="#6366f1"),
    :bmm            => (label="bmm — Bayesian Measurement Models",                 color="#d946ef"),
    :flocker        => (label="flocker — Bayesian Occupancy Models",               color="#84cc16"),
    :mvgam           => (label="mvgam — Multivariate Dynamic GAMs",                 color="#0ea5e9"),
    :inla            => (label="R-INLA — INLA with f() random effects",             color="#fb923c"),
    :mcmcglmm        => (label="MCMCglmm — MCMC GLMMs with animal models",          color="#e879f9"),
    :glmmtmb         => (label="glmmTMB — GLMMs with ZI/hurdle and dispersion",     color="#34d399"),
    :lme4            => (label="lme4 — Linear/Generalised Mixed Models (R)",        color="#f43f5e"),
    :mixed_models_jl => (label="MixedModels.jl — Mixed Models (Julia)",             color="#a855f7"),
    :glm_jl          => (label="GLM.jl — Generalised Linear Models (Julia)",        color="#22c55e"),
)

_source_meta(src::Symbol) = get(SOURCE_META, src, (label=string(src), color="#6b7280"))

# ── YAML / docstring helpers ──────────────────────────────────────────────────

function parse_yaml(yaml_str::AbstractString)::Dict{String,Any}
    isempty(strip(yaml_str)) && return Dict{String,Any}()
    raw = try
        YAML.load(yaml_str)
    catch
        return Dict{String,Any}()
    end
    raw isa Dict || return Dict{String,Any}()
    return Dict{String,Any}(string(k) => v for (k, v) in raw)
end

function _raw_docstring(source_modules::Dict, source::Symbol, key::Symbol)
    mod     = source_modules[source]
    sig     = Core.apply_type(Tuple, Core.apply_type(Val, key))
    binding = Base.Docs.Binding(mod, :examples)
    try
        multidoc = get(Base.Docs.meta(mod), binding, nothing)
        multidoc === nothing && return nothing
        docstr = get(multidoc.docs, sig, nothing)
        docstr === nothing && return nothing
        return join(docstr.text, "")
    catch
        return nothing
    end
end

function _describe(source_modules::Dict, source::Symbol, key::Symbol)
    mod = source_modules[source]
    sig = Core.apply_type(Tuple, Core.apply_type(Val, key))
    return Base.Docs.doc(Base.Docs.Binding(mod, :examples), sig)
end

function split_docstring(source_modules::Dict, source::Symbol, key::Symbol)
    raw = _raw_docstring(source_modules, source, key)
    if raw !== nothing
        m = match(r"\n---+\n", raw)
        if m !== nothing
            yaml_text = raw[1:m.offset-1]
            desc_text = strip(raw[m.offset+length(m.match):end])
            return parse_yaml(yaml_text), Markdown.parse(desc_text)
        end
        return Dict{String,Any}(), _describe(source_modules, source, key)
    end
    doc    = _describe(source_modules, source, key)
    hr_idx = findfirst(x -> x isa Markdown.HorizontalRule, doc.content)
    hr_idx === nothing && return Dict{String,Any}(), doc
    yaml_md = Markdown.MD(doc.content[1:hr_idx-1])
    desc_md = Markdown.MD(doc.content[hr_idx+1:end])
    return parse_yaml(sprint(show, MIME("text/plain"), yaml_md)), desc_md
end

const _Cache = Dict{Tuple{Symbol,Symbol}, Tuple{Dict{String,Any}, Markdown.MD}}

function entry_meta!(cache::_Cache, source_modules::Dict, source::Symbol, key::Symbol)
    get!(cache, (source, key)) do
        split_docstring(source_modules, source, key)
    end
end

# ── HTML generation ───────────────────────────────────────────────────────────

esc_html(s) = replace(string(s), "&"=>"&amp;", "<"=>"&lt;", ">"=>"&gt;", "\""=>"&quot;")
esc_attr(s) = replace(string(s), "&"=>"&amp;", "<"=>"&lt;", ">"=>"&gt;", "\""=>"&quot;", "\n"=>"&#10;")

# ── Parseable / sampleable predicates ────────────────────────────────────────
# Replace these with real implementations once available.
is_parseable(source::Symbol, key::Symbol, formula::AbstractString, dataset::AbstractString) = false
is_sampleable(source::Symbol, key::Symbol, formula::AbstractString, dataset::AbstractString) = false

is_hidden(cache::_Cache, source::Symbol, key::Symbol) =
    get(cache[(source, key)][1], "hidden", false) === true

function render_table_row(cache::_Cache, source::Symbol, key::Symbol)
    meta, desc = cache[(source, key)]
    verified   = get(meta, "verified", false) === true
    example    = get(meta, "example", "")
    dataset    = get(meta, "dataset", "")
    formula    = get(meta, "formula", "")
    family     = get(meta, "family", "gaussian")
    chapter    = get(meta, "chapter", "")
    source_url = get(meta, "source", "")
    parseable  = is_parseable(source, key, formula, dataset)
    sampleable = is_sampleable(source, key, formula, dataset)
    color      = _source_meta(source).color
    row_id     = "tr-$(source)-$(key)"

    verified_cell = verified ?
        """<td class="col-verified-true" title="Verified">&#10003;</td>""" :
        """<td class="col-verified-false" title="Not verified">—</td>"""
    chapter_tag = !isempty(chapter) ?
        """<span class="row-detail-chapter">$(esc_html(chapter))</span>""" : ""
    source_tag = if source_url == "synthetic"
        """<span class="row-detail-synthetic">synthetic</span>"""
    elseif !isempty(source_url)
        """<a class="row-detail-link" href="$(esc_html(source_url))" target="_blank" rel="noopener">&#8599; source</a>"""
    else
        ""
    end
    family_tag = !isempty(family) ?
        """<span class="badge-family" title="Family">$(esc_html(family))</span>""" : ""
    detail_row = """<tr id="$(row_id)-detail" class="detail-row" style="display:none">
      <td colspan="5"><div class="row-detail-inner">
        <div class="row-detail-meta">$(chapter_tag)$(source_tag)$(family_tag)
          <span class="badge-flag $(parseable  ? "badge-flag-on" : "badge-flag-off")" title="$(parseable  ? "Parseable"     : "Not parseable")">$(parseable  ? "&#10003;&nbsp;" : "")parseable</span>
          <span class="badge-flag $(sampleable ? "badge-flag-on" : "badge-flag-off")" title="$(sampleable ? "Sampleable"    : "Not sampleable")">$(sampleable ? "&#10003;&nbsp;" : "")sampleable</span>
        </div>
        <div class="row-detail-body">$(Markdown.html(desc))</div>
      </div></td>
    </tr>"""

    return """<tr id="$(row_id)" class="catalog-row"
                  onclick="toggleDetail('$(row_id)')"
                  data-source="$(source)"
                  data-verified="$(verified ? "true" : "false")"
                  data-example="$(esc_attr(example))"
                  data-dataset="$(esc_attr(dataset))"
                  data-formula="$(esc_attr(formula))"
                  data-family="$(esc_attr(family))"
                  data-parseable="$(parseable ? "true" : "false")"
                  data-sampleable="$(sampleable ? "true" : "false")">
      $(verified_cell)
      <td><code style="color:$(color)">:$(source)</code></td>
      <td>$(esc_html(example))</td>
      <td>$(esc_html(dataset))</td>
      <td class="formula-cell"><div class="formula-flex"><div class="formula-code-wrap"><code style="user-select:all">$(esc_html(formula))</code></div><button class="copy-btn" onclick="event.stopPropagation();copyFormulaBtn(this)" title="Copy formula">&#x2398;</button></div></td>
    </tr>
    $(detail_row)"""
end

function render_card(cache::_Cache, source::Symbol, key::Symbol)
    meta, desc = cache[(source, key)]
    verified   = get(meta, "verified", false) === true
    name       = get(meta, "name", "")
    source_url = get(meta, "source", "")
    chapter    = get(meta, "chapter", "")
    example    = get(meta, "example", "")
    dataset    = get(meta, "dataset", "")
    formula    = get(meta, "formula", "")
    family     = get(meta, "family", "gaussian")
    body       = Markdown.html(desc)
    color      = _source_meta(source).color

    subtitle = let parts = split(name, " — ", limit=2)
        length(parts) == 2 ? parts[2] : name
    end

    parseable  = is_parseable(source, key, formula, dataset)
    sampleable = is_sampleable(source, key, formula, dataset)

    verified_badge = verified ?
        """<span class="badge-verified" title="Human-verified">&#10003;&nbsp;verified</span>""" :
        """<span class="badge-unverified" title="Not yet verified">unverified</span>"""
    subtitle_span = !isempty(subtitle) ?
        """<span class="card-subtitle">$(esc_html(subtitle))</span>""" : ""
    chapter_span = !isempty(chapter) ?
        """<span class="card-chapter">$(esc_html(chapter))</span>""" : ""
    source_link = if source_url == "synthetic"
        """<span class="card-source-synthetic" title="Synthetic / generated data">synthetic</span>"""
    elseif !isempty(source_url)
        """<a class="card-source-link" href="$(esc_html(source_url))" target="_blank" rel="noopener" title="Original source">&#8599;</a>"""
    else
        ""
    end
    formula_row = !isempty(formula) ?
        """<div class="card-formula"><code style="user-select:all">$(esc_html(formula))</code><button class="copy-btn" onclick="copyFormulaBtn(this)" title="Copy formula">&#x2398;</button></div>""" : ""
    dataset_tag = !isempty(dataset) ?
        """<span class="tag-dataset" title="Dataset">$(esc_html(dataset))</span>""" : ""
    family_badge = """<span class="badge-family" title="Family">$(esc_html(family))</span>"""
    flag_badge(label, val) = val ?
        """<span class="badge-flag badge-flag-on" title="$(label)">&#10003;&nbsp;$(label)</span>""" :
        """<span class="badge-flag badge-flag-off" title="Not $(label)">$(label)</span>"""

    card_class = verified ? "card card-verified" : "card"
    return """
        <article class="$(card_class)" id="$(source)-$(key)"
                 data-source="$(source)"
                 data-verified="$(verified ? "true" : "false")"
                 data-example="$(esc_attr(example))"
                 data-dataset="$(esc_attr(dataset))"
                 data-formula="$(esc_attr(formula))"
                 data-family="$(esc_attr(family))"
                 data-parseable="$(parseable ? "true" : "false")"
                 data-sampleable="$(sampleable ? "true" : "false")">
          <div class="card-header" style="border-left:4px solid $(color)">
            <span class="card-key">:$(key)</span>$(verified_badge)$(subtitle_span)$(chapter_span)$(source_link)
          </div>$(formula_row)
          <div class="card-body">$(body)</div>
          <div class="card-footer">$(dataset_tag)$(family_badge)$(flag_badge("parseable", parseable))$(flag_badge("sampleable", sampleable))</div>
        </article>"""
end

function _group_by_example(cache::_Cache, entries)
    groups = Dict{String, Vector}()
    order  = String[]
    for e in entries
        ex = get(cache[(e.source, e.key)][1], "example", "")
        if !haskey(groups, ex)
            push!(order, ex)
            groups[ex] = []
        end
        push!(groups[ex], e)
    end
    return [(ex, groups[ex]) for ex in order]
end

function _example_heading(cache::_Cache, ex::String, entries)
    names = [get(cache[(e.source, e.key)][1], "name", "") for e in entries]
    prefixes = unique(first(split(n, " — ", limit=2)) for n in names if !isempty(n))
    length(prefixes) == 1 && return first(prefixes)
    return isempty(ex) ? "" : titlecase(replace(ex, "_" => " "))
end

function render_page(catalog, source_modules::Dict)
    cache = _Cache()
    for e in catalog
        entry_meta!(cache, source_modules, e.source, e.key)
    end

    visible = filter(e -> !is_hidden(cache, e.source, e.key), catalog)

    n_verified = count(e -> get(cache[(e.source, e.key)][1], "verified", false) === true, visible)

    nav_items = join([
        """<button class="nav-btn" data-source="$(src)" style="border-bottom:3px solid $(_source_meta(src).color)" _="on click call handleNavClick(me)">:$(src)</button>"""
        for src in unique(e.source for e in visible)], "\n        ")

    sections = join([begin
        src_meta = _source_meta(src)
        entries  = filter(e -> e.source == src, visible)
        groups   = sort(_group_by_example(cache, entries), by = ((ex, grp),) -> begin
            has_v = any(e -> get(cache[(e.source, e.key)][1], "verified", false) === true, grp)
            (has_v ? 0 : 1, ex)
        end)

        groups_html = join([begin
            grp_entries = sort(grp_entries,
                by  = e -> get(cache[(e.source, e.key)][1], "verified", false) === true ? 0 : 1,
                alg = MergeSort)
            heading = _example_heading(cache, ex, grp_entries)
            n_v     = count(e -> get(cache[(e.source, e.key)][1], "verified", false) === true, grp_entries)
            vbadge  = n_v > 0 ? """<span class="badge badge-v">$(n_v)&#10003;</span>""" : ""
            cards   = join(render_card(cache, e.source, e.key) for e in grp_entries)
            h_html  = isempty(heading) ? "" :
                """<h3 class="example-heading">$(esc_html(heading))<span class="badge">$(length(grp_entries))</span>$(vbadge)</h3>"""
            """<div class="example-group" data-example="$(esc_html(ex))">$(h_html)
              <div class="card-grid">$(cards)
              </div>
            </div>"""
        end for (ex, grp_entries) in groups])

        n_v_src    = count(e -> get(cache[(e.source, e.key)][1], "verified", false) === true, entries)
        vbadge_src = n_v_src > 0 ? " <span class=\"badge badge-v\" title=\"$(n_v_src) verified\">$(n_v_src)&#10003;</span>" : ""
        """
        <section id="$(src)">
          <h2 style="border-left:6px solid $(src_meta.color);padding-left:.75rem">
            <code>:$(src)</code> — $(esc_html(src_meta.label))
            <span class="badge">$(length(entries))</span>$(vbadge_src)
          </h2>
          $(groups_html)
        </section>"""
    end for src in unique(e.source for e in visible)])

    table_rows = join(render_table_row(cache, e.source, e.key) for e in visible)

    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>BayesianRegressionModels — Example Catalog</title>
  <script src="https://unpkg.com/hyperscript.org@0.9.12"></script>
  <script type="text/hyperscript">
    def filterCatalog()
      set q to #search.value.toLowerCase()
      set src           to #active-source.value
      set verifiedOnly  to #verified-only.checked
      set parseableOnly to #parseable-only.checked
      set sampleableOnly to #sampleable-only.checked
      for card in <article.card/>
        set text to card.textContent.toLowerCase()
        set srcOk       to src is '' or card.dataset.source is src
        set textOk      to q is '' or text contains q
        set verifiedOk  to not verifiedOnly  or card.dataset.verified  is 'true'
        set parseableOk to not parseableOnly or card.dataset.parseable is 'true'
        set sampleableOk to not sampleableOnly or card.dataset.sampleable is 'true'
        if srcOk and textOk and verifiedOk and parseableOk and sampleableOk
          show card
        else
          hide card
        end
      end
      -- hide example groups that have no visible cards
      for grp in <.example-group/>
        set hasVisible to false
        for card in <.card/> in grp
          if card.style.display is not 'none'
            set hasVisible to true
          end
        end
        if hasVisible then show grp else hide grp end
      end
      -- hide source sections that have no visible groups
      for sec in <section/>
        set hasVisible to false
        for grp in <.example-group/> in sec
          if grp.style.display is not 'none'
            set hasVisible to true
          end
        end
        if hasVisible then show sec else hide sec end
      end
      -- filter table rows
      for row in <#catalog-tbody tr.catalog-row/>
        set text to row.textContent.toLowerCase()
        set srcOk       to src is '' or row.dataset.source is src
        set textOk      to q is '' or text contains q
        set verifiedOk  to not verifiedOnly  or row.dataset.verified  is 'true'
        set parseableOk to not parseableOnly or row.dataset.parseable is 'true'
        set sampleableOk to not sampleableOnly or row.dataset.sampleable is 'true'
        if srcOk and textOk and verifiedOk and parseableOk and sampleableOk
          show row
        else
          hide row
        end
      end
      call syncDetailRows()
    end

    def handleNavClick(btn)
      if btn matches .active
        remove .active from btn
        set #active-source.value to ''
      else
        remove .active from .nav-btn
        add .active to btn
        set #active-source.value to btn.dataset.source
      end
      call filterCatalog()
    end
  </script>
  <style>
    *{box-sizing:border-box}
    body{font-family:system-ui,sans-serif;font-size:15px;line-height:1.6;
         color:#1a1a1a;background:#f8f9fa;margin:0}
    header{background:#1a1a2e;color:#fff;padding:1.5rem 2rem;
           display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap}
    header h1{margin:0;font-size:1.5rem}
    header p{margin:.25rem 0 0;opacity:.7;font-size:.9rem}
    .header-text{flex:1;min-width:0}
    .github-link{color:#fff;opacity:.75;text-decoration:none;font-size:.85rem;
                 border:1px solid rgba(255,255,255,.3);border-radius:6px;
                 padding:.3rem .7rem;white-space:nowrap;flex-shrink:0}
    .github-link:hover{opacity:1;background:rgba(255,255,255,.1)}
    nav{position:sticky;top:0;z-index:100;background:#fff;
        border-bottom:1px solid #e0e0e0;padding:.6rem 2rem;
        display:flex;gap:1rem;flex-wrap:wrap;align-items:center}
    .nav-btn{background:none;border:none;border-bottom:3px solid transparent;
             cursor:pointer;color:#555;font-size:.85rem;font-weight:600;
             font-family:monospace;padding:.15rem .4rem;border-radius:4px 4px 0 0;
             transition:color .15s,background .15s}
    .nav-btn:hover{color:#111;background:#f0f0f0}
    .nav-btn.active{color:#111;background:#f0f0f0}
    .nav-filter{display:flex;align-items:center;gap:.35rem;
                font-size:.83rem;color:#555;cursor:pointer;
                padding:.15rem .5rem;border-radius:4px;
                border:1px solid #d0d0d0;background:#fafafa;
                white-space:nowrap;user-select:none}
    .nav-filter:hover{background:#f0f0f0}
    .nav-filter input{cursor:pointer;accent-color:#16a34a}
    #search{margin-left:auto;padding:.3rem .65rem;border:1px solid #d0d0d0;
            border-radius:6px;font-size:.85rem;width:210px;outline:none}
    #search:focus{border-color:#888;box-shadow:0 0 0 2px rgba(0,0,0,.06)}
    main{max-width:1400px;margin:0 auto;padding:2rem}
    section{margin-bottom:3rem}
    section h2{font-size:1.1rem;margin:0 0 1rem;
               display:flex;align-items:center;gap:.75rem}
    .example-group{margin-bottom:1.75rem}
    .example-heading{font-size:.95rem;font-weight:600;color:#333;
                     margin:0 0 .6rem;display:flex;align-items:center;gap:.5rem}
    .badge{background:#e0e0e0;color:#555;font-size:.75rem;font-weight:600;
           border-radius:999px;padding:.1rem .55rem}
    .badge-v{background:#dcfce7;color:#16a34a}
    .card-grid{display:grid;
               grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:.75rem}
    .card{background:#fff;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden}
    .card-header{padding:.55rem .9rem;background:#fafafa;
                 border-bottom:1px solid #e0e0e0;
                 display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
    .card-key{font-family:monospace;font-size:.85rem;font-weight:700;color:#444}
    .card-subtitle{font-size:.8rem;color:#555;flex:1;min-width:0;
                   overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .card-chapter{font-size:.72rem;color:#777;background:#f0f0f0;
                  border-radius:4px;padding:.1rem .35rem;white-space:nowrap}
    .card-source-link{font-size:.78rem;color:#bbb;text-decoration:none;margin-left:auto}
    .card-source-link:hover{color:#333}
    .card-source-synthetic{font-size:.7rem;color:#999;background:#f5f5f5;
                            border:1px solid #e0e0e0;border-radius:4px;
                            padding:.05rem .35rem;margin-left:auto;white-space:nowrap}
    .badge-verified{font-size:.7rem;font-weight:700;color:#16a34a;
                    background:#dcfce7;border:1px solid #bbf7d0;
                    border-radius:999px;padding:.05rem .4rem;white-space:nowrap;
                    margin-left:auto}
    .badge-unverified{font-size:.7rem;color:#aaa;background:#f5f5f5;
                      border:1px solid #e0e0e0;border-radius:999px;
                      padding:.05rem .4rem;white-space:nowrap;margin-left:auto}
    .card-verified{border-color:#bbf7d0;box-shadow:0 0 0 1px #bbf7d0}
    .card-footer{padding:.3rem .9rem .45rem;display:flex;gap:.4rem;flex-wrap:wrap;
                 align-items:center;border-top:1px solid #f0f0f0}
    .badge-flag{font-size:.68rem;border-radius:4px;
                padding:.05rem .4rem;white-space:nowrap}
    .badge-flag-on{color:#2563eb;background:#eff6ff;border:1px solid #bfdbfe}
    .badge-flag-off{color:#d1d5db;background:#f9fafb;border:1px solid #e5e7eb}
    .badge-family{font-size:.68rem;border-radius:4px;padding:.05rem .4rem;
                  white-space:nowrap;color:#7c3aed;background:#f5f3ff;
                  border:1px solid #ddd6fe}
    .card-formula{padding:.5rem .9rem;background:#f3f4f6;
                  border-bottom:1px solid #e5e7eb;overflow-x:auto;
                  display:flex;align-items:flex-start;gap:.4rem}
    .card-formula code{font-family:monospace;font-size:.8rem;
                       white-space:pre;color:#1a1a1a;flex:1}
    .copy-btn{background:none;border:1px solid #d0d0d0;border-radius:4px;
              cursor:pointer;font-size:.75rem;color:#888;padding:.05rem .3rem;
              line-height:1.4;white-space:nowrap;flex-shrink:0}
    .copy-btn:hover{color:#333;background:#f0f0f0;border-color:#999}
    .card-body{padding:.6rem .9rem;font-size:.82rem}
    .card-body p{margin:.35rem 0}
    .card-body ul{margin:.35rem 0;padding-left:1.3rem}
    .card-body li{margin:.15rem 0}
    .card-body code{font-family:monospace;font-size:.78rem;
                    background:#f0f0f0;border-radius:3px;padding:.1em .3em}
    .card-body pre{background:#f0f0f0;border-radius:5px;
                   padding:.55rem .75rem;overflow-x:auto;font-size:.76rem}
    .card-body pre code{background:none;padding:0}
    .tag-dataset{font-size:.7rem;color:#666;background:#f0f0f0;
                 border-radius:4px;padding:.05rem .4rem;margin-right:.25rem}
    footer{text-align:center;color:#888;font-size:.8rem;
           padding:2rem;border-top:1px solid #e0e0e0}
    /* ── Table view ── */
    #table-view{max-width:1400px;margin:0 auto;padding:2rem}
    .table-wrap{border:1px solid #e0e0e0;border-radius:8px;overflow:hidden}
    #catalog-table{width:100%;border-collapse:collapse;table-layout:fixed;
                   font-size:.82rem;background:#fff}
    #catalog-table th:nth-child(1){width:2.5rem}
    #catalog-table th:nth-child(2){width:9rem}
    #catalog-table th:nth-child(3){width:12rem}
    #catalog-table th:nth-child(4){width:8rem}
    #catalog-table th{background:#f3f4f6;color:#444;font-weight:700;
                      padding:.5rem .75rem;text-align:left;cursor:pointer;
                      border-bottom:2px solid #e0e0e0;white-space:nowrap;
                      overflow:hidden;text-overflow:ellipsis;user-select:none}
    #catalog-table th:hover{background:#e9eaec}
    #catalog-table th.sort-asc::after{content:" ▲";font-size:.65rem;color:#888}
    #catalog-table th.sort-desc::after{content:" ▼";font-size:.65rem;color:#888}
    #catalog-table th:not(.sort-asc):not(.sort-desc)::after{content:" ↕";font-size:.65rem;color:#ccc}
    #catalog-table td{padding:.4rem .75rem;border-bottom:1px solid #f0f0f0;
                      vertical-align:middle;white-space:nowrap;
                      overflow:hidden;text-overflow:ellipsis}
    #catalog-table tr:last-child td{border-bottom:none}
    #catalog-table tr.catalog-row{cursor:pointer}
    #catalog-table tr.catalog-row:hover td{background:#f0f9ff}
    #catalog-table tr.row-expanded td{background:#e0f2fe}
    #catalog-table td.formula-cell{padding:0;overflow:hidden;white-space:normal}
    .formula-flex{display:flex;align-items:center;gap:.3rem;
                  padding:.4rem .75rem}
    .formula-code-wrap{flex:1;min-width:0;overflow-x:auto}
    #catalog-table .formula-cell code{font-family:monospace;font-size:.76rem;
                                       white-space:pre;display:block}
    .col-verified-true{background:#dcfce7;color:#16a34a;font-weight:700;
                       text-align:center;font-size:1rem}
    .col-verified-false{color:#d1d5db;text-align:center;font-size:1rem}
    .bool-true{color:#16a34a;font-weight:700;text-align:center}
    .bool-false{color:#9ca3af;text-align:center}
    #catalog-table tr.detail-row td{padding:0;background:#f8fafc;border-bottom:2px solid #e0e0e0;white-space:normal}
    .row-detail-inner{padding:.75rem 1.25rem}
    .row-detail-meta{display:flex;gap:.75rem;align-items:center;
                     margin-bottom:.5rem;font-size:.8rem}
    .row-detail-chapter{background:#f0f0f0;border-radius:4px;
                        padding:.1rem .35rem;color:#666}
    .row-detail-link{color:#3b82f6;text-decoration:none}
    .row-detail-link:hover{text-decoration:underline}
    .row-detail-synthetic{color:#999;font-size:.75rem;background:#f5f5f5;
                          border:1px solid #e0e0e0;border-radius:4px;
                          padding:.05rem .35rem}
    .row-detail-body{font-size:.82rem}
    .row-detail-body p{margin:.3rem 0}
    .row-detail-body ul{margin:.3rem 0;padding-left:1.3rem}
    .row-detail-body code{font-family:monospace;font-size:.78rem;
                          background:#f0f0f0;border-radius:3px;padding:.1em .3em}
    .view-toggle{background:#1a1a2e;color:#fff;border:none;border-radius:6px;
                 padding:.3rem .8rem;font-size:.82rem;font-weight:600;
                 cursor:pointer;white-space:nowrap}
    .view-toggle:hover{background:#2d2d4e}
  </style>
  <script>
    const DEFAULT_SORT = [
      {col:'verified', dir:'desc'},
      {col:'source',   dir:'asc'},
      {col:'example',  dir:'asc'},
      {col:'dataset',  dir:'asc'},
    ];

    function _applySort(sortKeys) {
      const tbody = document.getElementById('catalog-tbody');
      const rows  = Array.from(tbody.querySelectorAll('tr.catalog-row'));
      rows.sort((a, b) => {
        for (const {col, dir} of sortKeys) {
          const va = (a.dataset[col] || '').toLowerCase();
          const vb = (b.dataset[col] || '').toLowerCase();
          const cmp = va.localeCompare(vb);
          if (cmp !== 0) return dir === 'asc' ? cmp : -cmp;
        }
        return 0;
      });
      rows.forEach(r => {
        tbody.appendChild(r);
        const d = document.getElementById(r.id + '-detail');
        if (d) tbody.appendChild(d);
      });
    }

    function sortTable(col) {
      const table   = document.getElementById('catalog-table');
      const prev    = table.dataset.sortCol;
      const prevDir = table.dataset.sortDir || 'asc';
      const dir     = prev === col && prevDir === 'asc' ? 'desc' : 'asc';
      table.dataset.sortCol = col;
      table.dataset.sortDir = dir;
      _applySort([{col, dir}]);
      document.querySelectorAll('#catalog-table th').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
        if (th.dataset.col === col) th.classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');
      });
    }

    function resetSort() {
      const table = document.getElementById('catalog-table');
      delete table.dataset.sortCol;
      delete table.dataset.sortDir;
      _applySort(DEFAULT_SORT);
      document.querySelectorAll('#catalog-table th').forEach(th =>
        th.classList.remove('sort-asc', 'sort-desc'));
    }

    function toggleDetail(rowId) {
      const row    = document.getElementById(rowId);
      const detail = document.getElementById(rowId + '-detail');
      if (!detail) return;
      const isOpen = row.classList.contains('row-expanded');
      // close all open detail rows
      document.querySelectorAll('#catalog-tbody tr.detail-row').forEach(d => d.style.display = 'none');
      document.querySelectorAll('#catalog-tbody tr.row-expanded').forEach(r => r.classList.remove('row-expanded'));
      if (!isOpen) {
        detail.style.display = '';
        row.classList.add('row-expanded');
      }
    }

    function syncDetailRows() {
      document.querySelectorAll('#catalog-tbody tr.catalog-row').forEach(row => {
        const detail = document.getElementById(row.id + '-detail');
        if (detail && row.style.display === 'none') detail.style.display = 'none';
      });
    }

    function syncCardSort() {
      const table    = document.getElementById('catalog-table');
      const col      = table.dataset.sortCol;
      const dir      = table.dataset.sortDir || 'asc';
      const sortKeys = col ? [{col, dir}] : DEFAULT_SORT;
      document.querySelectorAll('.card-grid').forEach(grid => {
        const cards = Array.from(grid.querySelectorAll('.card'));
        cards.sort((a, b) => {
          for (const {col: c, dir: d} of sortKeys) {
            const va = (a.dataset[c] || '').toLowerCase();
            const vb = (b.dataset[c] || '').toLowerCase();
            const cmp = va.localeCompare(vb);
            if (cmp !== 0) return d === 'asc' ? cmp : -cmp;
          }
          return 0;
        });
        cards.forEach(c => grid.appendChild(c));
      });
    }

    function copyFormulaBtn(btn) {
      var prev = btn.previousElementSibling;
      var text = (prev.tagName === 'CODE' ? prev : prev.querySelector('code')).textContent;
      navigator.clipboard.writeText(text).then(function() {
        var orig = btn.textContent;
        btn.textContent = '\u2713';
        setTimeout(function() { btn.textContent = orig; }, 1200);
      });
    }

    document.addEventListener('DOMContentLoaded', resetSort);
  </script>
</head>
<body>
  <header>
    <div class="header-text">
      <h1>BayesianRegressionModels — Example Catalog</h1>
      <p>$(length(visible)) formulas · $(length(unique(e.source for e in visible))) sources · $(n_verified) verified</p>
    </div>
    <a class="github-link" href="https://github.com/PTWaade/BayesianRegressionModels.jl" target="_blank" rel="noopener">&#8599; GitHub</a>
  </header>
  <nav>
    <input type="hidden" id="active-source" value="">
    $(nav_items)
    <label class="nav-filter">
      <input type="checkbox" id="verified-only"
             _="on change call filterCatalog()">
      verified
    </label>
    <label class="nav-filter">
      <input type="checkbox" id="parseable-only"
             _="on change call filterCatalog()">
      parseable
    </label>
    <label class="nav-filter">
      <input type="checkbox" id="sampleable-only"
             _="on change call filterCatalog()">
      sampleable
    </label>
    <input id="search" type="text" placeholder="Filter by name, formula, dataset…"
           _="on keyup if event.key is 'Escape' set my value to '' end call filterCatalog()">
    <button class="view-toggle" id="view-toggle"
            _="on click
                 if #table-view.style.display is 'none'
                   hide #card-view
                   show #table-view
                   set my textContent to 'Cards'
                 else
                   call syncCardSort()
                   show #card-view
                   hide #table-view
                   set my textContent to 'Table'
                 end">Cards</button>
    <button class="nav-btn" onclick="resetSort()" title="Restore default sort">&#8635; Reset sort</button>
  </nav>
  <main id="card-view" style="display:none">$(sections)</main>
  <div id="table-view">
    <div class="table-wrap"><table id="catalog-table">
      <thead><tr>
        <th data-col="verified"   onclick="sortTable('verified')">&#10003;</th>
        <th data-col="source"     onclick="sortTable('source')">Source</th>
        <th data-col="example"    onclick="sortTable('example')">Example</th>
        <th data-col="dataset"    onclick="sortTable('dataset')">Dataset</th>
        <th data-col="formula"    onclick="sortTable('formula')">Formula</th>
      </tr></thead>
      <tbody id="catalog-tbody">$(table_rows)</tbody>
    </table></div>
  </div>
  <footer>
    <p style="max-width:60ch;margin:.5rem auto">A reference catalog of regression model formulas drawn from textbooks, R and Julia packages,
    and community vignettes — intended as a test suite and inspiration source for
    <a href="https://github.com/PTWaade/BayesianRegressionModels.jl" target="_blank" rel="noopener" style="color:#aaa">BayesianRegressionModels.jl</a>.</p>
    <p>Generated from docstrings in <code>scripts/examples/</code> &mdash; <a href="https://github.com/PTWaade/BayesianRegressionModels.jl" target="_blank" rel="noopener" style="color:#888">&#8599; PTWaade/BayesianRegressionModels.jl</a></p>
  </footer>
</body>
</html>"""
end

function serve(catalog, source_modules::Dict; port::Int=8080, host::String="0.0.0.0")
    @info "Building catalog page…"
    page    = render_page(catalog, source_modules)
    n_vis   = length(filter(e -> !isempty(get(Dict{String,Any}(), "x", "")), catalog))
    @info "Done. Starting server on $host:$port"

    @get "/"           function() html(page) end
    @get "/index.html" function() html(page) end
    oxygen_serve(host=host, port=port, show_banner=false)
end

end # module
