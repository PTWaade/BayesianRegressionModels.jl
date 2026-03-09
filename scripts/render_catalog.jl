# scripts/render_catalog.jl
#
# Render the catalog HTML page to docs/index.html without starting a server.
# Used by CI (GitHub Actions) to build the GitHub Pages site.
#
# Usage (from the repo root):
#   julia --project=scripts scripts/render_catalog.jl

include(joinpath(@__DIR__, "server.jl"))

@info "Rendering catalog…"
page = render_page()
out  = joinpath(@__DIR__, "..", "docs", "index.html")
mkpath(dirname(out))
write(out, page)
let n_vis = count(e -> !is_hidden(e.source, e.key), CATALOG), n_hid = length(CATALOG) - n_vis
    @info "Written $out ($n_vis entries$(n_hid > 0 ? ", $n_hid hidden" : ""))"
end
