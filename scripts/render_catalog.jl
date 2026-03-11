# scripts/render_catalog.jl
#
# Render the catalog HTML page to docs/index.html without starting a server.
# Used by CI (GitHub Actions) to build the GitHub Pages site.
#
# Usage (from the repo root):
#   julia --project=scripts scripts/render_catalog.jl

using CatalogServer

include(joinpath(@__DIR__, "examples", "all.jl"))

@info "Rendering catalog…"
page = CatalogServer.render_page(CATALOG, _SOURCE_MODULES)
out  = joinpath(@__DIR__, "..", "docs", "index.html")
mkpath(dirname(out))
write(out, page)
n_vis = length(CATALOG)
@info "Written $out ($n_vis entries)"
