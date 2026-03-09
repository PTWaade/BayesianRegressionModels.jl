# scripts/server.jl
#
# Serve a browsable HTML catalog of all regression examples.
#
# Usage (from the repo root):
#   julia --project=scripts scripts/server.jl        # http://localhost:8080
#   PORT=9090 julia --project=scripts scripts/server.jl
#
# The server binds to 0.0.0.0 so it is reachable from other machines
# on the local network at http://<your-ip>:PORT

using CatalogServer

include(joinpath(@__DIR__, "examples", "all.jl"))

port = parse(Int, get(ENV, "PORT", "8080"))
host = get(ENV, "HOST", "0.0.0.0")

CatalogServer.serve(CATALOG, _SOURCE_MODULES; port, host)
