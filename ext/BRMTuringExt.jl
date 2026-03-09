module BRMTuringExt

import BayesianRegressionModels as BRM

include("BRMTuringExt/bijectors.jl")

using Turing
using FlexiChains
include("BRMTuringExt/submodels.jl")

end # module
