## For the regression distribution ##
using Distributions
using LinearAlgebra
using Random
using PDMats #This gives efficient computation for psitive definite matrices, used for the LKJCholesky stuff
using DimensionalData #This allows for named dimensions in arrays
using DimensionalData: @dim

## For the Turing model ##
using Turing
using FlexiChains



include("1_RegressionPriors.jl")

include("2_RegressionPredictors.jl")

include("3_linear_combination.jl")

include("4_TuringExt.jl")

include("5_formula.jl")
