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



include("1_regression_prior.jl")

include("2_linear_combination.jl")

include("3_TuringExt.jl")
