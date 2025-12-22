################
### PREAMBLE ###
################

#TODO:
# - X. Errors
#   - 1. Fix the MethodError when trying to sample from the model. 
#     Option A: overload Turing transform functions. Complicated, but clean and efficient.
#            to_vec(), from_vec_transform(), to_linked_vec_transform(), from_linked_vec_transform()
#     Option B: don't use a custom variate type, but use a different container (a vector perhaps). Clumsy, possibly easier, uncertain.
#     Option C: don't use a custom distribution, but instead use a custom Turing submodel that samples the coefficients internally. Likely to work, but suddenly TUring-specific
# - A. Core features
#   - 1. Allow for different datasets with different sizes for the different regressions
# - B. Utilities
#   - 1. use containers with named dimensions, store names in specifications (use DimensionalData.jl)
#   - 2. make convenient constructor function for RegressionPrior which deals with dimensionalities and things
#   - 3. update getter functions to be granular, able to get specific regression/factor/group/term
#   - 4. mae summary functionalities for FlexiChains
#   - 5. fancy plotting functions (e.g. grouped plots)
# - C. Functionality
#   - 1. unit tests
#   - 2. optimisation
#   - 3. documentation
# - D. Small fixes
#   - 1. Allow for different functional forms of priors? Remove type requirement in RegressionPrior, make check in constructor
#   - 2. Add comments with canonical mathematical notation
#   - 3. Change naming of Cholesky factor
#   - 4. Allow empty fixed effects
#   - 5. Use some appropriate package (like RecursiveArraytTools.jl) to make the nested vectors and the slicing more manageable
# - E. Usage
#   - 1. FIt the example Turing model with FlexiChains
#   - 2. Make example with Horseshoe priors on something.
# - F. Near future features
#   - 1. Allow for discrete priors on fixed effects (e.g., spike-and-slab)
#   - 2. Allow for sharing parameters across regressions (e.g., fixed effects beign identical in multiple regressions)
# - G. Long Future features
#   - 1. Hierarchical group means
#   - 2. Structured random effects across levels (e.g., gaussian process, AR1, etc.)
#   - 3. Latent mixture models with random effect factor level values or group assignments being latent variables

### TERMINOLOGY ###
# - Regression (r): A single regression model. Multiple can be connected.
# - Fixed effect terms (p): The coefficients associated with the predictors (e.g., Intercept, Age, Gender).
# - Random effect factor (f): The identifier variable (e.g., SubjectID, ItemID).
# - Random effect levels (l): The unique instances within a factor (e.g., Subject 1, Subject 2).
# - Random effect groups (g): Sub-partitioning levels of a factor (Strata) for independent random effect variances and random effects (e.g., healthy vs. clinical). brms syntax: (1 | gr(subjID, by = diagnosis))
# - Random effect terms (q): The variables that vary across a factor (e.g., Intercept, Age, Gender). Is often collapsed across regressions to q_total.
# - Random effect blocks (b): Sets of random effect terms that are internally correlated (e.g., Intercept, Age, Gender). Blocks can be across regressions. brms syntax: (1 + age |p| subjID)
# - observations (n): Each row in the data frame

### FUNCTIONALITY ###
# - multiple fixed effect terms (intercept and multiple predictors)
# - multiple random effect factors (e.g., subjects and items) with multiple random effect terms (intercept and multiple predictors)
# - grouped random effects (e.g., healthy vs. clinical subjects)
# - multiple regressions (e.g., multivariate outcomes)
# - random effect correlations within groups and between terms and regressions
# - multiple random effect correlation blocks within a factor
# - can use centered and non-centered parameterisations for random effects, on a per-factor basis

### CONSTRAINTS ###
# - random effect groups g must be applied across all regressions r 
# - random effect levels l can only belong to a single group g within a factor f
# - there must be entries for each group in each factor for each regression. If a regression does not use a given factor, pass an empty vector instead of a vector with priors for each term.
# - priors must be independent across terms
# - random effect correlations must be specified for all terms across regressions but within a group and within a factor
# - I've made a hard assumption that the covariance matrieces use a LKJCholesky prior, and not a LKJ for example.
# - We do not allow random effect blocks to be different within different random effect groups. Implementationally this would get difficult.
# - The priors over the fixed effects is a multivariate distribution. If there is only a single fied effect, this must still be a multivariate distribution, such as a one-dimensional MvNormal.
# - One perhaps actually limiting constraint is that random effect factor variables must be known at prior construction time, so they cannot be generated during the model. This precludes latent grouping values and infinite mixture models. These can still be created by hand-specifying only that part in Turing, but this won't allow for random effect correlations between these two steps. brms also cannot do this, however.

### POINTS OF UNCERTAINTY ###
# - currenty, the typing of RegressionPrior only accepts vectors where all distributions are of the same type (concrete and fast),
#   or where the user has manually specified the vector type to be of the abstract distribution type (flexible but slow). 
#   Solutions: 1. use Tuples instead (always concrete, but less convenient), 2. do not constrain type parameters (less safe, but more convenient).
# - adding the eps() jitter to the covariance matrices when reconstructing them for logpdf calculations. Is this done right? 

### DIFFERENCES TO brms ###
# - instead of the gr() syntax for grouping random effects, the grouping should perhaps be passed as a separate argument.
#   The reason is that a single factor must have the same grouping across all regressions.
#   We could also allow users to specify it in the brms way, but then we would need to check consistency across regressions.



######################
### IMPLEMENTATION ###
######################

### SETUP ###
## For the regression distribution ##
using Distributions
using LinearAlgebra
using Random
using PDMats #This gives efficient computation for psitive definite matrices, used for the LKJCholesky stuff

## For the Turing model ##
using Turing
using FlexiChains

## Random effect parameterisation enum ##
@enum RandomEffectParameterization Centered NonCentered


### CONTAINERS ###

## 1. Specifications struct, containing information about the model ##
struct RegressionSpecifications
    
    #Vector (F random effect factors) of vectors (L random effect levels) of group assignments (1:G)
    random_effect_group_assignments::Vector{Vector{Int64}}

    #Vector (F random effect factors) of vectors (Q_total random effect terms) of block assignments (1:B)
    random_effect_block_assignments::Vector{Vector{Int64}}

    #Vector (F random effect factors) of RandomEffectParameterization enums
    random_effect_parameterisations::Vector{RandomEffectParameterization}
end

## 2. Prior struct, distribution that coefficients can be sampled from ##
struct RegressionPriors{D1<:ContinuousMultivariateDistribution, D2<:ContinuousUnivariateDistribution, D3<:ContinuousDistribution{CholeskyVariate}} <: ContinuousMultivariateDistribution
    
    #Vector (R regressions) of multivariate priors (across P fixed effect terms)
    fixed_effects::Vector{D1}

    #Vector (R regressions) of vectors (F Random effect factors) of vectors (G random effect groups) of vectors (Q random effect terms)
    random_effect_sds::Vector{Vector{Vector{Vector{D2}}}}

    #Vector (F random effect factors) of vectors (G random effect groups) of vectors (B random effect blocks) of LKJCholesky correlation priors
    random_effect_correlations::Vector{Vector{Vector{D3}}}

    #Model specifications
    specifications::RegressionSpecifications
end

## 3. Coefficients struct ##
struct RegressionCoefficients{T<:Real}

    #Vector (R regressions) of vectors (P fixed effect terms)
    fixed_effects::Vector{Vector{T}}

    #Vector (R regressions) of vectors (F random effect factors) of vectors (G random effect groups) of vectors (Q random effect terms)
    random_effect_sds::Vector{Vector{Vector{Vector{T}}}}

    #Vector (F random effect factors) of vectors (G random effect groups) of vectors (B random effect blocks) of Cholesky correlations (Q_total random effect terms, Q_total random effect terms)
    random_effect_correlations::Vector{Vector{Vector{Cholesky{T,Matrix{T}}}}}

    #Vector (F random effect factors) of matrices (J random effect levels, Q_total random effect terms)
    #Stores actual values or z-scores for centered and non-centered parameterisations respectively
    random_effects::Vector{Matrix{T}}
end

## 4. Predictors struct ##
struct RegressionPredictors{T<:Real}
    #Vector (R regressions) of fixed effect design matrices (N observations x P fixed effect terms)
    fixed_effect_design_matrices::Vector{Matrix{T}}

    #Vector (R regressions) of Vector (F factors) of random effect design matrices (N observations x Q random effect terms)
    random_effect_design_matrices::Vector{Vector{Matrix{T}}}

    #Vector (F factors) of vectors (N observations) of level indices
    #Mapping each observation to its random effect level (e.g., Subject 1, Item 2)
    level_indices::Vector{Vector{Int64}}
end




### DISTRIBUTION FUNCTIONS ###

## 3. Sampling function ##
function Distributions.rand(rng::AbstractRNG, d::D) where {D<:RegressionPriors}

    ## 0. Extract information ##
    specifications = d.specifications
    n_regressions = length(d.fixed_effects)
    n_factors = length(specifications.random_effect_group_assignments)

    ## 1. Sample all fixed effects p for each regression r ##
    fixed_effects = [rand(rng, d.fixed_effects[r]) for r in 1:n_regressions]

    ## 2. Sample random effect standard deviations for each term q in each group g per factor f in regression r ##
    random_effect_sds = [
        [
            [
                [rand(rng, sd_prior) for sd_prior in d.random_effect_sds[r][f][g]]
                for g in eachindex(d.random_effect_sds[r][f])
            ] for f in 1:n_factors
        ] for r in 1:n_regressions
    ]

    ## 3. Sample the random effects and their correlations, factor by factor ## 
    #Initialise storage
    random_effects = Vector{Matrix{Float64}}(undef, n_factors)
    random_effect_correlations = [
        [
            Vector{Cholesky{Float64,Matrix{Float64}}}(undef, length(d.random_effect_correlations[f][g]))
            for g in 1:length(d.random_effect_correlations[f])
        ]
        for f in 1:n_factors
    ]

    #Go through each factor f
    for f in 1:n_factors

        # 3.0 setup
        #Extract information about factor f
        group_assignments_f = specifications.random_effect_group_assignments[f]
        block_assignments_f = specifications.random_effect_block_assignments[f]
        parameterisation_f = specifications.random_effect_parameterisations[f]
        n_levels_f = length(group_assignments_f)
        n_groups_f = length(d.random_effect_correlations[f])
        n_terms_f = length(block_assignments_f)
        unique_blocks_f = unique(block_assignments_f)

        #Initialise empty random effects matrix for factor f
        random_effects_f = zeros(n_levels_f, n_terms_f)

        #Go through every random effect group g
        for g in 1:n_groups_f

            # 3.1 collect priors for the group g across all regressions r
            sds_g = Float64[]
            for r in 1:n_regressions
                append!(sds_g, random_effect_sds[r][f][g])
            end

            #Go through every block b
            for (b_idx, b_id) in enumerate(unique_blocks_f)

                # 3.2 identify random effect terms belonging to this block
                term_indices = findall(==(b_id), block_assignments_f)

                # 3.3 sample and store random effect correlations for this block
                random_effect_correlations_b = rand(rng, d.random_effect_correlations[f][g][b_idx])
                random_effect_correlations[f][g][b_idx] = random_effect_correlations_b

                #For non-centered parameterisations
                if parameterisation_f == NonCentered

                    #3.4 Sample random effects as z-scores for this block
                    #Go through each level l
                    for l in 1:n_levels_f
                        #Check if level l belongs to group g
                        if group_assignments_f[l] == g
                            #Sample random effect z-scores from a standard normal
                            random_effects_f[l, term_indices] = randn(rng, length(term_indices))
                        end
                    end
                    
                #For centered parameterisations
                elseif parameterisation_f == Centered
                    #3.5 Extract random effect standard deviations
                    sds_b = sds_g[term_indices]

                    # 3.6 Construct random effect covariances for this block
                    random_effect_covariances_b = Diagonal(sds_b) * random_effect_correlations_b.L

                    # 3.7 Sample random effects for this block
                    #Go through each level l
                    for l in 1:n_levels_f
                        #Check if level l belongs to group g
                        if group_assignments_f[l] == g
                            #Sample random effects for this block (multiply the Cholesky factor with standard normal samples)
                            random_effects_f[l, term_indices] = random_effect_covariances_b * randn(rng, length(term_indices))
                        end
                    end
                end
            end
        end

        #Store the random effects matrix for this factor
        random_effects[f] = random_effects_f
    end

    return RegressionCoefficients(fixed_effects, random_effect_sds, random_effect_correlations, random_effects)
end

## 4. Logpdf function ##
function Distributions.logpdf(d::D, x::T) where {D<:RegressionPriors,T<:RegressionCoefficients}

    ## 0. Setup ##
    # Extract information
    specifications = d.specifications
    n_regressions = length(d.fixed_effects)
    n_factors = length(specifications.random_effect_group_assignments)
    #Initialise logprob
    logprob = 0.0


    ## 1. Add logprob of each fixed effect term p across regressions r ##
    for r in 1:n_regressions
        logprob += logpdf(d.fixed_effects[r], x.fixed_effects[r])
    end

    ## 2. Add logprob of SDs of each random effect term, across group, factors and regressions ##
    for r in 1:n_regressions, f in 1:n_factors, g in eachindex(d.random_effect_sds[r][f])
        for q in eachindex(d.random_effect_sds[r][f][g])
            logprob += logpdf(d.random_effect_sds[r][f][g][q], x.random_effect_sds[r][f][g][q])
        end
    end

    ## 3. Add logprob for random effects for each group in each factor ##
    #Go through each factor f
    for f in 1:n_factors

        # 3.0 Setup ##
        #Extract information about factor f
        group_assignments_f = specifications.random_effect_group_assignments[f]
        block_assignments_f = specifications.random_effect_block_assignments[f]
        parameterisation_f = specifications.random_effect_parameterisations[f]
        unique_blocks_f = unique(block_assignments_f)

        #For non-centered parameterisations
        if parameterisation_f == NonCentered

            # 3.1 Add logprobs for random effect z-scores, using a standard normal
            logprob += sum(logpdf.(Normal(0, 1), x.random_effects[f]))

            # 3.2 Add logprobs for random effect correlations
            #Go through every group g and block b
            for g in eachindex(x.random_effect_correlations[f]), b_idx in eachindex(unique_blocks_f)
                #Extract and add logprob for random effect correlations in the block
                logprob += logpdf(d.random_effect_correlations[f][g][b_idx], x.random_effect_correlations[f][g][b_idx])
            end

        #For centered parameterisations
        else

            #Go through each group g
            for g in eachindex(x.random_effect_correlations[f])

                # 3.3 collect sd priors for this group g across all regressions r
                sds_g = Float64[]
                for r in 1:n_regressions
                    append!(sds_g, x.random_effect_sds[r][f][g])
                end

                #Go through every block b
                for (b_idx, b_id) in enumerate(unique_blocks_f)

                    # 3.4 identify random effect terms belonging to this block
                    term_indices = findall(==(b_id), block_assignments_f)
                    sds_b = sds_g[term_indices]

                    # 3.5 Extract and add logprob for the block correlation
                    random_effect_correlations_b = x.random_effect_correlations[f][g][b_idx]
                    logprob += logpdf(d.random_effect_correlations[f][g][b_idx], random_effect_correlations_b)

                    # 3.6 Reconstruct block covariance and random effect distribution
                    random_effect_covariances_b = Diagonal(sds_b) * random_effect_correlations_b.L
                    random_effect_covariances_b = PDMat(Cholesky(Matrix(random_effect_covariances_b + eps() * I), 'L', 0))
                    dist_b = MvNormal(zeros(length(term_indices)), random_effect_covariances_b)

                    # 3.7 Add logprobs for random effects in this block
                    #Go through each level l
                    for l in eachindex(group_assignments_f)
                        #Check if level l belongs to group g
                        if group_assignments_f[l] == g
                            #Add logprob of random effects for this level
                            logprob += logpdf(dist_b, x.random_effects[f][l, term_indices])
                        end
                    end
                end
            end
        end
    end

    return logprob
end




#########################
### UTILITY FUNCTIONS ###
#########################


## 1. Materialiser function for getting actual random effects irrespective of parameterisation ##
function get_random_effects(coefficients::RegressionCoefficients, specifications::RegressionSpecifications)

    ## 0. Setup ##
    #Extract information
    n_factors = length(specifications.random_effect_group_assignments)
    #Initialise storage
    random_effects = Vector{Matrix{Float64}}(undef, n_factors)

    #Go through each factor
    for f in 1:n_factors

        ## 0.0 Setup 
        parameterisation_f = specifications.random_effect_parameterisations[f]
        unprocessed_random_effects_f = coefficients.random_effects[f]

        # 1 Process centered parameterisations
        if parameterisation_f == Centered
            
            # 1.1 Copy the random effects, since they are already actual values
            random_effects[f] = copy(unprocessed_random_effects_f)
        
        # 2 Process non-centered parameterisations
        elseif parameterisation_f == NonCentered

            # 2.0 Setup
            #Extract information
            n_levels, n_terms = size(unprocessed_random_effects_f)
            group_assignments_f = specifications.random_effect_group_assignments[f]
            block_assignments_f = specifications.random_effect_block_assignments[f]
            unique_blocks_f = unique(block_assignments_f)

            #Initialise storage for processed random effects
            processed_random_effects_f = zeros(n_levels, n_terms)

            #Go through every group g
            for g in eachindex(coefficients.random_effect_correlations[f])

                # 2.1 collect sd priors for this group g across all regressions r
                sds_g = Float64[]
                for r in eachindex(coefficients.fixed_effects)
                    append!(sds_g, coefficients.random_effect_sds[r][f][g])
                end

                #Go through every block b
                for (b_idx, b_id) in enumerate(unique_blocks_f)

                    # 2.2 identify random effect terms belonging to this block
                    term_indices = findall(==(b_id), block_assignments_f)
                    sds_b = sds_g[term_indices]

                    # 2.3 Reconstruct random effect covariances for this block
                    random_effect_correlations_b = coefficients.random_effect_correlations[f][g][b_idx]
                    random_effect_covariances_b = Diagonal(sds_b) * random_effect_correlations_b.L

                    # 2.4 transform z-scores to actual random effects
                    #Go through each level l
                    for l in 1:n_levels
                        #Check if level l belongs to group g
                        if group_assignments_f[l] == g
                            #Extract z-scores for the block
                            z_b = unprocessed_random_effects_f[l, term_indices]
                            #Transform to actual random effects by multiplying with covariances, and store
                            processed_random_effects_f[l, term_indices] = random_effect_covariances_b * z_b
                        end
                    end
                end
            end
            #Store the processed random effects for this factor
            random_effects[f] = processed_random_effects_f
        end
    end

    return random_effects
end

## 1B. Alternate signature using the RegressionPriors as argument ##
function get_random_effects(coefficients::RegressionCoefficients, priors::RegressionPriors)
    return get_random_effects(coefficients, priors.specifications)
end

## 2. Getter function for the fixed effects ##
function get_fixed_effects(coefficients::RegressionCoefficients, specifications::RegressionSpecifications)
    return coefficients.fixed_effects
end

## 2B. Alternate signature using the RegressionPriors as argument ##
function get_fixed_effects(coefficients::RegressionCoefficients, priors::RegressionPriors)
    return get_fixed_effects(coefficients, priors.specifications)
end



##################
### EVALUATION ###
##################

## Regression 1 ##
# 3 fixed effect terms P
# 2 Random effect factors F
#    - Factor 1: 2 groups G, 2 terms Q
#    - Factor 2: 1 group G, 1 term Q

#Fixed effect priors
r1_fixed = product_distribution([Normal(0, 1), Normal(0, 1), Normal(0, 1)])

#Random effect SD priors
r1_sds = [

    #Factor 1, with 2 terms
    [
        # Group 1
        [Gamma(2, 0.1), Gamma(2, 0.1)],
        # Group 2
        [Gamma(2, 0.5), Gamma(2, 0.5)]
    ],

    #Factor 2, with 1 term
    [
    #Group 1
        [Gamma(2, 0.1)]
    ]
]



## Regression 2 ##
# 5 fixed effect terms P
# 2 Random effect factors F
#    - Factor 1: 2 groups G, 3 terms Q
#    - Factor 2: 1 groups G, 2 terms Q

#Fixed effect priors
r2_fixed = product_distribution([Normal(0, 1), Normal(0, 1), Normal(0, 1), Normal(0, 1), Normal(0, 1)])

#Random effect SD priors
r2_sds = [

    #Factor 1, with 3 terms
    [
        # Group 1
        [Gamma(2, 0.1), Gamma(2, 0.1), Gamma(2, 0.1)],
        # Group 2
        [Gamma(2, 0.5), Gamma(2, 0.5), Gamma(2, 0.1)]
    ],

    #Factor 2, with 0 terms (not used in the regression)
    [
        #Group 1
        Gamma{Float64}[]
    ],]



## Random effect correlation priors ##
#Factor 1: 2 groups, 5 total terms across regressions (Reg 1: 2, Reg 2: 3). Block 1 has terms 1-2, block 2 has terms 3-5.
#Factor 2: 1 group, 1 total term across regressions (Reg 1: 1, Reg 2: 0). Block has has term 1.

f1_blocks = [1, 1, 2, 2, 2]
f1_cor_priors = [
    #Group 1
    [
        #Block 1
        LKJCholesky(2, 1.0), 
        #Block 2
        LKJCholesky(3, 1.0)  
    ],

    #Group 2
    [
        #Block 1
        LKJCholesky(2, 1.0), 
        #Block 2
        LKJCholesky(3, 1.0)  
    ]
]

f2_blocks = [1]
f2_cor_priors = [
#Group 1
    [
        #Block 1
        LKJCholesky(1, 1.0) 
    ]
]

## Random effect group assignments ##
# Factor 1: 4 subjects (2 Healthy, 2 Sick)
# Factor 2: 3 items (All 1 group)
group_assignments = [
    #Factor 1
    [1, 1, 2, 2],
    #Factor 2
    [1, 1, 1]
]

## Random effect parameterisations ##
# Factor 1: Non-centered
# Factor 2: Centered
random_effect_parameterisations = [
    #Factor 1
    NonCentered,
    #Factor 2
    Centered
]

#Initialise specifications
block_assignments = [f1_blocks, f2_blocks]
regression_specifications = RegressionSpecifications(group_assignments, block_assignments, random_effect_parameterisations)

#Initialise priors 
fixed_effect_priors = [r1_fixed, r2_fixed]
random_effect_sd_priors = [r1_sds, r2_sds]
correlation_priors = [f1_cor_priors, f2_cor_priors]
priors = RegressionPriors(fixed_effect_priors, random_effect_sd_priors, correlation_priors, regression_specifications)

# Execute
coeffs = rand(priors)
total_logprob = logpdf(priors, coeffs)
fixed_effects = get_fixed_effects(coeffs, regression_specifications)
random_effects = get_random_effects(coeffs, regression_specifications)






######################
### PREDICTOR DATA ###
######################
using Random
Random.seed!(123)

# 1. Setup Observation IDs
N = 12
# Mapping observations to Subject (1-4) and Item (1-3)
subj_idx = repeat(1:4, inner=3)  # [1,1,1, 2,2,2, 3,3,3, 4,4,4]
item_idx = repeat(1:3, outer=4)  # [1,2,3, 1,2,3, 1,2,3, 1,2,3]

# 2. Regression 1 Data (3 Fixed Effects)
# X1: Intercept + 2 Predictors
X1 = hcat(ones(N), randn(N, 2))

# Random Effect Design Matrices for Regression 1
# Factor 1 (Subject) uses 2 terms (e.g., Intercept and first Predictor)
Z1_f1 = X1[:, 1:2] 
# Factor 2 (Item) uses 1 term (e.g., Intercept)
Z1_f2 = X1[:, 1:1] 

# 3. Regression 2 Data (5 Fixed Effects)
# X2: Intercept + 4 Predictors
X2 = hcat(ones(N), randn(N, 4))

# Random Effect Design Matrices for Regression 2
# Factor 1 (Subject) uses 3 terms (e.g., Intercept and first two Predictors)
Z2_f1 = X2[:, 1:3]
# Factor 2 (Item) uses 0 terms (As specified in your evaluation: empty vector)
Z2_f2 = zeros(N, 0) 

# Summarize for the model
fixed_effect_design_matrices = [X1, X2]
random_effect_design_matrices = [
    [Z1_f1, Z1_f2], # Reg 1: Factor 1, Factor 2
    [Z2_f1, Z2_f2]  # Reg 2: Factor 1, Factor 2
]
level_indices = [subj_idx, item_idx]

predictors = RegressionPredictors(fixed_effect_design_matrices, random_effect_design_matrices, level_indices)





####################
### TURING MODEL ###
####################

@model function linear_regression(predictors::RegressionPredictors, priors::RegressionPriors)
    
    # 0. Setup
    #Extract information
    n_regressions = length(predictors.fixed_effect_design_matrices)
    n_factors = length(predictors.level_indices)
    #Initialise storage for regression outcomes (vector over R regressions, each with N outcomes)
    outcomes = Vector{Vector{Float64}}(undef, n_regressions)
    #Initialise counter for which column in the random effects belong to the current regression
    random_effect_column_offsets = ones(Int, n_factors) 
    
    # 1. Sample coefficients
    coefficients ~ priors

    # 2. extract fixed effects and random effects
    # Vector (R regressions) of vectors (P fixed effect terms)
    fixed_effects = get_fixed_effects(coefficients, priors)
    # Vector (F factors) of matrices (J random effect levels, Q_total random effect terms)
    random_effects = get_random_effects(coefficients, priors)

    #Go through each regression
    for r in 1:n_regressions

        # 3. Multiply the fixed effect design matrix with the fixed effects
        outcomes_r = predictors.fixed_effect_design_matrices[r] * fixed_effects[r]

        #Go through each factor
        for f in 1:n_factors

            # 4. Extract number of random effect terms for this regression and factor
            n_terms_f = size(predictors.random_effect_design_matrices[r][f], 2)

            #If there are random effects
            if n_terms_f > 0

                # 5. Get the column range in nthe random effect matrix corresponding to this regression and factor
                random_effect_column_range = random_effect_column_offsets[f]:(random_effect_column_offsets[f] + n_terms_f - 1)

                # 6. Identify which random effect level each observation belongs to
                observation_levels = predictors.level_indices[f]

                # 7. Slice the random effect coefficients for these specific levels and terms
                random_effect_slice = random_effects[f][observation_levels, random_effect_column_range]

                # 8. Multiply random effect 
                outcomes_r += sum(predictors.random_effect_design_matrices[r][f] .* random_effect_slice, dims=2)[:]

                # 9. Update the column offset for the next regression
                random_effect_column_offsets[f] += n_terms_f
            end
        end
        #Store outcomes for this regression
        outcomes[r] = outcomes_r
    end

    return outcomes
end

model = linear_regression(predictors, priors)

chain = sample(model, Prior(), 1000, chain_type=VNChain)

