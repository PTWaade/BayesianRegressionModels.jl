################
### PREAMBLE ###
################

#TODO:
# - A. Core features
#   - 1. Use nested block indexes so that terms can match across regressions and fixed/random effects
#       - 1.1 and fix the construction of the Z matrices in the evaluation
#   - 2. Refactor to using a single multivariate distribution for separate priors
#   - 3. Overload for gradient compat: to_vec(), from_vec_transform(), to_linked_vec_transform(), from_linked_vec_transform()
# - B. Core Utilities
#   - 1. Make constructor function for RegressionPrior
#   - 2. Make custom summary functionalities (FlexiChains & MCMCChains)
#   - 3. Make custom plotting functions (FlexiChains & MCMCChains)
#   - 4. Make sectioning method for being able to interact with sets of coefficients at a time
#   - 5. Allow for passing "nothing" instead of a LKJCholesky prior to specify that random effects in this block are uncorrelated
# - C. Fixes
#   - 1. Allow for different functional forms of priors? Remove type requirement in RegressionPrior, make check in constructor
#   - 2. Add comments with canonical mathematical notation
#   - 3. Change naming of Cholesky factor
#   - 4. Allow empty fixed effects
# - D. Functionality
#   - 1. unit tests
#   - 2. optimisation
#   - 3. type stability
#   - 3. documentation
# - E. Usage
#   - 1. Fit the example Turing model with FlexiChains
#   - 2. Make example with Horseshoe priors.
#   - 3. Make example with Spike-and-slab priors.
# - F. Near future features
#   - 1. Allow for discrete priors on fixed effects (e.g., spike-and-slab)
#   - 2. Allow for sharing parameters across regressions (e.g., fixed effects beign identical in multiple regressions)
#   - 3. Make constructor for combining multivariate distributions so that they sample vectors
#   - 4. add labels for categorical predictors
# - G. Long Future features
#   - 1. Hierarchical group means
#   - 2. Structured random effects across levels (e.g., gaussian process, AR1, etc.)
#   - 3. Latent mixture models with random effect factor level values or group assignments being latent variables
#   - 4. Variance Component Analysis - letting random effect sd priors come from a multivariate distribution which can weigh between them. Would probably need to be all random effects from one big distribution.

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
# - adding the eps() jitter to the covariance matrices when reconstructing them for logpdf calculations. Is this done right? How large should it be?

### DIFFERENCES TO brms ###
# - instead of the gr() syntax for grouping random effects, the grouping should perhaps be passed as a separate argument.
#   The reason is that a single factor must have the same grouping across all regressions.
#   We could also allow users to specify it in the brms way, but then we would need to check consistency across regressions.



###################################
### DISTRIBUTION IMPLEMENTATION ###
###################################

### SETUP ###
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

## Random effect parameterisation enum ##
@enum RandomEffectParameterization Centered NonCentered

## Dimension names ##
abstract type RegressionDimension{T} <: DimensionalData.Dimension{T} end
@dim RegressionDim RegressionDimension "Regression"                      #r
@dim FixedEffectTermDim RegressionDimension "Fixed Effect Term"          #p
@dim RandomEffectFactorDim RegressionDimension "Random Effect Factor"    #f
@dim RandomEffectTermDim RegressionDimension "Random Effect Term"        #q
@dim RandomEffectGroupDim RegressionDimension "Random Effect Group"      #g
@dim RandomEffectBlockDim RegressionDimension "Correlation Block"        #b
@dim RandomEffectLevelDim RegressionDimension "Random Effect Level"      #l
@dim ObservationDim RegressionDimension "Observation"                    #n


### CONTAINERS ###

## 1. Labels struct, containing labels for all components of the regression ##
struct RegressionLabels{
    Tregressions<:RegressionDim,
    Tfixed_effect_terms<:DimVector,
    Trandom_effect_factors<:RandomEffectFactorDim,
    Trandom_effect_terms<:DimVector,
    Trandom_effect_groups<:DimVector,
    Trandom_effect_blocks<:DimVector,
    Trandom_effect_levels<:DimVector
}

    #Vector (R regressions) of regression labels
    regressions::Tregressions

    #Vector (R regressions) of vectors (P fixed effect terms) of fixed effect labels
    fixed_effect_terms::Tfixed_effect_terms

    #Vector (F random effect factors) of factor labels
    random_effect_factors::Trandom_effect_factors

    #Vector (R regressions) of vectors (F random effect factors) of vectors (Q random effect terms) of random effect term labels
    random_effect_terms::Trandom_effect_terms

    #Vector (F random effect factors) of vectors (G random effect groups) of group labels
    random_effect_groups::Trandom_effect_groups

    #Vector (F random effect factors) of vectors (B random effect blocks) of block labels
    random_effect_blocks::Trandom_effect_blocks

    #Vector (F random effect factors) of vectors (J random effect levels) of level labels
    random_effect_levels::Trandom_effect_levels

end


## 2. Specifications struct, containing information about the model ##
struct RegressionSpecifications{Tgroups<:AbstractVector,Tblocks<:AbstractVector,Tparameterisations<:AbstractVector}

    #Vector (F random effect factors) of vectors (L random effect levels) of group assignments (1:G)
    random_effect_group_assignments::Tgroups

    #Vector (F random effect factors) of vectors (Q_total random effect terms) of block assignments (1:B)
    random_effect_block_assignments::Tblocks

    #Vector (F random effect factors) of RandomEffectParameterization enums
    random_effect_parameterisations::Tparameterisations

    #Labels for components of the regression
    labels::RegressionLabels

end

## 3. Prior struct, distribution that coefficients can be sampled from ##
struct RegressionPriors{Tfixed<:AbstractVector,Tsds<:AbstractVector,Tcorrs<:AbstractVector,Tspecs<:RegressionSpecifications} <: ContinuousMultivariateDistribution

    #Vector (R regressions) of multivariate priors (across P fixed effect terms)
    fixed_effects::Tfixed

    #Vector (R regressions) of vectors (F Random effect factors) of vectors (G random effect groups) of vectors (Q random effect terms)
    random_effect_sds::Tsds

    #Vector (F random effect factors) of vectors (G random effect groups) of vectors (B random effect blocks) of LKJCholesky correlation priors
    random_effect_correlations::Tcorrs

    #Model specifications
    specifications::Tspecs
end

## 4. Coefficients struct ##
struct RegressionCoefficients{Tfixed<:AbstractVector,Tsds<:AbstractVector,Tcorrs<:AbstractVector,Tranef<:AbstractVector, Tspecs<:RegressionSpecifications}

    #Vector (R regressions) of vectors (P fixed effect terms)
    fixed_effects::Tfixed

    #Vector (R regressions) of vectors (F random effect factors) of vectors (G random effect groups) of vectors (Q random effect terms)
    random_effect_sds::Tsds

    #Vector (F random effect factors) of vectors (G random effect groups) of vectors (B random effect blocks) of Cholesky correlations (Q_total random effect terms, Q_total random effect terms)
    random_effect_correlations::Tcorrs

    #Vector (F random effect factors) of matrices (J random effect levels, Q_total random effect terms)
    #Stores actual values or z-scores for centered and non-centered parameterisations respectively
    random_effects::Tranef

    #Model specifications
    specifications::Tspecs
end



### DISTRIBUTION FUNCTIONS ###

## 1. Sampling function ##
function Distributions.rand(rng::AbstractRNG, d::D) where {D<:RegressionPriors}

    ## 0. Extract information ##
    specifications = d.specifications
    labels = specifications.labels

    ## 1. Sample all fixed effects p for each regression r ##
    fixed_effects = DimArray([
            DimArray(rand(rng, d.fixed_effects[At(r)]), labels.fixed_effect_terms[At(r)])
            for r in labels.regressions
        ], labels.regressions)

    ## 2. Sample random effect standard deviations for each term q in each group g per factor f in regression r ##
    random_effect_sds = DimArray([
            DimArray([
                    DimArray([
                            DimArray(
                                [rand(rng, sd_prior) for sd_prior in d.random_effect_sds[At(r)][At(f)][At(g)]],
                                labels.random_effect_terms[At(r)][At(f)]
                            )
                            for g in labels.random_effect_groups[At(f)]
                        ], labels.random_effect_groups[At(f)]) for f in labels.random_effect_factors
                ], labels.random_effect_factors) for r in labels.regressions
        ], labels.regressions)

    ## 3. Sample the random effect correlations ## 
    random_effect_correlations = DimArray([
            DimArray([
                    DimArray([
                            rand(rng, d.random_effect_correlations[At(f)][At(g)][At(b)])
                            for b in labels.random_effect_blocks[At(f)]],
                        labels.random_effect_blocks[At(f)])
                    for g in labels.random_effect_groups[At(f)]
                ], labels.random_effect_groups[At(f)])
            for f in labels.random_effect_factors
        ], labels.random_effect_factors)

    ## 4. Sample the random effects themselves, factor by factor ## 
    #Initialise storage for random effect values
    random_effects = DimArray(Vector{DimArray{Float64,2,<:Tuple{RandomEffectLevelDim,RandomEffectTermDim}}}(undef, length(labels.random_effect_factors)), labels.random_effect_factors) #CURRENTLY HERE

    #Go through each factor f
    for f in labels.random_effect_factors

        # 4.0 setup
        #Extract information about factor f
        group_assignments_f = specifications.random_effect_group_assignments[At(f)]
        block_assignments_f = specifications.random_effect_block_assignments[At(f)]
        parameterisation_f = specifications.random_effect_parameterisations[At(f)]

        random_effect_block_labels_f = labels.random_effect_blocks[At(f)]
        random_effect_group_labels_f = labels.random_effect_groups[At(f)]
        random_effect_level_labels_f = labels.random_effect_levels[At(f)]
        random_effect_term_labels_f = dims(block_assignments_f, RandomEffectTermDim)

        #Initialise empty random effects matrix for factor f
        random_effects_f = DimArray(
            zeros(length(random_effect_level_labels_f), length(random_effect_term_labels_f)),
            (random_effect_level_labels_f, random_effect_term_labels_f)
        )

        #Go through every random effect group g
        for g in random_effect_group_labels_f

            # 4.1 Find levels belonging to this group
            random_effect_level_labels_g = random_effect_level_labels_f[group_assignments_f .== g]

            # 4.2 collect priors for the group g across all regressions r
            sds_g = Float64[]
            for r in labels.regressions
                append!(sds_g, parent(random_effect_sds[At(r)][At(f)][At(g)]))
            end

            #Go through every block b
            for b in random_effect_block_labels_f

                # 4.3 identify random effect terms belonging to this block
                random_effect_term_labels_b = random_effect_term_labels_f[findall(==(b), block_assignments_f)]

                #For non-centered parameterisations
                if parameterisation_f == NonCentered

                    #4.4 Sample random effects as z-scores for this block
                    #Go through each level l
                    for l in random_effect_level_labels_g
                        #Sample random effect z-scores from a standard normal
                        random_effects_f[At(l), At(parent(random_effect_term_labels_b))] = randn(rng, length(random_effect_term_labels_b))
                    end

                    #For centered parameterisations
                elseif parameterisation_f == Centered

                    # 4.5 extract random effect correlations for this block
                    random_effect_correlations_b = random_effect_correlations[At(f)][At(g)][At(b)]

                    #4.6 Extract random effect standard deviations for this block
                    # We need the indices relative to the flattened sds_g vector
                    sds_b = sds_g[findall(==(b), block_assignments_f)]

                    # 4.7 Construct random effect covariances for this block
                    random_effect_covariances_b = Diagonal(sds_b) * random_effect_correlations_b.L

                    # 4.8 Sample random effects for this block
                    #Go through each level l
                    for l in random_effect_level_labels_g
                        #Sample random effects for this block (multiply the Cholesky factor with standard normal samples)
                        random_effects_f[At(l), At(parent(random_effect_term_labels_b))] = random_effect_covariances_b * randn(rng, length(random_effect_term_labels_b))
                    end
                end
            end
        end

        #Store the random effects matrix for this factor
        random_effects[At(f)] = random_effects_f
    end

    return RegressionCoefficients(fixed_effects, random_effect_sds, random_effect_correlations, random_effects, specifications)
end

## 2. Logpdf function ##
function Distributions.logpdf(d::D, x::T) where {D<:RegressionPriors,T<:RegressionCoefficients}

    ## 0. Setup ##
    # Extract information
    specifications = d.specifications
    labels = specifications.labels
    #Initialise logprob
    logprob = 0.0

    ## 1. Add logprob of each fixed effect term p across regressions r ##
    for r in labels.regressions
        logprob += logpdf(d.fixed_effects[At(r)], x.fixed_effects[At(r)])
    end

    ## 2. Add logprob of SDs of each random effect term, across group, factors and regressions ##
    for r in labels.regressions, f in labels.random_effect_factors, g in labels.random_effect_groups[At(f)]
        logprob += sum(logpdf.(d.random_effect_sds[At(r)][At(f)][At(g)], x.random_effect_sds[At(r)][At(f)][At(g)]))
    end

    ## 3. Add logprob for random effects for each group in each factor ##
    #Go through each factor f
    for f in labels.random_effect_factors

        # 3.0 Setup ##
        #Extract information about factor f
        group_assignments_f = specifications.random_effect_group_assignments[At(f)]
        block_assignments_f = specifications.random_effect_block_assignments[At(f)]
        parameterisation_f = specifications.random_effect_parameterisations[At(f)]
        random_effect_term_labels_f = dims(block_assignments_f, RandomEffectTermDim)

        #Go through every group g
        for g in labels.random_effect_groups[At(f)]

            # 3.1 collect sd priors for this group g across all regressions r
            sds_g = Float64[]
            for r in labels.regressions
                append!(sds_g, parent(x.random_effect_sds[At(r)][At(f)][At(g)]))
            end

            # 3.2 Identify levels belonging to this group
            random_effect_levels_g = labels.random_effect_levels[At(f)][group_assignments_f .== g]

            # Go through every correlation block b
            for b in labels.random_effect_blocks[At(f)]

                # 3.3 extract random effect correlations, and add logprob
                random_effect_correlations_b = x.random_effect_correlations[At(f)][At(g)][At(b)]
                logprob += logpdf(d.random_effect_correlations[At(f)][At(g)][At(b)], random_effect_correlations_b)

                # 3.4 identify random effect terms belonging to this block
                random_effect_term_labels_b = random_effect_term_labels_f[block_assignments_f .== b]

                #For non-centered parameterisations
                if parameterisation_f == NonCentered

                    # 3.5 Extract random effect z-scores for this block and group
                    z_scores_b = x.random_effects[At(f)][At(parent(random_effect_levels_g)), At(parent(random_effect_term_labels_b))]

                    # 3.6 Add logprobs for random effect z-scores, using a standard normal
                    logprob += sum(logpdf.(Normal(0, 1), z_scores_b))

                else #for centered parameterisations

                    # 3.7 identify random effect terms belonging to this block
                    random_effect_term_indices_b = findall(==(b), block_assignments_f)
                    sds_b = sds_g[random_effect_term_indices_b]

                    # 3.8 Reconstruct block covariance and random effect distribution
                    L = random_effect_correlations_b.L
                    random_effect_covariances_b = Diagonal(sds_b) * (L * L') * Diagonal(sds_b)
                    random_effect_covariances_b = PDMat(Symmetric(Matrix(random_effect_covariances_b + 1e-8 * I)))
                    dist_b = MvNormal(zeros(length(sds_b)), random_effect_covariances_b)

                    # 3.7 Add logprobs for random effects in this block
                    #Go through each level l in the group g
                    for l in random_effect_levels_g
                        #Add logprob of random effects for this level
                        logprob += logpdf(dist_b, x.random_effects[At(f)][At(l), At(parent(random_effect_term_labels_b))])
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
function get_random_effects(coefficients::Tcoefs) where {Tcoefs<:RegressionCoefficients}

    ## 0. Setup ##
    #Extract information
    specifications = coefficients.specifications
    labels = specifications.labels

    #Initialise storage
    random_effects = DimArray(
        Vector{DimArray{Float64, 2}}(undef, length(labels.random_effect_factors)), 
        labels.random_effect_factors
    )

    #Go through each factor
    for f in labels.random_effect_factors

        ## 0.0 Setup ##
        unprocessed_random_effects_f = coefficients.random_effects[At(f)]
        parameterisation_f = specifications.random_effect_parameterisations[At(f)]

        # 1 Process centered parameterisations
        if parameterisation_f == Centered

            # 1.1 Copy the random effects, since they are already actual values
            random_effects[At(f)] = copy(unprocessed_random_effects_f)

        # 2 Process non-centered parameterisations
        elseif parameterisation_f == NonCentered

            # 2.0 Setup
            #Extract information
            group_assignments_f = specifications.random_effect_group_assignments[At(f)]
            block_assignments_f = specifications.random_effect_block_assignments[At(f)]

            random_effect_level_labels_f = labels.random_effect_levels[At(f)]
            random_effect_term_labels_f = dims(block_assignments_f, RandomEffectTermDim)

            #Initialise storage for processed random effects
            processed_random_effects_f = DimArray(
                zeros(size(unprocessed_random_effects_f)), 
                (random_effect_level_labels_f, random_effect_term_labels_f)
            )

            #Go through every group g
            for g in labels.random_effect_groups[At(f)]

                # 2.1 collect sd priors for this group g across all regressions r
                sds_g = Float64[]
                for r in labels.regressions
                    append!(sds_g, parent(coefficients.random_effect_sds[At(r)][At(f)][At(g)]))
                end

                # 2.2 Identify levels belonging to this group
                random_effect_levels_g = parent(random_effect_level_labels_f[group_assignments_f .== g])

                #Go through every block b
                for b in labels.random_effect_blocks[At(f)]

                    # 2.3 Identify terms and SDs for this block
                    random_effect_term_indices_b = findall(==(b), block_assignments_f)
                    random_effect_term_labels_b = parent(random_effect_term_labels_f[random_effect_term_indices_b])
                    sds_b = sds_g[random_effect_term_indices_b]

                    # 2.4 Reconstruct random effect covariances for this block
                    random_effect_covariances_b = Diagonal(sds_b) * coefficients.random_effect_correlations[At(f)][At(g)][At(b)].L

                    # 2.5 transform z-scored random effects to actual random effects, and store them
                    processed_random_effects_f[At(random_effect_levels_g), At(random_effect_term_labels_b)] = unprocessed_random_effects_f[At(random_effect_levels_g), At(random_effect_term_labels_b)] * random_effect_covariances_b'

                end
            end
            #Store the processed random effects for this factor
            random_effects[At(f)] = processed_random_effects_f
        end
    end

    return random_effects
end

## 2. Getter function for the fixed effects ##
function get_fixed_effects(coefficients::RegressionCoefficients)
    return coefficients.fixed_effects
end

################################
### EVALUATION: DISTRIBUTION ###
################################

## Labels ##
#Names for the different regressions
regression_labels = RegressionDim([:Regression1, :Regression2])

#Names for the fixed effect terms in each regression
fixed_effect_term_labels = DimArray([
        #Regression 1
        FixedEffectTermDim([:r1_Term1, :r1_Term2, :r1_Term3]),
        #Regression 2
        FixedEffectTermDim([:r2_Term1, :r2_Term2, :r2_Term3, :r2_Term4, :r2_Term5])
    ], regression_labels)

#Names for the random effect factors
random_effect_factor_labels = RandomEffectFactorDim([:SubjectFactor, :ItemFactor])

#Names and structure of random effect terms in each regression and factor
random_effect_term_labels = DimArray([
        #Regression 1
        DimArray([
                #Factor 1
                RandomEffectTermDim([:r1_f1_Term1, :r1_f1_Term2]),
                #Factor 2
                RandomEffectTermDim([:r1_f2_Term1])
            ], random_effect_factor_labels),
        #Regression 2
        DimArray([
                #Factor 1
                RandomEffectTermDim([:r2_f1_Term1, :r2_f1_Term2, :r2_f1_Term3]),
                #Factor 2
                RandomEffectTermDim([])
            ], random_effect_factor_labels)
    ], regression_labels)

#Names and structure of random effect groups in each factor
random_effect_group_labels = DimArray([
        #Factor 1
        RandomEffectGroupDim([:f1_HealthyGroup, :f1_SickGroup]),
        #Factor 2
        RandomEffectGroupDim([:f2_AllItemsGroup])
    ], random_effect_factor_labels)

#Names and structure of random effect correlation blocks in each factor
random_effect_block_labels = DimArray([
        #Factor 1
        RandomEffectBlockDim([:f1_Block1, :f1_Block2]),
        #Factor 2
        RandomEffectBlockDim([:f2_Block1])
    ], random_effect_factor_labels)

#Names for individual random effect levels in each factor
random_effect_level_labels = DimArray([
        #Factor 1
        RandomEffectLevelDim([:Subj1, :Subj2, :Subj3, :Subj4]),
        #Factor 2
        RandomEffectLevelDim([:Item1, :Item2, :Item3])
    ], random_effect_factor_labels)

## Regression 1 ##
# 3 fixed effect terms P
# 2 Random effect factors F
#    - Factor 1: 2 groups G, 2 terms Q
#    - Factor 2: 1 group G, 1 term Q

#Fixed effect priors
r1_fixed = product_distribution([Normal(0, 1), Normal(0, 1), Normal(0, 1)])

#Random effect SD priors
r1_sds = DimArray([

    #Factor 1, with 2 terms
    DimArray([
        # Group 1
        DimArray([Gamma(2, 0.1), Gamma(2, 0.1)], random_effect_term_labels[At(:Regression1)][At(:SubjectFactor)]),
        # Group 2
        DimArray([Gamma(2, 0.5), Gamma(2, 0.5)], random_effect_term_labels[At(:Regression1)][At(:SubjectFactor)]),

    ], random_effect_group_labels[At(:SubjectFactor)]),

    #Factor 2, with 1 term
    DimArray([

        # Group 1
        DimArray([Gamma(2, 0.1)], random_effect_term_labels[At(:Regression1)][At(:ItemFactor)]),

    ], random_effect_group_labels[At(:ItemFactor)]),
], random_effect_factor_labels)


## Regression 2 ##
# 5 fixed effect terms P
# 2 Random effect factors F
#    - Factor 1: 2 groups G, 3 terms Q
#    - Factor 2: 1 groups G, 0 terms Q

#Fixed effect priors
r2_fixed = product_distribution([Normal(0, 1), Normal(0, 1), Normal(0, 1), Normal(0, 1), Normal(0, 1)])

#Random effect SD priors
r2_sds = DimArray([

    #Factor 1, with 3 terms
    DimArray([
        
        # Group 1
        DimArray([Gamma(2, 0.1), Gamma(2, 0.1), Gamma(2, 0.1)], random_effect_term_labels[At(:Regression2)][At(:SubjectFactor)]),
        # Group 2
        DimArray([Gamma(2, 0.5), Gamma(2, 0.5), Gamma(2, 0.1)], random_effect_term_labels[At(:Regression2)][At(:SubjectFactor)]),

    ], random_effect_group_labels[At(:SubjectFactor)]),

    #Factor 2, with 0 terms (not used in the regression)
    DimArray([

        # Group 1
        DimArray(Gamma{Float64}[], random_effect_term_labels[At(:Regression2)][At(:ItemFactor)]),

    ], random_effect_group_labels[At(:ItemFactor)]),

], random_effect_factor_labels)


## Random effect correlation priors ##
#Factor 1: 2 groups, 5 total terms across regressions (Reg 1: 2, Reg 2: 3). Block 1 has terms 1-2, block 2 has terms 3-5.
#Factor 2: 1 group, 1 total term across regressions (Reg 1: 1, Reg 2: 0). Block 1 has has term 1.

#Gather all random effect terms for factor 1 in a flattened vector
f1_random_effect_term_labels = RandomEffectTermDim([
    q
    for r in regression_labels
    for q in random_effect_term_labels[At(r)][At(random_effect_factor_labels[1])]
])

#Specify block assignments for each term in factor 1
f1_blocks = DimArray([:f1_Block1, :f1_Block1, :f1_Block2, :f1_Block2, :f1_Block2], f1_random_effect_term_labels)

f1_cor_priors = DimArray([
    #Group 1
    DimArray([
        #Block 1
        LKJCholesky(2, 1.0),
        #Block 2
        LKJCholesky(3, 1.0)
    ], random_effect_block_labels[At(random_effect_factor_labels[1])]),

    #Group 2
    DimArray([
        #Block 1
        LKJCholesky(2, 1.0),
        #Block 2
        LKJCholesky(3, 1.0)
    ], random_effect_block_labels[At(random_effect_factor_labels[1])]),
], random_effect_group_labels[At(random_effect_factor_labels[1])])


#Gather all random effect terms for factor 2 in a flattened vector
f2_random_effect_term_labels = RandomEffectTermDim([
    q
    for r in regression_labels
    for q in random_effect_term_labels[At(r)][At(random_effect_factor_labels[2])]
])

f2_blocks = DimArray([:f2_Block1], f2_random_effect_term_labels)


f2_cor_priors = DimArray([
    #Group 1
    DimArray([
        #Block 1
        LKJCholesky(1, 1.0),
    ], random_effect_block_labels[At(random_effect_factor_labels[2])])
], random_effect_group_labels[At(random_effect_factor_labels[2])])

## Random effect group assignments ##
# Factor 1: 4 subjects (2 Healthy, 2 Sick)
# Factor 2: 3 items (All 1 group)
group_assignments = DimArray([
    #Factor 1
    DimArray([:f1_HealthyGroup, :f1_HealthyGroup, :f1_SickGroup, :f1_SickGroup], random_effect_level_labels[At(random_effect_factor_labels[1])]),
    #Factor 2
    DimArray([:f2_AllItemsGroup, :f2_AllItemsGroup, :f2_AllItemsGroup], random_effect_level_labels[At(random_effect_factor_labels[2])])
], random_effect_factor_labels)

## Random effect parameterisations ##
# Factor 1: Non-centered
# Factor 2: Centered
random_effect_parameterisations = DimArray([
    #Factor 1
    NonCentered,
    #Factor 2
    Centered
], random_effect_factor_labels)

#Final structuring of information
block_assignments = DimArray([f1_blocks, f2_blocks], random_effect_factor_labels)
fixed_effect_priors = DimArray([r1_fixed, r2_fixed], regression_labels)
random_effect_sd_priors = DimArray([r1_sds, r2_sds], regression_labels)
correlation_priors = DimArray([f1_cor_priors, f2_cor_priors], random_effect_factor_labels)

#Initialise specifications and priors
labels = RegressionLabels(
    regression_labels,
    fixed_effect_term_labels,
    random_effect_factor_labels,
    random_effect_term_labels,
    random_effect_group_labels,
    random_effect_block_labels,
    random_effect_level_labels
)
regression_specifications = RegressionSpecifications(group_assignments, block_assignments, random_effect_parameterisations, labels)
priors = RegressionPriors(fixed_effect_priors, random_effect_sd_priors, correlation_priors, regression_specifications)

#Execute
coeffs = rand(priors)
total_logprob = logpdf(priors, coeffs)
fixed_effects = get_fixed_effects(coeffs)
random_effects = get_random_effects(coeffs)


#######################
### FULL REGRESSION ###
#######################

### STRUCTS ###
## Predictors struct ##
struct RegressionPredictors{Tfixedeffects<:DimArray, Trandomeffects<:DimArray, Tlevels<:DimArray}
    #Vector (R regressions) of fixed effect design matrices (N observations x P fixed effect terms)
    fixed_effect_design_matrices::Tfixedeffects

    #Vector (R regressions) of Vector (F factors) of random effect design matrices (N observations x Q random effect terms)
    random_effect_design_matrices::Trandomeffects

    #Vector (R regressions) of vectors (F factors) of vectors (N observations) of level labels
    #Mapping each observation to its random effect level (e.g., Subject 1, Item 2)
    level_labels::Tlevels
end

### REGRESSION FUNCTION ###
## Function for combining predictor data and coefficients to calculate outcomes ##
function linear_regression(predictors::Tpredictors, coefficients::Tcoefficients) where {Tpredictors<:RegressionPredictors,Tcoefficients<:RegressionCoefficients}

    ## 0. Setup ##
    #Extract information
    specifications = coefficients.specifications
    labels = specifications.labels

    #Extract coefficients
    #Vector (R regressions) of vectors (P fixed effect terms)
    fixed_effects = get_fixed_effects(coefficients)
    #Vector (F factors) of matrices (J random effect levels, Q_total random effect terms)
    random_effects = get_random_effects(coefficients)

    #Initialise storage for regression outcomes (vector over R regressions, each with N outcomes)
    outcomes = DimArray(
        Vector{DimArray{Float64, 1}}(undef, length(labels.regressions)), 
        labels.regressions
    )

    #Go through each regression
    for r in labels.regressions

        # 1. Extract labels for the observations in this regression
        observation_labels_r = dims(predictors.fixed_effect_design_matrices[At(r)], ObservationDim)

        # 2. Multiply the fixed effect design matrix with the fixed effects
        outcomes_r = parent(predictors.fixed_effect_design_matrices[At(r)]) * parent(fixed_effects[At(r)])

        #Go through each factor
        for f in labels.random_effect_factors

            # 3. Extract labels for of random effect terms for this regression r and factor f
            random_effect_term_labels_f = labels.random_effect_terms[At(r)][At(f)]

            # If there are no random effect terms for this regression and factor, skip to next factor
            if isempty(random_effect_term_labels_f)
                continue
            end

            # 4. Extract random effects for this factor f and these levels l
            random_effect_level_labels_f = predictors.level_labels[At(r)][At(f)]
            random_effects_l = random_effects[At(f)][At(parent(random_effect_level_labels_f)), At(parent(random_effect_term_labels_f))]

            # 5. Extract random effect design matrix for this regression r and factor f
            random_effect_design_matrix_f = predictors.random_effect_design_matrices[At(r)][At(f)]

            # 6. Multiply random effect design matrix with random effects
            outcomes_r .+= sum(parent(random_effect_design_matrix_f) .* parent(random_effects_l), dims=2)
        end

        #Store outcomes for this regression
        outcomes[At(r)] = DimArray(vec(outcomes_r), observation_labels_r)
    end

    return outcomes
end








#######################################
### EVALUATION: REGRESSION FUNCTION ###
#######################################
using Random
Random.seed!(123)

# 1. Setup Observation IDs
N = 12
# Mapping observations to Subject (1-4) and Item (1-3)


subj_idx = repeat([:Subj1, :Subj2, :Subj3, :Subj4], inner=3)  # [1,1,1, 2,2,2, 3,3,3, 4,4,4]
item_idx = repeat([:Item1, :Item2, :Item3], outer=4)  # [1,2,3, 1,2,3, 1,2,3, 1,2,3]

# 2. Regression 1 Data (3 Fixed Effects)
# X1: Intercept + 2 Predictors
X1 = DimArray(hcat(ones(N), randn(N, 2)), (ObservationDim(1:N), fixed_effect_term_labels[At(:Regression1)]))

# Random Effect Design Matrices for Regression 1
# Factor 1 (Subject) uses 2 terms (e.g., Intercept and first Predictor)
Z1_f1 = X1[:, 1:2]
# Factor 2 (Item) uses 1 term (e.g., Intercept)
Z1_f2 = X1[:, 1:1]

# 3. Regression 2 Data (5 Fixed Effects)
# X2: Intercept + 4 Predictors
X2 = DimArray(hcat(ones(N), randn(N, 4)), (ObservationDim(1:N), fixed_effect_term_labels[At(:Regression2)]))

# Random Effect Design Matrices for Regression 2
# Factor 1 (Subject) uses 3 terms (e.g., Intercept and first two Predictors)
Z2_f1 = X2[:, 1:3]
# Factor 2 (Item) uses 0 terms (As specified in your evaluation: empty vector)
Z2_f2 = X2[:, []]

# Summarize for the model
fixed_effect_design_matrices = DimArray([X1, X2], regression_labels)
random_effect_design_matrices = DimArray([
    DimArray([Z1_f1, Z1_f2], random_effect_factor_labels), # Reg 1: Factor 1, Factor 2
    DimArray([Z2_f1, Z2_f2], random_effect_factor_labels)  # Reg 2: Factor 1, Factor 2
], regression_labels)

level_labels = DimArray([
    #Regression 1
    DimArray([subj_idx, item_idx], random_effect_factor_labels),
    #Regression 2
    DimArray([subj_idx, item_idx], random_effect_factor_labels)
    ], regression_labels)

predictors = RegressionPredictors(fixed_effect_design_matrices, random_effect_design_matrices, level_labels)

coefficients = rand(priors)

outcomes = linear_regression(predictors, coefficients)


################################
### EVALUATION: TURING MODEL ###
################################

@model function m(predictors::RegressionPredictors, priors::RegressionPriors)

    # 1. Sample coefficients
    coefficients ~ priors

    # 2. Calculate outcomes
    outcomes = linear_regression(predictors, coefficients)

    # 3. Here the likelihood would come

    return outcomes
end

model = m(predictors, priors)

chain = sample(model, Prior(), 1000, chain_type=VNChain)
