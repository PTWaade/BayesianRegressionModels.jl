#############
### TYPES ###
#############
## 1. Random effect geometry enumerator ##
@enum RandomEffectGeometry Centered NonCentered

## 2. Prior struct, distribution that coefficients can be sampled from ##
struct RegressionPrior{Tfixed<:MultivariateDistribution,Tsds<:MultivariateDistribution,Tcorrs<:AbstractVector,Tspecs<:RegressionSpecifications} <: ContinuousMultivariateDistribution

    #Multivariate distribution flattened from vector (R regressions) of multivariate priors (across P fixed effect terms)
    fixed_effects::Tfixed

    #Multivariate distribution flattened from vector (R regressions) of vectors (F Random effect factors) of vectors (G random effect groups) of vectors (Q random effect terms)
    random_effect_sds::Tsds

    #Vector (F random effect factors) of vectors (G random effect groups) of vectors (B random effect blocks) of LKJCholesky correlation priors
    random_effect_correlations_cholesky::Tcorrs

    #Model specifications
    specifications::Tspecs
end


##################################
### DISTRIBUTION FUNCTIONALITY ###
##################################
## 1. Sampling function ##
function Distributions.rand(rng::AbstractRNG, d::D) where {D<:RegressionPrior}

    ## 0. Extract information ##
    specifications = d.specifications
    labels = specifications.labels

    ## 1. Sample all fixed effects p for each regression r ##
    fixed_effects_flat = rand(rng, d.fixed_effects)
    
    ## 2. Sample random effect standard deviations for each term q in each group g per factor f in regression r ##
    random_effect_sds_flat = rand(rng, d.random_effect_sds)

    ## 3. Sample the random effect correlations ## 
    random_effect_correlations_cholesky = DimArray([
            DimArray([
                    DimArray([
                        #For every block b
                        begin
                                #Extract the prior
                                prior_random_effect_correlations_cholesky_b = d.random_effect_correlations_cholesky[At(f)][At(g)][At(b)]

                                #If there is no LKJCholesky prior
                                if isnothing(prior_random_effect_correlations_cholesky_b)

                                    #Get the number of terms in this correlation block
                                    n_terms = length(specifications.random_effect_sds_block_indices[At(f)][At(g)][At(b)])
                                    
                                    #Generate a Cholesky object corresponding to an identity covariance matrix implying uncorrelated random effects
                                    Cholesky(Matrix{Float64}(I, n_terms, n_terms), :L, 0)

                                else #If there is a LKJCholesky prior

                                    #Sample a Cholesky object from it
                                    rand(rng, prior_random_effect_correlations_cholesky_b)
                                end
                            end
                            for b in labels.random_effect_blocks[At(f)]],
                        labels.random_effect_blocks[At(f)])
                    for g in labels.random_effect_groups[At(f)]
                ], labels.random_effect_groups[At(f)])
            for f in labels.random_effect_factors
        ], labels.random_effect_factors)

    ## 4. Sample the random effects themselves, factor by factor ## 
    #Initialise storage for random effect values
    random_effects = DimArray(Vector{DimArray{Float64,2,<:Tuple{CategoricalLevelDim,RandomEffectTermDim}}}(undef, length(labels.random_effect_factors)), labels.random_effect_factors) 

    #Go through each factor f
    for f in labels.random_effect_factors

        # 4.0 setup
        #Extract information about factor f
        group_assignments_f = specifications.random_effect_group_assignments[At(f)]
        geometry_f = specifications.random_effect_geometries[At(f)]

        random_effect_block_labels_f = labels.random_effect_blocks[At(f)]
        random_effect_group_labels_f = labels.random_effect_groups[At(f)]
        random_effect_level_labels_f = labels.categorical_levels[At(f)]
        random_effect_term_labels_f = RandomEffectTermDim(vcat([parent(labels.random_effect_terms[At(r)][At(f)]) for r in labels.regressions]...))

        #Initialise empty random effects matrix for factor f
        random_effects_f = DimArray(
            zeros(length(random_effect_level_labels_f), length(random_effect_term_labels_f)),
            (random_effect_level_labels_f, random_effect_term_labels_f)
        )

        #Go through every random effect group g
        for g in random_effect_group_labels_f

            # 4.1 Find levels belonging to this group
            random_effect_level_labels_g = random_effect_level_labels_f[group_assignments_f .== g]

            #Go through every block b
            for b in random_effect_block_labels_f

                # 4.2 extract random effect terms for this block b
                #Initialise storage
                random_effect_term_labels_b = []
                #For each regression r
                for r in labels.regressions
                    #Extract block assignments for this regression and factor
                    random_effect_block_assignments_f = specifications.random_effect_block_assignments[At(r)][At(f)]
                    #For each random effect term q
                    for q in dims(random_effect_block_assignments_f, RandomEffectTermDim)
                        #If the term belongs to this block
                        if random_effect_block_assignments_f[At(q)] == b
                            #Store its label
                            push!(random_effect_term_labels_b, q)
                        end
                    end
                end
                
                # 4.3 If there are no terms in this block, skip to next block
                isempty(random_effect_term_labels_b) && continue

                #For non-centered geometries
                if geometry_f == NonCentered

                    # 4.4 Sample random effects as z-scores for this block
                    #Go through each level l
                    for l in random_effect_level_labels_g
                        #Sample random effect z-scores from a standard normal
                        random_effects_f[At(l), At(parent(random_effect_term_labels_b))] = randn(rng, length(random_effect_term_labels_b))
                    end

                    #For centered geometries
                elseif geometry_f == Centered

                    # 4.5 extract random effect sds for this block
                    random_effect_sds_b = view(random_effect_sds_flat, specifications.random_effect_sds_block_indices[At(f)][At(g)][At(b)])

                    # 4.6 extract random effect correlations for this block
                    random_effect_correlations_cholesky_b = random_effect_correlations_cholesky[At(f)][At(g)][At(b)]

                    # 4.7 Construct random effect covariances for this block
                    random_effect_covariances_cholesky_b = Diagonal(random_effect_sds_b) * random_effect_correlations_cholesky_b.L

                    # 4.8 Sample random effects for this block
                    #Go through each level l
                    for l in random_effect_level_labels_g
                        #Sample random effects for this block (multiply the SD-scaled Cholesky factor with standard normal samples)
                        random_effects_f[At(l), At(parent(random_effect_term_labels_b))] = random_effect_covariances_cholesky_b * randn(rng, length(random_effect_term_labels_b))
                    end
                end
            end
        end

        #Store the random effects matrix for this factor
        random_effects[At(f)] = random_effects_f
    end

    return RegressionCoefficients(fixed_effects_flat, random_effect_sds_flat, random_effect_correlations_cholesky, random_effects, specifications)
end

## 2. Logpdf function ##
function Distributions.logpdf(d::D, x::T) where {D<:RegressionPrior,T<:RegressionCoefficients}

    ## 0. Setup ##
    # Extract information
    specifications = d.specifications
    labels = specifications.labels
    #Initialise logprob
    logprob = 0.0

    ## 1. Add logprob of each fixed effect term p across regressions r ##
    logprob += logpdf(d.fixed_effects, x.fixed_effects_flat)

    ## 2. Add logprob of SDs of each random effect term, across group, factors and regressions ##
    logprob += logpdf(d.random_effect_sds, x.random_effect_sds_flat)

    ## 3. Add logprob for random effects for each group in each factor ##
    #Go through each factor f
    for f in labels.random_effect_factors

        # 3.0 Setup ##
        #Extract information about factor f
        group_assignments_f = specifications.random_effect_group_assignments[At(f)]
        geometry_f = specifications.random_effect_geometries[At(f)]

        #Go through every group g
        for g in labels.random_effect_groups[At(f)]

            # 3.1 Identify levels belonging to this group
            random_effect_levels_g = labels.categorical_levels[At(f)][group_assignments_f .== g]

            # Go through every correlation block b
            for b in labels.random_effect_blocks[At(f)]

                # 3.3 extract random effect correlations, and add logprob
                random_effect_correlations_cholesky_b = x.random_effect_correlations_cholesky[At(f)][At(g)][At(b)]
                prior_random_effect_correlations_cholesky_b = d.random_effect_correlations_cholesky[At(f)][At(g)][At(b)]
                #If there is a LKJCholesky prior
                if !isnothing(prior_random_effect_correlations_cholesky_b)
                    #Add the logprob
                     logprob += logpdf(prior_random_effect_correlations_cholesky_b, random_effect_correlations_cholesky_b)
                end

                # 3.4 Extract random effect terms for this block across all regressions
                #Initialise storage
                random_effect_term_labels_b = Symbol[]
                #For each regression r
                for r in labels.regressions
                    #Extract block assignments for this regression and factor
                    random_effect_block_assignments_f = specifications.random_effect_block_assignments[At(r)][At(f)]
                    #For each random effect term q
                    for q in dims(random_effect_block_assignments_f, RandomEffectTermDim)
                        #If the term belongs to this block
                        if random_effect_block_assignments_f[At(q)] == b
                            #Store its label
                            push!(random_effect_term_labels_b, q)
                        end
                    end
                end

                # 3.5 If there are no terms in this block, skip to next block
                isempty(random_effect_term_labels_b) && continue

                #For non-centered geometries
                if geometry_f == NonCentered

                    # 3.6 Extract random effect z-scores for this block and group
                    z_scores_b = x.random_effects[At(f)][At(parent(random_effect_levels_g)), At(parent(random_effect_term_labels_b))]

                    # 3.7 Add logprobs for random effect z-scores, using a standard normal
                    logprob += sum(logpdf.(Normal(0, 1), z_scores_b))

                elseif geometry_f == Centered #For centered geometries

                    # 3.8 extract random effect sds for this block
                    random_effect_sds_b = view(x.random_effect_sds_flat, specifications.random_effect_sds_block_indices[At(f)][At(g)][At(b)])

                    # 3.9 Calculate the random effect correlation matrix
                    random_effect_correlations_b = random_effect_correlations_cholesky_b.L * random_effect_correlations_cholesky_b.L'

                    # 3.10 Generate the full covariance matrix
                    random_effect_covariances_b = Diagonal(random_effect_sds_b) * random_effect_correlations_b * Diagonal(random_effect_sds_b)
                    random_effect_covariances_b = PDMat(Symmetric(Matrix(random_effect_covariances_b + 1e-8 * I)))

                    # 3.11 Generate the full distribution for the random effects
                    dist_b = MvNormal(zeros(length(random_effect_sds_b)), random_effect_covariances_b)

                    # 3.12 Add logprobs for random effects in this block
                    #Extract the random effects for this block [Levels x Terms]
                    random_effects_b = x.random_effects[At(f)][At(parent(random_effect_levels_g)), At(random_effect_term_labels_b)]
                    #Add logprobs for all random effects in this block
                    logprob += sum(logpdf(dist_b, collect(parent(random_effects_b)')))
                end
            end
        end
    end

    return logprob
end


#############################
### CONSTRUCTOR FUNCTIONS ###
#############################
## 5. Constructor function for generating indices mapping from the flattened coefficient vectors to structured representations and block assignments ##
function generate_indices(labels::RegressionLabels, random_effect_block_assignments::T) where {T<:AbstractVector}

    ## Fixed effects ##
    fixed_effect_indices = DimArray(
        Vector{UnitRange{Int}}(undef, length(labels.regressions)), 
        labels.regressions
    )
    curr_idx = 1
    for r in labels.regressions
        n_terms = length(labels.fixed_effect_terms[At(r)])
        fixed_effect_indices[At(r)] = curr_idx:(curr_idx + n_terms - 1)
        curr_idx += n_terms
    end

    ## Random effect SDs ##
    random_effect_sds_indices = DimArray([
        DimArray([
            DimArray(
                Vector{UnitRange{Int}}(undef, length(labels.random_effect_groups[At(f)])), 
                labels.random_effect_groups[At(f)]
            )
            for f in labels.random_effect_factors
        ], labels.random_effect_factors)
        for r in labels.regressions
    ], labels.regressions)
    curr_idx = 1
    for r in labels.regressions
        for f in labels.random_effect_factors
            for g in labels.random_effect_groups[At(f)]
                n_terms = length(labels.random_effect_terms[At(r)][At(f)])
                random_effect_sds_indices[At(r)][At(f)][At(g)] = curr_idx:(curr_idx + n_terms - 1)
                curr_idx += n_terms
            end
        end
    end

    ## Random effect SDs -> block indices ##
    random_effect_sds_block_indices = DimArray([
        DimArray([
            DimArray([
                # Collect absolute indices across all regressions
                reduce(vcat, [
                    begin
                        r_f_g_range = random_effect_sds_indices[At(r)][At(f)][At(g)]
                        
                        # Find which terms in this Regression-Factor belong to block 'b'
                        # parent() is used to get the raw vector of block symbols
                        local_indices = findall(==(b), parent(random_effect_block_assignments[At(r)][At(f)]))
                        
                        # If local_indices is empty or the range is empty (1:0), 
                        # we return an empty Int vector.
                        if isempty(local_indices) || isempty(r_f_g_range)
                            Int[]
                        else
                            # Map local term indices to absolute indices in the flat vector
                            # We take the start of the range and add the local offsets
                            # local_indices are 1-based, so subtract 1
                            collect((r_f_g_range.start - 1) .+ local_indices)
                        end
                    end
                    for r in labels.regressions
                ])
                for b in labels.random_effect_blocks[At(f)]
            ], labels.random_effect_blocks[At(f)])
            for g in labels.random_effect_groups[At(f)]
        ], labels.random_effect_groups[At(f)])
        for f in labels.random_effect_factors
    ], labels.random_effect_factors)

    return (fixed_effect_indices, random_effect_sds_indices, random_effect_sds_block_indices)

end