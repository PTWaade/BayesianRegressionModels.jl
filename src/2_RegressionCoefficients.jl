#############
### TYPES ###
#############

## 1. Coefficients struct ##
struct RegressionCoefficients{Tfixed<:AbstractVector,Tsds<:AbstractVector,Tcorrs<:AbstractVector,Tranef<:AbstractVector, Tspecs<:RegressionSpecifications}

    #Vector (R regressions) of vectors (P fixed effect terms) - flattened to a single vector
    fixed_effects_flat::Tfixed

    #Vector (R regressions) of vectors (F random effect factors) of vectors (G random effect groups) of vectors (Q random effect terms) - flattened to a single vector
    random_effect_sds_flat::Tsds

    #Vector (F random effect factors) of vectors (G random effect groups) of vectors (B random effect blocks) of Cholesky correlations (Q_total random effect terms, Q_total random effect terms)
    random_effect_correlations_cholesky::Tcorrs

    #Vector (F random effect factors) of matrices (J random effect levels, Q_total random effect terms)
    #Stores actual values or z-scores for centered and non-centered geometries respectively
    random_effects::Tranef

    #Model specifications
    specifications::Tspecs
end

#########################
### UTILITY FUNCTIONS ###
#########################

## 1. Unflattening function for fixed effects ##
function unflatten_fixed_effects(fixed_effects_flat::Vector{R}, specifications::RegressionSpecifications) where {R<:Real}

    # Extract information
    labels = specifications.labels

    # Structure fixed effects properly, and add labels
    fixed_effects = DimArray([
        DimArray(
            view(fixed_effects_flat, specifications.fixed_effect_indices[At(r)]),
            labels.fixed_effect_terms[At(r)]
        )
        for r in labels.regressions
    ], labels.regressions)

    return fixed_effects

end

## 2. Unflattening function for random effect sds ##
function unflatten_random_effect_sds(random_effect_sds_flat::Vector{R}, specifications::RegressionSpecifications) where {R<:Real}

    # Extract information
    labels = specifications.labels

    # Structure random effect sds properly, and add labels
    random_effect_sds = DimArray([
        DimArray([
            DimArray([
                DimArray(
                    view(random_effect_sds_flat, specifications.random_effect_sds_indices[At(r)][At(f)][At(g)]),
                    labels.random_effect_terms[At(r)][At(f)]
                )
                for g in labels.random_effect_groups[At(f)]
            ], labels.random_effect_groups[At(f)]) 
            for f in labels.random_effect_factors
        ], labels.random_effect_factors) 
        for r in labels.regressions
    ], labels.regressions)

    return random_effect_sds

end


########################
### GETTER FUNCTIONS ###
########################
## 1. Getter function for the fixed effects ##
function get_fixed_effects(coefficients::RegressionCoefficients)

    #Unflatten and return fixed effects
    return unflatten_fixed_effects(coefficients.fixed_effects_flat, coefficients.specifications)

end

## 2. Materialiser function for getting actual random effects irrespective of geometry ##
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
        geometry_f = specifications.random_effect_geometries[At(f)]

        # 1. Process non-centered geometries
        if geometry_f == NonCentered

            # 1.0 Setup
            #Extract information
            group_assignments_f = specifications.random_effect_group_assignments[At(f)]
            random_effect_level_labels_f = labels.categorical_levels[At(f)]
            random_effect_term_labels_f = RandomEffectTermDim(vcat([parent(labels.random_effect_terms[At(r)][At(f)]) for r in labels.regressions]...))

            #Initialise storage for processed random effects
            processed_random_effects_f = DimArray(
                zeros(size(unprocessed_random_effects_f)), 
                (random_effect_level_labels_f, random_effect_term_labels_f)
            )

            #Go through every group g
            for g in labels.random_effect_groups[At(f)]

                # 1.1 Identify levels belonging to this group
                random_effect_levels_g = parent(random_effect_level_labels_f[group_assignments_f .== g])

                #Go through every block b
                for b in labels.random_effect_blocks[At(f)]

                    # 1.2 Identify random effect terms and SDs for this block
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

                    # 1.3 If there are no terms in this block, skip to next block
                    isempty(random_effect_term_labels_b) && continue

                    # 1.4 extract random effect sds for this block
                    random_effect_sds_b = view(coefficients.random_effect_sds_flat, specifications.random_effect_sds_block_indices[At(f)][At(g)][At(b)])

                    # 1.5 Reconstruct random effect covariances for this block
                    random_effect_covariances_cholesky_b = Diagonal(random_effect_sds_b) * coefficients.random_effect_correlations_cholesky[At(f)][At(g)][At(b)].L

                    # 1.6 transform z-scored random effects to actual random effects, and store them
                    processed_random_effects_f[At(random_effect_levels_g), At(random_effect_term_labels_b)] = unprocessed_random_effects_f[At(random_effect_levels_g), At(random_effect_term_labels_b)] * random_effect_covariances_cholesky_b'

                end
                
            end
            #Store the processed random effects for this factor
            random_effects[At(f)] = processed_random_effects_f

        # 2. Process centered geometries
        elseif geometry_f == Centered

            # 2.1 Copy the random effects, since they are already actual values
            random_effects[At(f)] = copy(unprocessed_random_effects_f)
        end
    end

    return random_effects
end