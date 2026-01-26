## 1. Function for calculating outcomes for a single regression ##
function linear_combination(; 
    fixed_effects::Tfixed_effects,
    random_effects::Trandom_effects, 
    fixed_effect_design_matrix::Tfixed_effects_design_matrix, 
    random_effect_design_matrices::Trandom_effect_design_matrices, 
    random_effect_level_assignments::Trandom_effect_level_assignments, 
    random_effect_term_labels::Trandom_effect_term_labels
    ) where {
        Tfixed_effects, Trandom_effects, Tfixed_effects_design_matrix, Trandom_effect_design_matrices, Trandom_effect_level_assignments, Trandom_effect_term_labels
    }

    # 1. Extract labels for the observations in this regression
    observation_labels = dims(fixed_effect_design_matrix, ObservationDim)

    # 2. Multiply the fixed effect design matrix with the fixed effects
    outcomes = parent(fixed_effect_design_matrix) * parent(fixed_effects)

    #Go through each factor
    for f in labels.random_effect_factors

        # 3. Extract labels for of random effect terms for this factor f
        random_effect_term_labels_f = random_effect_term_labels[At(f)]

        # 4. If there are no random effect terms for this factor, skip to next factor
        isempty(random_effect_term_labels_f) && continue

        # 5. Extract random effects for this factor f and these levels l
        random_effect_level_assignments_f = random_effect_level_assignments[:, At(f)]
        random_effects_l = random_effects[At(f)][parent(random_effect_level_assignments_f), At(parent(random_effect_term_labels_f))]

        # 6. Extract random effect design matrix for this regression r and factor f
        random_effect_design_matrix_f = random_effect_design_matrices[At(f)]

        # 7. Multiply random effect design matrix with random effects
        outcomes .+= sum(parent(random_effect_design_matrix_f) .* parent(random_effects_l), dims=2)
    end

    return DimArray(vec(outcomes), observation_labels)

end


## Medium-level functions for simplifying input, by allowing inputting the predictors as a single object ##
function linear_combination(
    fixed_effects::Tfixed_effects,
    random_effects::Trandom_effects, 
    predictors::Tpredictors,
    labels::Tlabels,
    r::Symbol
    ) where {
        Tfixed_effects, Trandom_effects, Tpredictors<:RegressionPredictors, Tlabels<:RegressionLabels
    }

    return linear_combination(
        fixed_effects = fixed_effects[At(r)], 
        random_effects = random_effects, 
        fixed_effect_design_matrix = predictors.fixed_effect_design_matrices[At(r)],
        random_effect_design_matrices = predictors.random_effect_design_matrices[At(r)],
        random_effect_level_assignments = predictors.random_effect_level_assignments[At(r)],
        random_effect_term_labels = labels.random_effect_terms[At(r)],
    )
end

## Medium-level functions for simplifying input, by allowing inputting specifications instead of labels ##
function linear_combination(
    fixed_effects::Tfixed_effects,
    random_effects::Trandom_effects, 
    predictors::Tpredictors,
    specifications::Tspecifications,
    r::Symbol
    ) where {
        Tfixed_effects, Trandom_effects, Tpredictors<:RegressionPredictors, Tspecifications<:RegressionSpecifications
    }

    return linear_combination(
        fixed_effects,
        random_effects, 
        predictors.labels,
        specifications,
        r
    )
end

## 3. High-level unction for calculating outcomes across a set of regressions ##
function linear_combination(predictors::Tpredictors, coefficients::Tcoefficients) where {Tpredictors<:RegressionPredictors,Tcoefficients<:RegressionCoefficients}

    ## 0. Setup ##
    #Extract information
    labels = coefficients.specifications.labels

    #Extract coefficients
    #Vector (R regressions) of vectors (P fixed effect terms)
    fixed_effects = get_fixed_effects(coefficients)
    #Vector (R regressions) of vectors (F factors) of matrices (J random effect levels, Q_total random effect terms)
    random_effects = get_random_effects(coefficients)

    #Calculate the linear predictions for each regression
    outcomes = DimArray([
        linear_combination(
            fixed_effects, 
            random_effects, 
            predictors, 
            labels, 
            r, 
        )
        for r in labels.regressions
    ], labels.regressions)

    return outcomes
end