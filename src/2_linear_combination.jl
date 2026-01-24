
#######################
### FULL REGRESSION ###
#######################

### STRUCTS ###
## 1. Predictors struct ##
struct RegressionPredictors{Tfixedeffects<:DimArray, Trandomeffects<:DimArray, Tlevels<:DimArray}
    #Vector (R regressions) of fixed effect design matrices (N observations x P fixed effect terms)
    fixed_effect_design_matrices::Tfixedeffects

    #Vector (R regressions) of Vector (F factors) of random effect design matrices (N observations x Q random effect terms)
    random_effect_design_matrices::Trandomeffects

    #Vector (R regressions) of vectors (F factors) of vectors (N observations) of level labels
    #Mapping each observation to its random effect level (e.g., Subject 1, Item 2)
    random_effect_level_assignments::Tlevels
end

### LINEAR PREDICTION FUNCTION ###
## 2. Function for calculating outcomes for a single regression ##
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
















### UTILITY FUNCTIONS ###
## 1. Types for updating
abstract type RegressionUpdateInfo end

struct UpdateContinuousPredictor{T<:Union{Nothing, Symbol}} <: RegressionUpdateInfo
    term_label::Symbol
    regression_label::T
end

struct UdpateCategoricalPredictor{T<:Union{Nothing, Symbol}} <: RegressionUpdateInfo
    term_label::Symbol
    regression_label::T
end

struct UpdateLevelAssignments{T<:Union{Nothing, Symbol}} <: RegressionUpdateInfo
    term_label::Symbol
    factor_label::Symbol
    regression_label::T
end


## 4. Function for updating a term in all regressions ##
function update_predictor(predictors::Tpredictors, values::Tvalues, term::Symbol) where {Tpredictors<:RegressionPredictors, Tvalues<:AbstractVector}

    #Update each regression one at a time
    return foldl((predictors_r, r) -> update_predictor(predictors_r, values, term, r), 
          val(dims(predictors.fixed_effect_design_matrices, RegressionDim)), 
          init = predictors)

end

## 5. Function for updating a term in a single regression ##
function update_predictor(predictors::Tpredictors, values::Tvalues, term::Symbol, r::Symbol) where {Tpredictors<:RegressionPredictors, Tvalues<:AbstractVector}
    
    ## 0. Extract information ##
    fixed_effect_design_matrices = predictors.fixed_effect_design_matrices
    random_effect_design_matrices = predictors.random_effect_design_matrices
    
    ## 1. Update fixed effects ##
    #Extract regression-specific matrix
    fixed_effect_design_matrix_r = fixed_effect_design_matrices[At(r)]
    #Update it
    updated_fixed_effect_design_matrix_r = update_design_matrix(fixed_effect_design_matrix_r, values, term)

    ## 2. Update random effects ##
    #Extract regression-specific matrices
    random_effect_design_matrices_r = random_effect_design_matrices[At(r)]

    #Make new copy of the vector containing the matrices
    updated_random_effect_design_matrices_r = copy(parent(random_effect_design_matrices_r))

    #Go through each factor f
    for idx_f in eachindex(dims(random_effect_design_matrices_r, RandomEffectFactorDim))

        #Replace the matrix with an updated matrix
        updated_random_effect_design_matrices_r[idx_f] = update_design_matrix(random_effect_design_matrices_r[idx_f], values, term)
    end

    #Rebuild the DimArray containing the vectors
    updated_random_effect_design_matrices_r = rebuild(random_effect_design_matrices_r, updated_random_effect_design_matrices_r)

    ## 3. Make copies of design matrix containers ##
    #Get regression integer index
    regression_idx = findfirst(==(r), val(dims(predictors.fixed_effect_design_matrices, RegressionDim)))

    #Copy original containers
    new_fixed_effect_design_matrices = copy(parent(fixed_effect_design_matrices))
    new_random_effect_design_matrices = copy(parent(random_effect_design_matrices))

    #Replace the matrices for the specific regression
    new_fixed_effect_design_matrices[regression_idx] = updated_fixed_effect_design_matrix_r
    new_random_effect_design_matrices[regression_idx] = updated_random_effect_design_matrices_r

    #Rebuild to make sure the format of the containers is identical
    new_fixed_effect_design_matrices = rebuild(fixed_effect_design_matrices, new_fixed_effect_design_matrices)
    new_random_effect_design_matrices = rebuild(random_effect_design_matrices, new_random_effect_design_matrices)
    
    ## 3. Return a new predictor struct ##
    return RegressionPredictors(
        new_fixed_effect_design_matrices,
        new_random_effect_design_matrices,
        predictors.random_effect_level_assignments
    )
end

## 6. Function for replacing a column in a design matrix ##
function update_design_matrix(design_matrix::Tmatrix, values::Tvalues, term::Symbol) where {Tmatrix <: AbstractMatrix, Tvalues <: AbstractVector}
    
    #If the term exists in the design matrix
    if term in val(dims(design_matrix, 2))

        #Find the column index of the term in the design matrix
        term_idx = findfirst(==(term), val(dims(design_matrix, 2)))

        #Return a new design matrix where the target column has been replaced
        return rebuild(design_matrix, hcat(parent(design_matrix)[:, 1:term_idx-1], values, parent(design_matrix)[:, term_idx+1:end]))

    else #If the term does not exist

        #Use the old design matrix
        return design_matrix
    end
end


## 7. Mutation function for updating a term in all regressions ##
function update_predictor!(predictors::Tpredictors, values::Tvalues, term::Symbol) where {Tpredictors<:RegressionPredictors, Tvalues<:AbstractVector}

    #For each regression r
    for r in val(dims(predictors.fixed_effect_design_matrices, RegressionDim))

        #Update the predictor in that regression
        update_predictor!(predictors, values, term, r)

    end

    return nothing
end


## 8. Mutation function for updating a term in a single regression ##
function update_predictor!(predictors::Tpredictors, values::Tvalues, term::Symbol, r::Symbol) where {Tpredictors<:RegressionPredictors, Tvalues<:AbstractVector}
    
    #Update fixed effects
    fixed_effect_design_matrix_r = predictors.fixed_effect_design_matrices[At(r)]
    update_design_matrix!(fixed_effect_design_matrix_r, values, term)

    #Update random effects
    random_effect_design_matrices_r = predictors.random_effect_design_matrices[At(r)]
    for f in dims(random_effect_design_matrices_r, RandomEffectFactorDim)
        update_design_matrix!(random_effect_design_matrices_r[At(f)], values, term)
    end

    return nothing
end


## 9. Mutation function for replacing a column in a design matrix ##
function update_design_matrix!(design_matrix::Tmatrix, values::Tvalues, term::Symbol) where {Tmatrix <: AbstractMatrix, Tvalues <: AbstractVector}
    
    #If the term exists in the design matrix
    if term in val(dims(design_matrix, 2))

        #Update the values in the column
        design_matrix[:, At(term)] .= values

    end

    return nothing
end


## 10. Function for updating level assignments in all regressions ##
function update_level_assignments(predictors::Tpredictors, assignments::Tassignments, f::Symbol) where {Tpredictors <: RegressionPredictors, Tassignments <: AbstractVector}

    # Update level assignments for each regression one at a time
    return foldl((predictors_r, r) -> update_level_assignments(predictors_r, assignments, f, r), 
          val(dims(predictors.random_effect_level_assignments, RegressionDim)), 
          init = predictors)
end


## 11. Function for updating level assignments in a single regression ##
function update_level_assignments(predictors::Tpredictors, assignments::Tassignments, f::Symbol, r::Symbol) where {Tpredictors <: RegressionPredictors, Tassignments <: AbstractVector}
    
    ## 0. Extract information ##
    level_assignments = predictors.random_effect_level_assignments
    
    ## 1. Create copies of the of the container and the level assignments for this regression ##
    new_level_assignments = copy(parent(level_assignments))
    updated_level_assignments_r = copy(level_assignments[At(r)])

    ## 2. Get indeces for the specific regression and factor ##
    regression_idx = findfirst(==(r), val(dims(predictors.random_effect_level_assignments, RegressionDim)))
    factor_idx = findfirst(==(f), val(dims(updated_level_assignments_r, RandomEffectFactorDim)))
    
    ## 3. Update the copy for this regression ##
    parent(updated_level_assignments_r)[:, factor_idx] .= assignments

    ## 4. Replace the updated matrix ##
    new_level_assignments[regression_idx] = updated_level_assignments_r

    ## 5. Rebuild the container to give it the correct labels
    new_level_assignments = rebuild(level_assignments, new_level_assignments)

    ## 6. Return a new struct ##
    return RegressionPredictors(
        predictors.fixed_effect_design_matrices,
        predictors.random_effect_design_matrices,
        new_level_assignments
    )
end


## 12. Mutation function for updating level assignments in all regressions ##
function update_level_assignments!(predictors::Tpredictors, assignments::Tassignments, f::Symbol) where {Tpredictors <: RegressionPredictors, Tassignments <: AbstractVector}

    #For each regression r
    for r in val(dims(predictors.random_effect_level_assignments, RegressionDim))

        #Update level assignments for that regression
        update_level_assignments!(predictors, assignments, f, r)

    end

    return nothing
end


## 13. Mutation function for updating level assignments in a single regression ##
function update_level_assignments!(predictors::Tpredictors, assignments::Tassignments, f::Symbol, r::Symbol) where {Tpredictors <: RegressionPredictors, Tassignments <: AbstractVector}

    #Update the appropriate factor with the level assignments
    predictors.random_effect_level_assignments[At(r)][:, At(f)] .= assignments

    return nothing
end


