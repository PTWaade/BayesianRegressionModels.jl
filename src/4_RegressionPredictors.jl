#############
### TYPES ###
#############
## 1. Abstract type and general function custom expansion functions ##
abstract type AbstractBasisExpansion end
function expand_into_basis_matrix!(value, basis_matrix, indices, expansion_type::AbstractBasisExpansion)
    error("No expansion function has been implemented for the type $(typeof(expansion_type))")
end

## 2. Abstract type for dispatching operators ##
abstract type AbstractInteractionOperator end
function update_interaction!(
    target_column::T_target, 
    basis_matrix::T_basis, 
    component_indices::Vector{Int}, 
    operator_type::AbstractInteractionOperator
) where {T_target<:AbstractVector{<:Real}, T_basis<:AbstractMatrix{<:Real}}
    error("update_interaction! has not been extended for the operator type: $(typeof(operator_type))")
end

## 3. Type for storing which interaction terms depend on ##
struct DependentInteractionIndices
    #Vector of columns in the fixed effects design matrix
    fixed_effects::Vector{Int}

    #Vector (F factors) of Tuples containing the factor and term indices
    random_effects::Vector{Tuple{Int, Int}}
end

## 4. Type that contains BitSets for makring which interactions to recompute ##
struct InteractionUpdateMarkers
    #BitSet of fixed effect design matrix column indices
    fixed_effects::BitSet

    #Vector (F random effect factors) of BitSet of design matrix column indices
    random_effects::Vector{BitSet}
end

## 5. Type for carrying information about a term ##
Base.@kwdef struct TermInfo{T<:AbstractBasisExpansion}
    
    #Type governing how the basis values are expanded
    basis_expansion_type::T

    #The columns in the basis batrix which contain the different levels
    basis_matrix_indices::Vector{Int}

    #Vector (basis matrix column indices) of vectors (corresponding design matrix column indices)
    fixed_effects_indices::Vector{Int}

    #Vector (basis matrix column indices) of vectors (F affected random effect factors) of Tuples (factor index, vector of corresponding design matrix columns)
    random_effects_indices::Vector{Tuple{Int,Vector{Int}}}

    #The column index in the level assignments matrix (0 means no level assignments)
    level_assignments_idx::Int

    #Column indices in the design matrices for interactions that depend on this term
    dependent_interaction_indices::DependentInteractionIndices
end

# 6. Type contianing the recipe for a specific interaction #
struct InteractionRecipe{T<:AbstractInteractionOperator}
    #Columns in the basis matrix that contain the components needed for the interaction
    basis_matrix_indices::Vector{Int}

    #The function used to combine the components. Default is standard multiplication
    operator::T
end

# 7. Struct for keeping predictors and all necessary information #
struct RegressionPredictors{
    Tbasis_matrices<:AbstractMatrix{<:Real},
    Tfixedeffects<:AbstractMatrix{<:Real},
    Trandomeffects<:AbstractVector{<:AbstractArray{<:Real}},
    Tlevels<:AbstractMatrix{Int},
    Tterms_info<:NamedTuple,
    Tfixed_interactions <: AbstractVector{<:Union{Nothing, <:InteractionRecipe}}, 
    Trandom_interactions <: AbstractVector#{<:AbstractVector{<:Union{Nothing, <:InteractionRecipe}}}
    }

    #Matrix (N observations x P+Q total number of predictor terms) which holds all predictors (categorical data in dummy code format) for use in the design matrices
    basis_matrix::Tbasis_matrices

    #Design matrix (N observations x P fixed effect terms)
    fixed_effect_design_matrix::Tfixedeffects

    #Vector (F factors) of random effect design matrices (N observations x Q random effect terms)
    random_effect_design_matrices::Trandomeffects

    #Matrix (N observations X F factors) containing indices mapping observations to random effect levels
    random_effect_level_assignments::Tlevels

    #NamedTuple keeping necessary information about each term
    terms_info::Tterms_info

    #Vector (P fixed effect terms) containing InteractionRecipe objects for each column in the fixed effects design matrix. They will be empty for non-interaction terms.
    fixed_effects_interaction_recipes::Tfixed_interactions
    
    #Vector (F factors) of vector (Q random effect terms) containing InteractionRecipe objects for each column in the fixed effects design matrix. They will be empty for non-interaction terms.
    random_effects_interaction_recipes::Trandom_interactions

    #Object for keeping track of which interactions must be recomputed
    interaction_update_markers::InteractionUpdateMarkers
end


########################
### UPDATE FUNCTIONS ###
########################
## 1. Function for updating basis and design matrices for a term ##
function update_matrices!(
    predictors::Tpredictors, 
    info::Tinfo, 
    values::Tvalues
) where {Tpredictors <: RegressionPredictors, Tinfo <: TermInfo, Tvalues <: AbstractVector}

    # 1.1 Update basis matrix
    expand_into_basis_matrix!(values, predictors.basis_matrix, info.basis_matrix_indices, info.basis_expansion_type)

    # 1.2 Update fixed effects design matrix
    #For each pair of basis and design matrix indices
    for (basis_matrix_idx, design_matrix_idx) in zip(info.basis_matrix_indices, info.fixed_effects_indices)
        #If the design matrix index is used
        if design_matrix_idx > 0
            #Update it
            view(predictors.fixed_effect_design_matrix, :, design_matrix_idx) .= view(predictors.basis_matrix, :, basis_matrix_idx)
        end
    end

    # 1.3 Update each of the random effect design matrices
    #For each factor which has indices to update
    for (f, design_matrix_indices) in info.random_effects_indices
        #Extract the relevant random effect design matrix
        random_effect_design_matrix_f = parent(predictors.random_effect_design_matrices[f])
        
        #For each pair of basis and design matrix indices
        for (basis_matrix_idx, design_matrix_idx) in zip(info.basis_matrix_indices, design_matrix_indices)
            #If the design matrix index is used
            if design_matrix_idx > 0
                #Update it
                view(random_effect_design_matrix_f, :, design_matrix_idx) .= view(parent(predictors.basis_matrix), :, basis_matrix_idx)
            end
        end
    end

    # 1.4 If there is a level assignments index
    if info.level_assignments_idx > 0
        # Update the level assignments matrix
        predictors.random_effect_level_assignments[:, info.level_assignments_idx] .= values
    end
end


## 2. Mediun-level dispatch for updating multiple variables in the predictors object ##
function update_predictors!(
    predictors::Tpredictors, 
    terms::Tuple{Vararg{Symbol}}, 
    values::Tuple{Vararg{AbstractVector}},
    ) where {Tpredictors<:RegressionPredictors}

    #Prepare marker sets for interaction columns to be updated
    interaction_update_markers = predictors.interaction_update_markers
    empty!(interaction_update_markers.fixed_effects)
    for marker_set in interaction_update_markers.random_effects
        empty!(marker_set)
    end

    #Go through each set of term -to-be-updated and corresponding values
    for (term, value) in zip(terms, values)

        #Get the information about the term
        term_info = predictors.terms_info[term]

        #Update the main effects
        update_matrices!(predictors, term_info, value)

        #Mark affected fixed effect columns
        union!(interaction_update_markers.fixed_effects, term_info.dependent_interaction_indices.fixed_effects)

        #Mark affected random effect columns
        for (f, affected_column) in term_info.dependent_interaction_indices.random_effects
            push!(interaction_update_markers.random_effects[f], affected_column)
        end

    end

    #For each fixed effect interaction marked for updating
    for column_idx in predictors.interaction_update_markers.fixed_effects

        # Extract the recipe for calculating it
        interaction_recipe = predictors.fixed_effects_interaction_recipes[column_idx]

        # If the index didn't point to a recipe
        if isnothing(interaction_recipe)
            # Error
            error("Fixed effect design matrix column $column_idx was marked as interaction, but no recipe exists.")
        end

        # Update the interaction
        update_interaction!(
            view(predictors.fixed_effect_design_matrix, :, column_idx),
            predictors.basis_matrix,
            interaction_recipe.basis_matrix_indices,
            interaction_recipe.operator
        )

    end

    #For each random effect factor
    for f in 1:length(predictors.interaction_update_markers.random_effects)
        
        #For each fixed effect interaction marked for updating
        for column_idx in predictors.interaction_update_markers.random_effects[f]

            # Extract the recipe for calculating it
            interaction_recipe = predictors.random_effects_interaction_recipes[f][column_idx]

             # If the index didn't point to a recipe
            if isnothing(interaction_recipe)
                # Error
                error("Random effect design matrix column $column_idx in factor $f was marked as interaction, but no recipe exists.")
            end

            # Update the interaction
            update_interaction!(
                view(predictors.random_effect_design_matrices[f], :, column_idx),
                predictors.basis_matrix,
                interaction_recipe.basis_matrix_indices,
                interaction_recipe.operator
            )
        end
    end    
end


## 3. High-level dispatches for differnet input structures ##
# 3.1 Dispatch when a single regression is targetted
function update_predictors!(
    predictors::Tpredictors, 
    terms::Tuple{Vararg{Symbol}}, 
    values::Tvalues,
    regression_label::Symbol
    ) where {Tpredictors <: AbstractVector{<:RegressionPredictors}, Tvalues <: Tuple{Vararg{<:AbstractVector}}}

    #Extract appropriate predictors object
    predictors_r = predictors[At(regression_label)]

    #Redispatch
    update_predictors!(predictors_r, terms, values)

end

# 3.3 Dispatch when multiple regressions are targetted
function update_predictors!(
    predictors::Tpredictors, 
    terms::Tuple{Vararg{Symbol}}, 
    values::Tuple{Vararg{<:AbstractVector}},
    regression_labels::Vector{Symbol}
    ) where {Tpredictors <: AbstractVector{<:RegressionPredictors}}

    #For each regression
    for regression_label in regression_labels
        #Redispatch
        update_predictors!(predictors, terms, values, regression_label)
    end

end

# 3.3 Global dispatch when all regressions are targetted
function update_predictors!(
    predictors::Tpredictors, 
    terms::Tuple{Vararg{Symbol}}, 
    values::Tuple{Vararg{AbstractVector}},
    regression_label::Nothing
    ) where {Tpredictors <: AbstractVector{<:RegressionPredictors}}

    #For each predictor
    for predictors_r in predictors

        #Redispatch
        update_predictors!(predictors_r, terms, values)

    end
end

# 3.4 Dispatch with a matrix instead of a tuple of vectors
function update_predictors!(
    predictors::Tpredictors, 
    terms::Tuple{Vararg{Symbol}}, 
    values::Tvalues,
    regression_label::Union{Symbol, Vector{Symbol}, Nothing} = nothing
) where {Tpredictors <: AbstractVector{<:RegressionPredictors}, Tvalues <: AbstractMatrix}

    #Make a tuple of vectors from the matrix
    views_tuple = Tuple(view(values, :, i) for i in 1:size(values, 2))

    #Redispatch
    update_predictors!(predictors, terms, views_tuple, regression_label)

end


# 3.5 Dispatch with only a single variable being updated
function update_predictors!(
    predictors::Tpredictors, 
    term::Symbol, 
    values::Tvalues,
    regression_label::Union{Symbol, Vector{Symbol}, Nothing} = nothing
) where {Tpredictors <: AbstractVector{<:RegressionPredictors}, Tvalues <: AbstractVector}
    
    #Make the term and the value into a Vector and a tuple and redispatch
    update_predictors!(predictors, (term,), (values,), regression_label)

end


########################
### GETTER FUNCTIONS ###
########################
## 1. Struct for specifying a linear prediction to extract ##
struct ExtractOutcome{T<:Function}

    #Which regression prediction to use
    regression_label::Symbol

    #Inverse link transformation
    inv_link::T

    #Internal constructor with identity as default inv_link
    function ExtractOutcome(reg::Symbol, inv_link::T = identity) where {T<:Function}
        return new{T}(reg, inv_link)
    end

end

## 2. Struct for specifying a predictor to extract ##
struct ExtractPredictors{T<:Function}

    #Which regression to take predictors from
    regression_label::Symbol
    
    #Which terms to take
    term_labels::Vector{Symbol}

    #Inverse link transformation
    inv_link::T

    #Internal constructor with identity as default inv_link
    function ExtractPredictors(reg::Symbol, terms::Vector{Symbol}, inv_link::T = identity) where {T<:Function}
        return new{T}(reg, terms, inv_link)
    end
end

## 3. Function for extracting specific basis term values from a set of predictors ##
function get_predictor_values(
    predictors::Tpredictors, 
    target::Ttarget
) where {Tpredictors <: AbstractVector{<:RegressionPredictors}, Ttarget <: ExtractPredictors}

    #Extract the appropriate basis matrix columns and transform them with the inverse link function
    return parent( #TODO: remove parent call here if not using DimArray basis matrix
        target.inv_link.(
            view(predictors[At(target.regression_label)].basis_matrix, :, At(target.term_labels))
            )
        )

end