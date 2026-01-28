#################################
### BASIS EXPANSION FUNCTIONS ###
#################################

## 1.3 Abstract type and general function custom expansion functions ##
abstract type AbstractBasisExpansion end
function expand_into_basis_matrix!(value, basis_matrix, indices, expansion_type::AbstractBasisExpansion)
    error("No expansion function has been implemented for the type $(typeof(expansion_type))")
end

## 1.1 Default identity expansion type and function ##
struct IdentityExpansion <: AbstractBasisExpansion end
function expand_into_basis_matrix!(
    values::Tvalues, 
    basis_matrix::Tbasis_matrix, 
    target_indices::Vector{Int},
    expansion_type::IdentityExpansion
    ) where {Tbasis_matrix<:AbstractMatrix, Tvalues<:AbstractVector}
    
    #Just insert the values in the basis matrix column
    basis_matrix[:, target_indices] .= values
    
end


## 1.2 Default dummy code expansions for categorical variables ##
struct DummyCodeExpansion <: AbstractBasisExpansion end
function expand_into_basis_matrix!(
    values::Tvalues, 
    basis_matrix::Tbasis_matrix, 
    target_indices::Vector{Int}, 
    expansion_type::DummyCodeExpansion
)where {Tbasis_matrix<:AbstractMatrix, Tvalues<:AbstractVector{Int}}
    
    #For each level in the categorical variable
    for (level_idx, col_idx) in enumerate(target_indices)

        #Set a 1 when the level is one above the value (since there is no column for the lowest level)
        basis_matrix[:, col_idx] .= (values .== (level_idx + 1))
        
    end
end


## 1.3 Polynomial expansion ##
struct PolynomialExpansion <: AbstractBasisExpansion end
function expand_into_basis_matrix!(
    values::Tvalues, 
    basis_matrix::Tbasis_matrix, 
    target_indices::Vector{Int}, 
    expansion_type::PolynomialExpansion
    ) where {Tbasis_matrix<:AbstractMatrix, Tvalues<:AbstractVector{<:Number}}
    
    #For each term in the polynomial
    for (polynomial_power, col_idx) in enumerate(target_indices)
        
        #Calculate and insert the transformed value
        basis_matrix[:, col_idx] .= values .^ polynomial_power

    end
end


######################################
### INTERACTION OPERATOR FUNCTIONS ###
######################################

## 1. Abstract type for dispatching operators ##
abstract type AbstractInteractionOperator end
function update_interaction!(
    target_column::T_target, 
    basis_matrix::T_basis, 
    component_indices::Vector{Int}, 
    operator_type::AbstractInteractionOperator
) where {T_target<:AbstractVector{<:Real}, T_basis<:AbstractMatrix{<:Real}}
    error("update_interaction! has not been extended for the operator type: $(typeof(operator_type))")
end


## 2. The default multiplication operator ##
struct MultiplicationOperator <: AbstractInteractionOperator end
function update_interaction!(
    target_column::T_target, 
    basis_matrix::T_basis, 
    component_indices::Vector{Int}, 
    operator_type::MultiplicationOperator
) where {T_target<:AbstractVector{<:Real}, T_basis<:AbstractMatrix{<:Real}}
    
    # Initialize target with the first component
    target_column .= view(basis_matrix, :, component_indices[1])
    
    # For each remaining component
    for i in 2:length(component_indices)
        # Multiply it onto the target
        target_column .*= view(basis_matrix, :, component_indices[i])
    end

end


## 3. Addition operator ##
struct AdditionOperator <: AbstractInteractionOperator end
function update_interaction!(
    target_column::T_target, 
    basis_matrix::T_basis, 
    component_indices::Vector{Int}, 
    operator_type::AdditionOperator
) where {T_target<:AbstractVector{<:Real}, T_basis<:AbstractMatrix{<:Real}}

    # Initialize target with the first component
    target_column .= view(basis_matrix, :, component_indices[1])
    
    # For each remaining component
    for i in 2:length(component_indices)
        # Add it to the target
        target_column .+= view(basis_matrix, :, component_indices[i])
    end

end


#############
### TYPES ###
#############

## 1. Types for keeping information about terms ##
## 1.1 Type for storing which interaction terms depend on ##
struct DependentInteractionIndices
    #Vector of columns in the fixed effects design matrix
    fixed_effects::Vector{Int}

    #Vector (F factors) of Tuples containing the factor and term indices
    random_effects::Vector{Tuple{Int, Int}}
end

## 1.2 Type that contains BitSets for makring which interactions to recompute ##
struct InteractionUpdateMarkers
    #BitSet of fixed effect design matrix column indices
    fixed_effects::BitSet

    #Vector (F random effect factors) of BitSet of design matrix column indices
    random_effects::Vector{BitSet}
end

## 1.4 Abstract type for containers of term information ##
abstract type AbstractTermInfo end

## 1.5 Type for continuous terms ##
Base.@kwdef struct ContinuousTermInfo{T<:AbstractBasisExpansion} <: AbstractTermInfo
    #The column index in the basis matrix
    basis_matrix_indices::Vector{Int}

    #Vector (basis matrix column indices) of vectors (corresponding design matrix column indices)
    fixed_effects_indices::Vector{Vector{Int}}

    #Vector (basis matrix column indices) of vectors (F affected random effect factors) of Tuples (factor index, vector of corresponding design matrix columns)
    random_effects_indices::Vector{Vector{Tuple{Int,Vector{Int}}}}

    #Type governing how the basis values are expanded, default identity
    basis_expansion_type::T = IdentityExpansion()

    #Column indices in the design matrices for interactions that depend on this term
    dependent_interaction_indices::DependentInteractionIndices
end

## 1.7 Type for categorical terms ##
Base.@kwdef struct CategoricalTermInfo{T<:AbstractBasisExpansion} <: AbstractTermInfo
    #The column index in the categorical data
    categorical_variables_index::Int

    #The columns in the basis batrix which contain the different levels
    basis_matrix_indices::Vector{Int}

    #Vector (basis matrix column indices) of vectors (corresponding design matrix column indices)
    fixed_effects_indices::Vector{Vector{Int}}

    #Vector (basis matrix column indices) of vectors (F affected random effect factors) of Tuples (factor index, vector of corresponding design matrix columns)
    random_effects_indices::Vector{Vector{Tuple{Int,Vector{Int}}}}

    #Type governing how the basis values are expanded, default is dummy coding
    basis_expansion_type::T = DummyCodeExpansion()

    #Column indices in the design matrices for interactions that depend on this term
    dependent_interaction_indices::DependentInteractionIndices
end


## 2. Types for keeping information about interactions ##
# 2.1 Type contianing the recipe for a specific interaction #
struct InteractionRecipe{T<:AbstractInteractionOperator}
    #Columns in the basis matrix that contain the components needed for the interaction
    basis_matrix_indices::Vector{Int}

    #The function used to combine the components. Default is standard multiplication
    operator::T
end


# 3. Struct for keeping predictors and all necessary information #
struct RegressionPredictors{
    Tcategorical_variables<:AbstractMatrix{Int},
    Tbasis_matrices<:AbstractMatrix{<:Real},
    Tfixedeffects<:AbstractMatrix{<:Real},
    Trandomeffects<:AbstractVector{<:AbstractMatrix{<:Real}},
    Tlevels<:AbstractMatrix{Int},
    Tterms_info<:NamedTuple,
    Tfixed_interactions <: Vector{<:Union{Nothing, <:InteractionRecipe}}, 
    Trandom_interactions <: Vector{<:Vector{<:Union{Nothing, <:InteractionRecipe}}}
    }

    #Matrix (N observations x number of categorical variables) which holds the categorical data in integer form
    categorical_variables::Tcategorical_variables

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
) where {Tpredictors <: RegressionPredictors, Tinfo <: AbstractTermInfo, Tvalues <: AbstractVector}

    # Update basis matrix
    expand_into_basis_matrix!(values, predictors.basis_matrix, info.basis_matrix_indices, info.basis_expansion_type)

    #For each basis column
    for (i, basis_matrix_idx) in enumerate(info.basis_matrix_indices)

        # Create a view of the data in the basis matrix
        basis_matrix_view = view(predictors.basis_matrix, :, basis_matrix_idx)

        # For the inheriting column in the fixed effect design matrix (vector is empty if it is not used)
        for design_matrix_idx in info.fixed_effects_indices[i]
            # Set the value
            predictors.fixed_effect_design_matrix[:, design_matrix_idx] .= basis_matrix_view
        end

        # For each set of random effect factor and inheriting columns
        for (factor_idx, design_matrix_indices) in info.random_effects_indices[i]
            # For the inheriting column in the fixed effect design matrix (vector is empty if it is not used)
            for design_matrix_indx in design_matrix_indices
                #Set the value
                predictors.random_effect_design_matrices[factor_idx][:, design_matrix_indx] .= basis_matrix_view
            end
        end
    end
end


## 2. Low-level dispatches for distinguishing categorical and continuous terms ##
# 2.1 Continuous terms #
function update_variables!(
    predictors::Tpredictors, 
    info::ContinuousTermInfo, 
    values::Tvalues
    ) where {Tvalues<:AbstractVector, Tpredictors <:RegressionPredictors}
    
    # Update basis and design matrices
    update_matrices!(predictors, info, values)

end

# 2.2 Categorical terms #
function update_variables!(
    predictors::Tpredictors, 
    info::CategoricalTermInfo, 
    values::Tvalues
    ) where {Tpredictors<:RegressionPredictors, Tvalues<:AbstractVector{Int}}
    
    # Update the categorical variables matrix
    predictors.categorical_variables[:, info.categorical_variables_index] .= values

    # Update basis and design matrices
    update_matrices!(predictors, info, values)

end


## 3. Mediun-level dispatch for updating multiple variables in the predictors object ##
function update_variables!(
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
        update_variables!(predictors, term_info, value)

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
# 3.1 Global dispatch when all regressions are targetted
function update_variables!(
    predictors::Tpredictors, 
    terms::Tuple{Vararg{Symbol}}, 
    values::Tuple{Vararg{AbstractVector}},
    regression_label::Nothing
    ) where {Tpredictors <: AbstractVector{<:RegressionPredictors}}

    #For each predictor
    for predictors_r in predictors

        #Redispatch
        update_variables!(predictors_r, terms, values)

    end
end

# 3.2 Dispatch when a single regression is targetted
function update_variables!(
    predictors::Tpredictors, 
    terms::Tuple{Vararg{Symbol}}, 
    values::Tvalues,
    regression_label::Symbol
    ) where {Tpredictors <: AbstractVector{<:RegressionPredictors}, Tvalues <: Tuple{Vararg{<:AbstractVector}}}

    #Extract appropriate predictors object
    predictors_r = predictors[At(regression_label)]

    #Redispatch
    update_variables!(predictors_r, terms, values)

end

# 3.3 Dispatch when multiple regressions are targetted
function update_variables!(
    predictors::Tpredictors, 
    terms::Tuple{Vararg{Symbol}}, 
    values::Tuple{Vararg{<:AbstractVector}},
    regression_labels::Vector{Symbol}
    ) where {Tpredictors <: AbstractVector{<:RegressionPredictors}}

    #For each regression
    for regression_label in regression_labels
        #Redispatch
        update_variables!(predictors, terms, values, regression_label)
    end

end

# 3.4 Dispatch with a matrix instead of a tuple of vectors
function update_variables!(
    predictors::Tpredictors, 
    terms::Tuple{Vararg{Symbol}}, 
    values::Tvalues,
    regression_label::Union{Symbol, Vector{Symbol}, Nothing} = nothing
) where {Tpredictors <: AbstractVector{<:RegressionPredictors}, Tvalues <: AbstractMatrix}

    #Make a tuple of vectors from the matrix
    views_tuple = Tuple(view(values, :, i) for i in 1:size(values, 2))

    #Redispatch
    update_variables!(predictors, terms, views_tuple, regression_label)

end


# 3.5 Dispatch with only a single variable being updated
function update_variables!(
    predictors::Tpredictors, 
    term::Symbol, 
    values::Tvalues,
    regression_label::Union{Symbol, Vector{Symbol}, Nothing} = nothing
) where {Tpredictors <: AbstractVector{<:RegressionPredictors}, Tvalues <: AbstractVector}
    #Make the term and the value into a Vector and a tuple and redispatch
    update_variables!(predictors, (term,), (values,), regression_label)

end



