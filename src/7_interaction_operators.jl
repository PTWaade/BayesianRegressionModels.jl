########################################
### PRECREATED INTERACTION OPERATORS ###
########################################

## 1. The default multiplication operator ##
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

## 2. Addition operator ##
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
