#####################################
### DEFAULT INTERACTION OPERATORS ###
#####################################
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

########################################
### PRECREATED INTERACTION OPERATORS ###
########################################
## 1. Addition operator ##
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

## 2. Max operator ##
struct MaxOperator <: AbstractInteractionOperator end
function update_interaction!(
    target_column::T_target, 
    basis_matrix::T_basis, 
    component_indices::Vector{Int}, 
    operator_type::MaxOperator
) where {T_target <: AbstractVector{<:Real}, T_basis <: AbstractMatrix{<:Real}}

    # Initialize with the first column
    target_column .= view(basis_matrix, :, component_indices[1])

    # Iteratively apply max for any additional columns (e.g., if you had 3-way interactions)
    for i in 2:length(component_indices)
        target_column .= max.(target_column, view(basis_matrix, :, component_indices[i]))
    end
end