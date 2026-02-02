######################################
### PRECREATED EXPANSION FUNCTIONS ###
######################################

## 1 Default identity expansion type and function ##
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

## 2 Default dummy code expansions for categorical variables ##
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



## 3 Functional Basis Expansion ##
# Allows for arbitrary transformations like [x, x^2, log(x)]
struct FunctionalExpansion{T<:Tuple{Vararg{Function}}} <: AbstractBasisExpansion 
    functions::T
end
function expand_into_basis_matrix!(
    values::Tvalues, 
    basis_matrix::Tbasis_matrix, 
    target_indices::Vector{Int}, 
    expansion_type::FunctionalExpansion
) where {Tbasis_matrix<:AbstractMatrix, Tvalues<:AbstractVector{<:Number}}
    
    # Iterate through the list of functions provided in the expansion type
    for (function_idx, col_idx) in enumerate(target_indices)
        
        # Apply the specific function to the raw values and update the basis column
        basis_matrix[:, col_idx] .= expansion_type.functions[function_idx].(values)

    end
end


## 4 Polynomial expansion as a special case ##
function PolynomialExpansion(degree::Int)
    return FunctionalExpansion(Tuple(x -> x^p for p in 1:degree))
end