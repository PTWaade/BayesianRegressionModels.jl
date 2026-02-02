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

## 3 Polynomial expansion ##
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