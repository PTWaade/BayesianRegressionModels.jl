#############
### TYPES ###
#############

## 1. Dimension names ##
abstract type RegressionDimension{T} <: DimensionalData.Dimension{T} end
@dim RegressionDim RegressionDimension "Regression"                      #r
@dim CategoricalVariableDim RegressionDimension "Categorical Variable"   #c
@dim CategoricalLevelDim RegressionDimension "Random Effect Level"       #l
@dim BasisTermDim RegressionDimension "Basis Term"                       #t
@dim FixedEffectTermDim RegressionDimension "Fixed Effect Term"          #p
@dim RandomEffectFactorDim RegressionDimension "Random Effect Factor"    #f
@dim RandomEffectTermDim RegressionDimension "Random Effect Term"        #q
@dim RandomEffectGroupDim RegressionDimension "Random Effect Group"      #g
@dim RandomEffectBlockDim RegressionDimension "Correlation Block"        #b
@dim ObservationDim RegressionDimension "Observation"                    #n

## 2. Labels struct, containing labels for all components of the regression ##
struct RegressionLabels{
    Tregressions,
    Tcategorical_levels,
    Tbasis_terms,
    Tfixed_effect_terms,
    Trandom_effect_factors,
    Trandom_effect_terms,
    Trandom_effect_groups,
    Trandom_effect_blocks,
    Tobservations
}
    #Vector (R regressions) of regression labels
    regressions::Tregressions

    #Vector (C categorical variables) of vectors (L categorical levels) of level labels
    categorical_levels::Tcategorical_levels

    #Vector (R regressions) of vectors (T basis terms) of term labels
    basis_terms::Tbasis_terms

    #Vector (R regressions) of vectors (P fixed effect terms) of fixed effect labels
    fixed_effect_terms::Tfixed_effect_terms

    #Vector (F random effect factors) of factor labels
    random_effect_factors::Trandom_effect_factors

    #Vector (R regressions) of vectors (F random effect factors) of vectors (Q random effect terms) of random effect term labels
    random_effect_terms::Trandom_effect_terms

    #Vector (F random effect factors) of vectors (G random effect groups) of group labels
    random_effect_groups::Trandom_effect_groups

    #Vector (F random effect factors) of vectors (B random effect blocks) of block labels
    random_effect_blocks::Trandom_effect_blocks

    #Vector (R regressions) of vectors (N observations) of observation labels
    observations::Tobservations

end

## 3. Specifications struct, containing information about the model ##
struct RegressionSpecifications{Tgroups<:AbstractVector,Tblocks<:AbstractVector,Tgeometries<:AbstractVector, Tfixed_effects<:AbstractVector,Trandom_effect_sds<:AbstractVector, Trandom_effect_sds_blocks<:AbstractVector}

    #Vector (F random effect factors) of vectors (L random effect levels) of group assignments (1:G)
    random_effect_group_assignments::Tgroups

    #Vector (R regressions) of vectors (F random effect factors) of vectors (Q_total random effect terms) of block assignments (1:B)
    random_effect_block_assignments::Tblocks

    #Vector (F random effect factors) of RandomEffectGeometry enums
    random_effect_geometries::Tgeometries


    #Mapping from flat vector of fixed effect coefficients to its structured format
    fixed_effect_indices::Tfixed_effects

    #Mapping from flat vector of random effect SDs to its structured format
    random_effect_sds_indices::Trandom_effect_sds

    #Mapping from flat vector of random effect SDs to memberships in each random effect block
    random_effect_sds_block_indices::Trandom_effect_sds_blocks

    #Labels for components of the regression
    labels::RegressionLabels

end