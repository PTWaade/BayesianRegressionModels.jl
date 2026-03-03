macro brm(x)
    dump(x)
end
# The example BRM model is supposed to work with three dataframes (dBMI, dpmean, dpsd), which IIUC, have to have the same numbers of rows in this example and could have been merged into one.
@brm model(dBMI, dpmean, dpsd) = begin 
    # BMI will be a model parameter
    BMI ~ Normal(dBMI.BMI_measured, 1) # equivalently: BMI ~ Normal(BMI_measured, 1) |> (data=dBMI) 
    # Age_first, Age_second are functions of data, and would be computed/updated exactly once
    Age_first, Age_second = ploynomial_expand(dpmean.Age; order=2) # equivalently: Age_first, Age_second = ploynomial_expand(Age; order=2) |> (data=dpmean) 
    # I'm assuming performance_mean is a function of data and model parameters - I think it kind of has to be
    performance_mean ~ 1 + Age_first * Treatment + Age_second + (1 + Treatment | Subject) + (1 + Age_first | Experimenter) |> (data=dpmean)
    # I'm assuming performance_sd is a function of data and model parameters - I think it kind of has to be
    log(performance_sd) ~ 1 + AGE * BMI + max(Age, BMI) + (1 + Age * BMI | Subject) |> (data=dpsd)
    # Peter didn't specify data here - but I think it would have to be specified? Or would it be added to the model via conditioning syntax?
    Performance ~ Normal(performance_mean, performance_sd) # |> (data=observations_df) as an example
    @defaults begin 
        gr(Subject, by=ClinicalGroup,
            Block1=>[Treatment, Age:BMI],
            Block2=>[Age, BMI]
        )
    end
end