const REF_FREQ = 1.0e9  

@inline sed(therm_norm, non_therm_norm, spec, freq) =
    therm_norm * (freq / REF_FREQ)^(-0.1) +
    non_therm_norm * (freq / REF_FREQ)^spec

function lnlike(params::AbstractVector, freq::AbstractVector,
                data::AbstractVector, data_err::AbstractVector)::Float64
    therm_norm, non_therm_norm, spec = params[1], params[2], params[3]
    ll = 0.0
    for j in eachindex(freq)
        model_val = sed(therm_norm, non_therm_norm, spec, freq[j])
        ll -= 0.5 * (log(2π * data_err[j]^2) + (data[j] - model_val)^2 / data_err[j]^2)
    end
    return ll
end

function lnprior(params::AbstractVector, ha_meas::Float64, ha_uncert::Float64)::Float64
    therm_norm, non_therm_norm, spec = params[1], params[2], params[3]
    therm_norm     <= 0.0 && return -Inf
    non_therm_norm <= 0.0 && return -Inf
    spec_prior = -0.5 * (1.0 / 0.4^2) * (spec + 0.8)^2
    therm_prior = if therm_norm <= ha_meas
        -0.5 * ((therm_norm - ha_meas) / ha_uncert)^2
    else
        0.0
    end

    return spec_prior + therm_prior
end

function lnpost(params::AbstractVector, ha_meas::Float64, ha_uncert::Float64,
                freq::AbstractVector, data::AbstractVector,
                data_err::AbstractVector)::Float64
    lp = lnprior(params, ha_meas, ha_uncert)
    !isfinite(lp) && return lp
    return lp + lnlike(params, freq, data, data_err)
end