const REF_FREQ = 1.0e9

struct PowerLaw end
struct CurvedPowerLaw end
struct ffa end
struct BrokenPowerLaw end

# SED models

@inline sed(therm_norm, non_therm_norm, spec, freq) =
    therm_norm * (freq / REF_FREQ)^(-0.1) +
    non_therm_norm * (freq / REF_FREQ)^spec

@inline function sed(therm_norm, non_therm_norm, spec, beta, freq, ::CurvedPowerLaw)
    r = log(freq / REF_FREQ)
    therm_norm * (freq / REF_FREQ)^(-0.1) + non_therm_norm * exp((spec + beta * r) * r)
end

@inline function sed(therm_norm, non_therm_norm, spec, tau0, freq, ::ffa)
    ratio = freq / REF_FREQ
    therm_norm * ratio^(-0.1) +
    non_therm_norm * ratio^spec * exp(-tau0 * ratio^(-2.1))
end

@inline function sed(therm_norm, non_therm_norm, spec, nu_b, freq, ::BrokenPowerLaw)
    r = freq / REF_FREQ
    therm = therm_norm * r^(-0.1)
    nt = freq <= nu_b ?
        non_therm_norm * r^spec :
        non_therm_norm * (nu_b / REF_FREQ)^spec * (freq / nu_b)^(spec - 0.5)
    return therm + nt
end

# Log-likelihood

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

function lnlike(params::AbstractVector, freq::AbstractVector,
                data::AbstractVector, data_err::AbstractVector, ::CurvedPowerLaw)::Float64
    therm_norm, non_therm_norm, spec, beta = params[1], params[2], params[3], params[4]
    ll = 0.0
    for j in eachindex(freq)
        model_val = sed(therm_norm, non_therm_norm, spec, beta, freq[j], CurvedPowerLaw())
        ll -= 0.5 * (log(2π * data_err[j]^2) + (data[j] - model_val)^2 / data_err[j]^2)
    end
    return ll
end

function lnlike(params::AbstractVector, freq::AbstractVector,
                data::AbstractVector, data_err::AbstractVector, ::ffa)::Float64
    therm_norm, non_therm_norm, spec, tau0 = params[1], params[2], params[3], params[4]
    ll = 0.0
    for j in eachindex(freq)
        model_val = sed(therm_norm, non_therm_norm, spec, tau0, freq[j], ffa())
        ll -= 0.5 * (log(2π * data_err[j]^2) + (data[j] - model_val)^2 / data_err[j]^2)
    end
    return ll
end

function lnlike(params::AbstractVector, freq::AbstractVector,
                data::AbstractVector, data_err::AbstractVector, ::BrokenPowerLaw)::Float64
    therm_norm, non_therm_norm, spec, nu_b = params[1], params[2], params[3], params[4]
    ll = 0.0
    for j in eachindex(freq)
        model_val = sed(therm_norm, non_therm_norm, spec, nu_b, freq[j], BrokenPowerLaw())
        ll -= 0.5 * (log(2π * data_err[j]^2) + (data[j] - model_val)^2 / data_err[j]^2)
    end
    return ll
end

# Log-prior

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

function lnprior(params::AbstractVector, ha_meas::Float64, ha_uncert::Float64,
                 ::CurvedPowerLaw)::Float64
    therm_norm, non_therm_norm, spec, beta = params[1], params[2], params[3], params[4]
    therm_norm     <= 0.0 && return -Inf
    non_therm_norm <= 0.0 && return -Inf
    spec_prior  = -0.5 * (1.0 / 0.4^2) * (spec + 0.8)^2
    beta_prior  = -0.5 * (1.0 / 0.5^2) * beta^2
    therm_prior = if therm_norm <= ha_meas
        -0.5 * ((therm_norm - ha_meas) / ha_uncert)^2
    else
        0.0
    end
    return spec_prior + beta_prior + therm_prior
end

function lnprior(params::AbstractVector, ha_meas::Float64, ha_uncert::Float64,
                 ::ffa)::Float64
    therm_norm, non_therm_norm, spec, tau0 = params[1], params[2], params[3], params[4]
    therm_norm     <= 0.0 && return -Inf
    non_therm_norm <= 0.0 && return -Inf
    tau0           <  0.0 && return -Inf
    spec_prior  = -0.5 * (1.0 / 0.4^2) * (spec + 0.8)^2
    tau0_prior  = -tau0 / 0.1 
    therm_prior = if therm_norm <= ha_meas
        -0.5 * ((therm_norm - ha_meas) / ha_uncert)^2
    else
        0.0
    end
    return spec_prior + tau0_prior + therm_prior
end

function lnprior(params::AbstractVector, ha_meas::Float64, ha_uncert::Float64,
                 ::BrokenPowerLaw)::Float64
    therm_norm, non_therm_norm, spec, nu_b = params[1], params[2], params[3], params[4]
    therm_norm     <= 0.0  && return -Inf
    non_therm_norm <= 0.0  && return -Inf
    nu_b           <= 1e7  && return -Inf  
    nu_b           >= 1e12 && return -Inf 
    spec_prior  = -0.5 * (1.0 / 0.4^2) * (spec + 0.8)^2
    nu_b_prior  = -log(nu_b)
    therm_prior = therm_norm <= ha_meas ?
        -0.5 * ((therm_norm - ha_meas) / ha_uncert)^2 : 0.0
    return spec_prior + nu_b_prior + therm_prior
end

# Log-posterior 

function lnpost(params::AbstractVector, ha_meas::Float64, ha_uncert::Float64,
                freq::AbstractVector, data::AbstractVector,
                data_err::AbstractVector)::Float64
    lp = lnprior(params, ha_meas, ha_uncert)
    !isfinite(lp) && return lp
    return lp + lnlike(params, freq, data, data_err)
end

function lnpost(params::AbstractVector, ha_meas::Float64, ha_uncert::Float64,
                freq::AbstractVector, data::AbstractVector,
                data_err::AbstractVector, m::CurvedPowerLaw)::Float64
    lp = lnprior(params, ha_meas, ha_uncert, m)
    !isfinite(lp) && return lp
    return lp + lnlike(params, freq, data, data_err, m)
end

function lnpost(params::AbstractVector, ha_meas::Float64, ha_uncert::Float64,
                freq::AbstractVector, data::AbstractVector,
                data_err::AbstractVector, m::ffa)::Float64
    lp = lnprior(params, ha_meas, ha_uncert, m)
    !isfinite(lp) && return lp
    return lp + lnlike(params, freq, data, data_err, m)
end

function lnpost(params::AbstractVector, ha_meas::Float64, ha_uncert::Float64,
                freq::AbstractVector, data::AbstractVector,
                data_err::AbstractVector, m::BrokenPowerLaw)::Float64
    lp = lnprior(params, ha_meas, ha_uncert, m)
    !isfinite(lp) && return lp
    return lp + lnlike(params, freq, data, data_err, m)
end
