using Statistics
using HDF5

function read_model_attr(samples_file::String)::String
    h5open(samples_file, "r") do hdf
        haskey(HDF5.attributes(hdf), "model") ?
            read(HDF5.attributes(hdf)["model"]) : "PowerLaw"
    end
end

function compute_quality_mask(samples_file::String;
    alpha_width_threshold::Float64=0.5,
    tfrac_width_threshold::Float64=0.4,
    beta_width_threshold::Float64=0.8,
    tau_width_threshold::Float64=1.0,
    nu_b_log_width_threshold::Float64=2.0)::BitVector
    dataset    = h5open(samples_file, "r") do hdf; read(hdf, "samples"); end
    model_str  = read_model_attr(samples_file)
    n_hex      = size(dataset, 1)
    has_4th    = size(dataset, 3) == 5
    mask       = trues(n_hex)

    for i in 1:n_hex
        samp = dataset[i, :, :]

        alpha_fin = filter(isfinite, samp[:, 3])
        if !isempty(alpha_fin)
            q = quantile(alpha_fin, [0.16, 0.84])
            q[2] - q[1] > alpha_width_threshold && (mask[i] = false; continue)
        end

        if has_4th
            col4_fin = filter(isfinite, samp[:, 4])
            if !isempty(col4_fin)
                masked = if model_str == "BrokenPowerLaw"
                    pos = filter(x -> x > 0, col4_fin)
                    if !isempty(pos)
                        q_log = quantile(log10.(pos), [0.16, 0.84])
                        q_log[2] - q_log[1] > nu_b_log_width_threshold
                    else
                        false
                    end
                elseif model_str == "ffa"
                    q = quantile(col4_fin, [0.16, 0.84])
                    q[2] - q[1] > tau_width_threshold
                else
                    q = quantile(col4_fin, [0.16, 0.84])
                    q[2] - q[1] > beta_width_threshold
                end
                masked && (mask[i] = false; continue)
            end
        end

        p16, _, p84 = thermal_frac_percentiles(samp)
        p84 - p16 > tfrac_width_threshold && (mask[i] = false)
    end

    n_masked = count(!, mask)
    println("Quality mask: $(n_hex - n_masked)/$n_hex bins pass ($(n_masked) masked)")
    return mask
end

function thermal_frac_percentiles(samples::AbstractMatrix)::NTuple{3,Float64}
    therm     = samples[:, 1]
    non_therm = samples[:, 2]
    frac      = therm ./ (therm .+ non_therm)
    p         = quantile(frac, [0.16, 0.50, 0.84])
    return (p[1], p[2], p[3])
end

thermal_frac_median(samples::AbstractMatrix) = thermal_frac_percentiles(samples)[2]

function extinction_calc(samples::AbstractMatrix)::Vector{Float64}
    therm_samp = samples[:, 1]
    ha_samp    = samples[:, end]
    ratio      = ha_samp ./ therm_samp
    result     = fill(NaN, length(ratio))
    pos        = ratio .> 0
    result[pos] .= -(2.5 / 2.54) .* log10.(ratio[pos])
    return result
end

function tau0_percentiles(samples::AbstractMatrix)::NTuple{3,Float64}
    tau_fin = filter(isfinite, samples[:, 4])
    isempty(tau_fin) && return (NaN, NaN, NaN)
    p = quantile(tau_fin, [0.16, 0.50, 0.84])
    return (p[1], p[2], p[3])
end

function emission_measure_calc(tau0_samples::AbstractVector;
                                T_e::Float64=8000.0)::Vector{Float64}
    @. tau0_samples * T_e^1.35 / 8.2e-2
end
