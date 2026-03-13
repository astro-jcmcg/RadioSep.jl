using Statistics
using HDF5

function compute_quality_mask(samples_file::String;
    alpha_width_threshold::Float64=0.5,
    tfrac_width_threshold::Float64=0.4)::BitVector
    dataset = h5open(samples_file, "r") do hdf
        read(hdf, "samples")
    end
    n_hex = size(dataset, 1)
    mask = trues(n_hex)

    for i in 1:n_hex
        samp = dataset[i, :, :]

        alpha = samp[:, 3]
        alpha_fin = filter(isfinite, alpha)
        if !isempty(alpha_fin)
            q = quantile(alpha_fin, [0.16, 0.84])
            alpha_width = q[2] - q[1]
            alpha_width > alpha_width_threshold && (mask[i] = false; continue)
        end

        p16, _, p84 = thermal_frac_percentiles(samp)
        (p84 - p16) > tfrac_width_threshold && (mask[i] = false)
    end

    n_masked = count(!, mask)
    println("Quality mask: $(n_hex - n_masked)/$n_hex bins pass ($(n_masked) masked)")
    return mask
end

function thermal_frac_percentiles(samples::AbstractMatrix)::NTuple{3,Float64}
    therm = samples[:, 1]
    non_therm = samples[:, 2]
    frac = therm ./ (therm .+ non_therm)
    p = quantile(frac, [0.16, 0.50, 0.84])
    return (p[1], p[2], p[3])
end

thermal_frac_median(samples::AbstractMatrix) = thermal_frac_percentiles(samples)[2]

function extinction_calc(samples::AbstractMatrix)::Vector{Float64}
    therm_samp = samples[:, 1]
    ha_samp = samples[:, 4]
    ratio = ha_samp ./ therm_samp
    result = fill(NaN, length(ratio))
    pos = ratio .> 0
    result[pos] .= -(2.5 / 2.54) .* log10.(ratio[pos])
    return result
end