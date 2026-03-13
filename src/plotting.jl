using CairoMakie
using PairPlots
using HDF5
using Statistics
using Printf
using FITSFiles

function _read_samples(samples_file::String)
    h5open(samples_file, "r") do hdf
        read(hdf, "samples")
    end
end

thermal(therm_norm, freq) = therm_norm .* (freq ./ 1e9) .^ (-0.1)
non_thermal(nt_norm, spec, freq) = nt_norm .* (freq ./ 1e9) .^ spec

function plot_sed(fd::FluxData, freq::AbstractVector, samples_file::String, output_dir::String;
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset = _read_samples(samples_file)
    n_hex = size(dataset, 1)
    sed_dir = joinpath(output_dir, "SED")
    mkpath(sed_dir)

    freq_range = 10 .^ range(8.5, 11.0; length=200)

    @showprogress "Plotting SEDs: " for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        samp = dataset[i, :, :]

        fig = Figure()
        ax = Axis(fig[1, 1];
            xlabel="Frequency (Hz)", ylabel="Flux Density (Jy)",
            xscale=log10, yscale=log10,
            limits=((10^8.5, 10^11), nothing))

        errorbars!(ax, freq, fd.dmatrix[i, :], fd.dnoisematrix[i, :];
            color=:black)
        scatter!(ax, freq, fd.dmatrix[i, :]; color=:black)

        errorbars!(ax, [1e9], [fd.dHa[i]], [0.5 * fd.dHa[i]];
            color=:red, whiskerwidth=8)
        scatter!(ax, [1e9], [fd.dHa[i]]; color=:red)

        best_T = median(samp[:, 1])
        best_nT = median(samp[:, 2])
        best_spec = median(samp[:, 3])
        lines!(ax, freq_range, sed.(best_T, best_nT, best_spec, freq_range);
            color=:black)

        idx = rand(1:size(samp, 1), 250)
        for j in idx
            samp[j, 1] > 0 && lines!(ax, freq_range, thermal(samp[j, 1], freq_range);
                color=(:red, 0.0075), linewidth=10)
            samp[j, 2] > 0 && lines!(ax, freq_range,
                non_thermal(samp[j, 2], samp[j, 3], freq_range);
                color=(:blue, 0.0075), linewidth=10)
        end

        ymin = minimum(filter(>(0), fd.dmatrix[i, :] .- fd.dnoisematrix[i, :]);
            init=1e-10)
        ymax = maximum(fd.dmatrix[i, :] .+ fd.dnoisematrix[i, :]; init=1e-5)
        ylims!(ax, ymin * 0.5, ymax * 2)

        save(joinpath(sed_dir, "$i.png"), fig)
        empty!(fig)
        GC.gc()
    end
end

function plot_corner(samples_file::String, output_dir::String;
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset = _read_samples(samples_file)
    n_hex = size(dataset, 1)
    post_dir = joinpath(output_dir, "posterior")
    mkpath(post_dir)

    @showprogress "Plotting corners: " for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        samp = dataset[i, :, 1:3]
        tbl = (A_prime=samp[:, 1], B_prime=samp[:, 2], alpha=samp[:, 3])
        fig = Figure()
        pairplot(fig[1, 1],
            PairPlots.Series(tbl) => (PairPlots.Scatter(),
                PairPlots.Contourf(),
                PairPlots.MarginDensity()))
        save(joinpath(post_dir, "$i.png"), fig)
        empty!(fig)
        GC.gc()
    end
end

function spec_vs_frac(samples_file::String, output_dir::String)
    dataset = _read_samples(samples_file)
    n_hex = size(dataset, 1)

    spec = zeros(n_hex, 3)
    tfrac = zeros(n_hex, 3)
    for i in 1:n_hex
        samp = dataset[i, :, :]
        q = quantile(samp[:, 3], [0.16, 0.50, 0.84])
        spec[i, :] = q
        p16, p50, p84 = thermal_frac_percentiles(samp)
        tfrac[i, :] = [p16, p50, p84]
    end

    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel="Thermal Fraction", ylabel="Recovered Spectral Index")
    errorbars!(ax, tfrac[:, 2], spec[:, 2],
        tfrac[:, 2] .- tfrac[:, 1],
        tfrac[:, 3] .- tfrac[:, 2];
        direction=:x)
    errorbars!(ax, tfrac[:, 2], spec[:, 2],
        spec[:, 2] .- spec[:, 1],
        spec[:, 3] .- spec[:, 2])
    scatter!(ax, tfrac[:, 2], spec[:, 2])
    save(joinpath(output_dir, "spec_vs_frac.png"), fig)
end

function ext_vs_ha(samples_file::String, output_dir::String)
    dataset = _read_samples(samples_file)
    n_hex = size(dataset, 1)

    ext = zeros(n_hex, 3)
    ha = zeros(n_hex, 3)
    for i in 1:n_hex
        samp = dataset[i, :, :]
        ext_samp = filter(isfinite, extinction_calc(samp))
        ext[i, :] = isempty(ext_samp) ? [NaN, NaN, NaN] :
                    quantile(ext_samp, [0.16, 0.50, 0.84])
        ha_fin = filter(isfinite, samp[:, 4])
        ha[i, :] = isempty(ha_fin) ? [NaN, NaN, NaN] :
                   quantile(ha_fin, [0.16, 0.50, 0.84])
    end

    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel="Hα Thermal Emission (Jy)", ylabel="Recovered E(B-V)")
    errorbars!(ax, ha[:, 2], ext[:, 2],
        ha[:, 2] .- ha[:, 1], ha[:, 3] .- ha[:, 2]; direction=:x)
    errorbars!(ax, ha[:, 2], ext[:, 2],
        ext[:, 2] .- ext[:, 1], ext[:, 3] .- ext[:, 2])
    scatter!(ax, ha[:, 2], ext[:, 2])
    save(joinpath(output_dir, "ha_vs_ext.png"), fig)
end

function plot_hex_maps(samples_file::String, dc::DataCube, hg::HexGrid,
    output_dir::String;
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset = _read_samples(samples_file)
    n_hex = size(dataset, 1)
    maps_dir = joinpath(output_dir, "maps")
    mkpath(maps_dir)

    ny = dc.header["NAXIS2"]
    nx = dc.header["NAXIS1"]

    T_map = fill(NaN, ny, nx)
    T_err = fill(NaN, ny, nx)
    nT_map = fill(NaN, ny, nx)
    nT_err = fill(NaN, ny, nx)
    spec_map = fill(NaN, ny, nx)
    spec_err = fill(NaN, ny, nx)
    tfrac_map = fill(NaN, ny, nx)
    tfrac_err = fill(NaN, ny, nx)

    for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        samp = dataset[i, :, :]

        qT = quantile(samp[:, 1], [0.16, 0.50, 0.84])
        qnT = quantile(samp[:, 2], [0.16, 0.50, 0.84])
        qspec = quantile(samp[:, 3], [0.16, 0.50, 0.84])
        tf16, tf50, tf84 = thermal_frac_percentiles(samp)

        for px in hg.pixel_members[i]
            T_map[px] = qT[2]
            T_err[px] = (qT[3] - qT[1]) / 2
            nT_map[px] = qnT[2]
            nT_err[px] = (qnT[3] - qnT[1]) / 2
            spec_map[px] = qspec[2]
            spec_err[px] = (qspec[3] - qspec[1]) / 2
            tfrac_map[px] = tf50
            tfrac_err[px] = (tf84 - tf16) / 2
        end
    end

    hdr = dc.header
    _write_fits_map(joinpath(maps_dir, "T.fits"), T_map, hdr)
    _write_fits_map(joinpath(maps_dir, "T_err.fits"), T_err, hdr)
    _write_fits_map(joinpath(maps_dir, "nT.fits"), nT_map, hdr)
    _write_fits_map(joinpath(maps_dir, "nT_err.fits"), nT_err, hdr)
    _write_fits_map(joinpath(maps_dir, "spec.fits"), spec_map, hdr)
    _write_fits_map(joinpath(maps_dir, "spec_err.fits"), spec_err, hdr)
    _write_fits_map(joinpath(maps_dir, "tfrac.fits"), tfrac_map, hdr)
    _write_fits_map(joinpath(maps_dir, "tfrac_err.fits"), tfrac_err, hdr)
end

function mag_equip_map(samples_file::String, fd::FluxData, dc::DataCube,
    hg::HexGrid, output_dir::String;
    freq_ghz::Float64=1.0,
    k::Float64=100.0,
    scale_height::Float64=0.1,
    disk_inclination::Float64=33.0,
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset = _read_samples(samples_file)
    n_hex = size(dataset, 1)
    maps_dir = joinpath(output_dir, "maps")
    mkpath(maps_dir)

    ny = dc.header["NAXIS2"]
    nx = dc.header["NAXIS1"]
    mag_map = fill(NaN, ny, nx)
    mag_err = fill(NaN, ny, nx)

    av_spec = median(median(dataset[:, :, 3]; dims=2))

    for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        nt_samples = dataset[i, :, 2]
        B_samples = bfield_revised(nt_samples, fd.area[i], freq_ghz,
            av_spec, k, scale_height, disk_inclination) .* 1e6
        qB = quantile(B_samples, [0.16, 0.50, 0.84])

        for px in hg.pixel_members[i]
            mag_map[px] = qB[2]
            mag_err[px] = (qB[3] - qB[1]) / 2
        end
    end

    _write_fits_map(joinpath(maps_dir, "mag_eq.fits"), mag_map, dc.header)
    _write_fits_map(joinpath(maps_dir, "mag_eq_err.fits"), mag_err, dc.header)
end

function extinction_map(samples_file::String, dc::DataCube, hg::HexGrid,
    output_dir::String;
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset = _read_samples(samples_file)
    n_hex = size(dataset, 1)
    maps_dir = joinpath(output_dir, "maps")
    mkpath(maps_dir)

    ny = dc.header["NAXIS2"]
    nx = dc.header["NAXIS1"]
    ext_map = fill(NaN, ny, nx)
    ext_err = fill(NaN, ny, nx)

    for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        samp = dataset[i, :, :]
        ext_fin = filter(isfinite, extinction_calc(samp))
        if isempty(ext_fin)
            continue
        end
        qE = quantile(ext_fin, [0.16, 0.50, 0.84])
        for px in hg.pixel_members[i]
            ext_map[px] = qE[2]
            ext_err[px] = (qE[3] - qE[1]) / 2
        end
    end

    _write_fits_map(joinpath(maps_dir, "ext.fits"), ext_map, dc.header)
    _write_fits_map(joinpath(maps_dir, "ext_err.fits"), ext_err, dc.header)
end

function plot_labeled_map(samples_file::String, dc::DataCube, hg::HexGrid,
    output_dir::String;
    parameter::Symbol=:tfrac,
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset = _read_samples(samples_file)
    n_hex = size(dataset, 1)
    ny = dc.header["NAXIS2"]
    nx = dc.header["NAXIS1"]

    pmap = fill(NaN, ny, nx)
    for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        samp = dataset[i, :, :]
        val = if parameter == :tfrac
            thermal_frac_median(samp)
        elseif parameter == :spec
            median(samp[:, 3])
        elseif parameter == :therm
            median(samp[:, 1])
        elseif parameter == :nontherm
            median(samp[:, 2])
        else
            error("Unknown parameter $parameter. Choose: :tfrac, :spec, :therm, :nontherm")
        end
        for px in hg.pixel_members[i]
            pmap[px] = val
        end
    end

    fig = Figure(size=(700, 700))
    ax = Axis(fig[1, 1]; aspect=DataAspect(),
        title="Bin labels — parameter: $parameter",
        xlabel="pixel col", ylabel="pixel row")

    heatmap!(ax, 1:nx, 1:ny, pmap'; nan_color=(:grey, 0.3), colormap=:viridis)

    for i in 1:n_hex
        cx, cy = hg.centers[i]
        col = isnothing(quality_mask) || quality_mask[i] ? :white : (:red, 0.7)
        text!(ax, cx, cy; text=string(i), fontsize=7,
            align=(:center, :center), color=col)
    end

    save(joinpath(output_dir, "labeled_map.png"), fig)
    empty!(fig)
    println("Labeled map saved to ", joinpath(output_dir, "labeled_map.png"))
end

function query_bin(bin_idx::Int, samples_file::String, fd::FluxData,
    dc::DataCube, output_dir::String)
    dataset = h5open(samples_file, "r") do hdf
        read(hdf, "samples")
    end
    n_hex = size(dataset, 1)
    1 <= bin_idx <= n_hex || error("bin_idx $bin_idx out of range 1:$n_hex")

    samp = dataset[bin_idx, :, :]

    function pct(v)
        vf = filter(isfinite, v)
        isempty(vf) && return (NaN, NaN, NaN)
        q = quantile(vf, [0.16, 0.50, 0.84])
        return (q[1], q[2], q[3])
    end

    T = pct(samp[:, 1])
    nT = pct(samp[:, 2])
    alpha = pct(samp[:, 3])
    Ha = pct(samp[:, 4])
    tf16, tf50, tf84 = thermal_frac_percentiles(samp)
    tfrac = (tf16, tf50, tf84)
    ext_samp = filter(isfinite, extinction_calc(samp))
    EBV = isempty(ext_samp) ? (NaN, NaN, NaN) :
          Tuple(quantile(ext_samp, [0.16, 0.50, 0.84]))

    sed_path = joinpath(output_dir, "SED", "$bin_idx.png")
    corner_path = joinpath(output_dir, "posterior", "$bin_idx.png")

    println("╔══════════════════════════════════════════════════════╗")
    println("║  Bin $bin_idx$(repeat(" ", 50 - ndigits(bin_idx)))║")
    println("╠══════════════════════════════════════════════════════╣")
    @printf("║  %-18s %12.4g  [%10.4g, %10.4g]  ║\n",
        "Thermal (Jy)", T[2], T[1], T[3])
    @printf("║  %-18s %12.4g  [%10.4g, %10.4g]  ║\n",
        "Non-thermal (Jy)", nT[2], nT[1], nT[3])
    @printf("║  %-18s %12.4f  [%10.4f, %10.4f]  ║\n",
        "Spectral index", alpha[2], alpha[1], alpha[3])
    @printf("║  %-18s %12.4f  [%10.4f, %10.4f]  ║\n",
        "Thermal fraction", tfrac[2], tfrac[1], tfrac[3])
    @printf("║  %-18s %12.4g  [%10.4g, %10.4g]  ║\n",
        "Hα (Jy)", Ha[2], Ha[1], Ha[3])
    @printf("║  %-18s %12.4f  [%10.4f, %10.4f]  ║\n",
        "E(B-V)", EBV[2], EBV[1], EBV[3])
    println("╠══════════════════════════════════════════════════════╣")
    println("║  SED:    ", rpad(isfile(sed_path) ? sed_path : "(not generated)", 44), "║")
    println("║  Corner: ", rpad(isfile(corner_path) ? corner_path : "(not generated)", 44), "║")
    println("╚══════════════════════════════════════════════════════╝")

    return (; T, nT, alpha, tfrac, Ha, EBV, sed_path, corner_path)
end

function query_bin(pixel_col::Int, pixel_row::Int, samples_file::String,
    fd::FluxData, dc::DataCube, hg::HexGrid, output_dir::String)
    ny, nx = size(hg.inverse_map)
    (1 <= pixel_row <= ny && 1 <= pixel_col <= nx) ||
        error("Pixel ($pixel_col, $pixel_row) out of image bounds ($nx × $ny)")
    bin_idx = hg.inverse_map[pixel_row, pixel_col]
    bin_idx == 0 && error("Pixel ($pixel_col, $pixel_row) is not assigned to any hex bin")
    println("Pixel (col=$pixel_col, row=$pixel_row) → bin $bin_idx\n")
    return query_bin(bin_idx, samples_file, fd, dc, output_dir)
end

function all_plots(fd::FluxData, dc::DataCube, hg::HexGrid,
    samples_file::String, output_dir::String;
    mask_poor_bins::Bool=true,
    alpha_width_threshold::Float64=0.5,
    tfrac_width_threshold::Float64=0.4)
    qmask = mask_poor_bins ?
            compute_quality_mask(samples_file;
        alpha_width_threshold, tfrac_width_threshold) :
            nothing
    spec_vs_frac(samples_file, output_dir)
    ext_vs_ha(samples_file, output_dir)
    plot_labeled_map(samples_file, dc, hg, output_dir; quality_mask=qmask)
    plot_sed(fd, dc.freq, samples_file, output_dir; quality_mask=qmask)
    plot_corner(samples_file, output_dir; quality_mask=qmask)
    plot_hex_maps(samples_file, dc, hg, output_dir; quality_mask=qmask)
    mag_equip_map(samples_file, fd, dc, hg, output_dir; quality_mask=qmask)
    extinction_map(samples_file, dc, hg, output_dir; quality_mask=qmask)
end

function _write_fits_map(path::String, map::Matrix{Float64}, header)
    data_out = permutedims(map, (2, 1))
    isfile(path) && rm(path)
    hdus = FITSFiles.HDU[FITSFiles.HDU(data_out, header)]
    Base.write(path, hdus)
end