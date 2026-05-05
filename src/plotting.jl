using CairoMakie
using PairPlots
using LaTeXStrings
using HDF5
using Statistics
using Printf
using FITSFiles

function read_samples(samples_file::String)
    h5open(samples_file, "r") do hdf
        read(hdf, "samples")
    end
end

thermal(therm_norm, freq) = therm_norm .* (freq ./ 1e9) .^ (-0.1)
non_thermal(nt_norm, spec, freq) = nt_norm .* (freq ./ 1e9) .^ spec
function non_thermal_curved(nt_norm, spec, beta, freq)
    r = log.(freq ./ 1e9)
    nt_norm .* exp.((spec .+ beta .* r) .* r)
end
function non_thermal_abs(nt_norm, spec, tau0, freq)
    ratio = freq ./ 1e9
    nt_norm .* ratio .^ spec .* exp.(-tau0 .* ratio .^ (-2.1))
end
function non_thermal_broken(nt_norm, spec, nu_b, freq)
    map(freq) do f
        f <= nu_b ?
            nt_norm * (f / 1e9)^spec :
            nt_norm * (nu_b / 1e9)^spec * (f / nu_b)^(spec - 0.5)
    end
end

function plot_sed(fd::FluxData, freq::AbstractVector, samples_file::String, output_dir::String;
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset    = read_samples(samples_file)
    model_str  = read_model_attr(samples_file)
    n_hex      = size(dataset, 1)
    is_curved  = model_str == "CurvedPowerLaw"
    is_ffa     = model_str == "ffa"
    is_bpl     = model_str == "BrokenPowerLaw"
    sed_dir    = joinpath(output_dir, "SED")
    mkpath(sed_dir)

    x_lo = minimum(freq) * 0.3
    x_hi = maximum(freq) * 3.0
    freq_range = 10 .^ range(log10(x_lo), log10(x_hi); length=200)

    all_lo = [fd.dmatrix[i, j] - fd.dnoisematrix[i, j]
              for i in 1:n_hex, j in eachindex(freq)
              if isfinite(fd.dmatrix[i, j]) && fd.dmatrix[i, j] > 0]
    all_hi = [fd.dmatrix[i, j] + fd.dnoisematrix[i, j]
              for i in 1:n_hex, j in eachindex(freq)
              if isfinite(fd.dmatrix[i, j])]
    ha_vals = filter(>(0), fd.dHa)
    y_lo = isempty(all_lo) ? 1e-10 : max(minimum(all_lo), 1e-10) * 0.3
    y_hi = isempty(all_hi) ? 1e-5  : max(maximum(all_hi), maximum(ha_vals; init=0.0)) * 3.0

    @showprogress "Plotting SEDs: " for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        samp = dataset[i, :, :]

        fig = Figure()
        ax  = Axis(fig[1, 1];
            xlabel="Frequency (Hz)", ylabel="Flux Density (Jy)",
            xscale=log10, yscale=log10,
            limits=(x_lo, x_hi, y_lo, y_hi))

        errorbars!(ax, freq, fd.dmatrix[i, :], fd.dnoisematrix[i, :]; color=:black)
        scatter!(ax, freq, fd.dmatrix[i, :]; color=:black)

        errorbars!(ax, [1e9], [fd.dHa[i]], [0.5 * fd.dHa[i]];
            color=:red, whiskerwidth=8)
        scatter!(ax, [1e9], [fd.dHa[i]]; color=:red)

        best_T    = median(samp[:, 1])
        best_nT   = median(samp[:, 2])
        best_spec = median(samp[:, 3])
        if is_curved
            best_p4 = median(samp[:, 4])
            lines!(ax, freq_range,
                sed.(best_T, best_nT, best_spec, best_p4, freq_range, Ref(CurvedPowerLaw()));
                color=:black)
        elseif is_ffa
            best_tau0 = median(samp[:, 4])
            lines!(ax, freq_range,
                sed.(best_T, best_nT, best_spec, best_tau0, freq_range, Ref(ffa()));
                color=:black)
        elseif is_bpl
            best_nu_b = median(samp[:, 4])
            lines!(ax, freq_range,
                sed.(best_T, best_nT, best_spec, best_nu_b, freq_range, Ref(BrokenPowerLaw()));
                color=:black)
        else
            lines!(ax, freq_range, sed.(best_T, best_nT, best_spec, freq_range); color=:black)
        end

        idx = rand(1:size(samp, 1), 250)
        for j in idx
            samp[j, 1] > 0 && lines!(ax, freq_range, thermal(samp[j, 1], freq_range);
                color=(:red, 0.0075), linewidth=10)
            if is_curved
                samp[j, 2] > 0 && lines!(ax, freq_range,
                    non_thermal_curved(samp[j, 2], samp[j, 3], samp[j, 4], freq_range);
                    color=(:blue, 0.0075), linewidth=10)
            elseif is_ffa
                samp[j, 2] > 0 && lines!(ax, freq_range,
                    non_thermal_abs(samp[j, 2], samp[j, 3], samp[j, 4], freq_range);
                    color=(:blue, 0.0075), linewidth=10)
            elseif is_bpl
                samp[j, 2] > 0 && lines!(ax, freq_range,
                    non_thermal_broken(samp[j, 2], samp[j, 3], samp[j, 4], freq_range);
                    color=(:blue, 0.0075), linewidth=10)
            else
                samp[j, 2] > 0 && lines!(ax, freq_range,
                    non_thermal(samp[j, 2], samp[j, 3], freq_range);
                    color=(:blue, 0.0075), linewidth=10)
            end
        end

        save(joinpath(sed_dir, "$i.png"), fig)
        empty!(fig)
        GC.gc()
    end
end

function plot_corner(samples_file::String, output_dir::String;
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset   = read_samples(samples_file)
    model_str = read_model_attr(samples_file)
    n_hex     = size(dataset, 1)
    is_curved = model_str == "CurvedPowerLaw"
    is_ffa    = model_str == "ffa"
    is_bpl    = model_str == "BrokenPowerLaw"
    post_dir  = joinpath(output_dir, "posterior")
    mkpath(post_dir)

    pl_labels   = Dict(:A_prime => L"A'", :B_prime => L"B'", :alpha => L"\alpha")
    cpl_labels  = merge(pl_labels, Dict(:beta  => L"\beta"))
    ffa_labels  = merge(pl_labels, Dict(:tau0  => L"\tau_0"))
    bpl_labels  = merge(pl_labels, Dict(:nu_b_GHz => L"\nu_b\,(\mathrm{GHz})"))

    @showprogress "Plotting corners: " for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        tbl, labels = if is_curved
            s = dataset[i, :, 1:4]
            (A_prime=s[:, 1], B_prime=s[:, 2], alpha=s[:, 3], beta=s[:, 4]), cpl_labels
        elseif is_ffa
            s = dataset[i, :, 1:4]
            (A_prime=s[:, 1], B_prime=s[:, 2], alpha=s[:, 3], tau0=s[:, 4]), ffa_labels
        elseif is_bpl
            s = dataset[i, :, 1:4]
            (A_prime=s[:, 1], B_prime=s[:, 2], alpha=s[:, 3], nu_b_GHz=s[:, 4] ./ 1e9), bpl_labels
        else
            s = dataset[i, :, 1:3]
            (A_prime=s[:, 1], B_prime=s[:, 2], alpha=s[:, 3]), pl_labels
        end
        fig = Figure()
        pairplot(fig[1, 1],
            PairPlots.Series(tbl) => (PairPlots.Scatter(),
                PairPlots.Contourf(),
                PairPlots.MarginDensity());
            labels)
        save(joinpath(post_dir, "$i.png"), fig)
        empty!(fig)
        GC.gc()
    end
end

function spec_vs_frac(samples_file::String, output_dir::String)
    dataset = read_samples(samples_file)
    n_hex   = size(dataset, 1)

    spec  = zeros(n_hex, 3)
    tfrac = zeros(n_hex, 3)
    for i in 1:n_hex
        samp    = dataset[i, :, :]
        q       = quantile(samp[:, 3], [0.16, 0.50, 0.84])
        spec[i, :] = q
        p16, p50, p84 = thermal_frac_percentiles(samp)
        tfrac[i, :] = [p16, p50, p84]
    end

    fig = Figure()
    ax  = Axis(fig[1, 1]; xlabel="Thermal Fraction", ylabel="Recovered Spectral Index")
    errorbars!(ax, tfrac[:, 2], spec[:, 2],
        tfrac[:, 2] .- tfrac[:, 1], tfrac[:, 3] .- tfrac[:, 2]; direction=:x)
    errorbars!(ax, tfrac[:, 2], spec[:, 2],
        spec[:, 2] .- spec[:, 1], spec[:, 3] .- spec[:, 2])
    scatter!(ax, tfrac[:, 2], spec[:, 2])
    save(joinpath(output_dir, "spec_vs_frac.png"), fig)
end

function ext_vs_ha(samples_file::String, output_dir::String)
    dataset = read_samples(samples_file)
    n_hex   = size(dataset, 1)

    ext = zeros(n_hex, 3)
    ha  = zeros(n_hex, 3)
    for i in 1:n_hex
        samp    = dataset[i, :, :]
        ext_fin = filter(isfinite, extinction_calc(samp))
        ext[i, :] = isempty(ext_fin) ? [NaN, NaN, NaN] :
                    quantile(ext_fin, [0.16, 0.50, 0.84])
        ha_fin    = filter(isfinite, samp[:, end])
        ha[i, :]  = isempty(ha_fin) ? [NaN, NaN, NaN] :
                    quantile(ha_fin, [0.16, 0.50, 0.84])
    end

    fig = Figure()
    ax  = Axis(fig[1, 1]; xlabel="Hα Thermal Emission (Jy)", ylabel="Recovered E(B-V)")
    errorbars!(ax, ha[:, 2], ext[:, 2],
        ha[:, 2] .- ha[:, 1], ha[:, 3] .- ha[:, 2]; direction=:x)
    errorbars!(ax, ha[:, 2], ext[:, 2],
        ext[:, 2] .- ext[:, 1], ext[:, 3] .- ext[:, 2])
    scatter!(ax, ha[:, 2], ext[:, 2])
    save(joinpath(output_dir, "ha_vs_ext.png"), fig)
end

function plot_hex_maps(samples_file::String, dc::DataCube, hg::HexGrid,
    output_dir::String;
    T_e::Float64=8000.0,
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset   = read_samples(samples_file)
    model_str = read_model_attr(samples_file)
    n_hex     = size(dataset, 1)
    is_curved = model_str == "CurvedPowerLaw"
    is_ffa    = model_str == "ffa"
    is_bpl    = model_str == "BrokenPowerLaw"
    maps_dir  = joinpath(output_dir, "maps")
    mkpath(maps_dir)

    ny = dc.header["NAXIS2"]
    nx = dc.header["NAXIS1"]

    T_map      = fill(NaN, ny, nx)
    T_err      = fill(NaN, ny, nx)
    nT_map     = fill(NaN, ny, nx)
    nT_err     = fill(NaN, ny, nx)
    spec_map   = fill(NaN, ny, nx)
    spec_err   = fill(NaN, ny, nx)
    tfrac_map  = fill(NaN, ny, nx)
    tfrac_err  = fill(NaN, ny, nx)
    beta_map   = is_curved ? fill(NaN, ny, nx) : nothing
    beta_err   = is_curved ? fill(NaN, ny, nx) : nothing
    tau_map    = is_ffa    ? fill(NaN, ny, nx) : nothing
    tau_err    = is_ffa    ? fill(NaN, ny, nx) : nothing
    em_map     = is_ffa    ? fill(NaN, ny, nx) : nothing
    em_err     = is_ffa    ? fill(NaN, ny, nx) : nothing
    nu_b_map   = is_bpl    ? fill(NaN, ny, nx) : nothing
    nu_b_err   = is_bpl    ? fill(NaN, ny, nx) : nothing

    for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        samp  = dataset[i, :, :]

        qT    = quantile(samp[:, 1], [0.16, 0.50, 0.84])
        qnT   = quantile(samp[:, 2], [0.16, 0.50, 0.84])
        qspec = quantile(samp[:, 3], [0.16, 0.50, 0.84])
        tf16, tf50, tf84 = thermal_frac_percentiles(samp)

        if is_curved
            qbeta = quantile(samp[:, 4], [0.16, 0.50, 0.84])
        elseif is_ffa
            tau16, tau50, tau84 = tau0_percentiles(samp)
            em_samples = emission_measure_calc(filter(isfinite, samp[:, 4]); T_e)
            qem = isempty(em_samples) ? [NaN, NaN, NaN] :
                  quantile(em_samples, [0.16, 0.50, 0.84])
        elseif is_bpl
            qnu_b = quantile(filter(isfinite, samp[:, 4]), [0.16, 0.50, 0.84])
        end

        for px in hg.pixel_members[i]
            T_map[px]     = qT[2];    T_err[px]    = (qT[3]    - qT[1])    / 2
            nT_map[px]    = qnT[2];   nT_err[px]   = (qnT[3]   - qnT[1])   / 2
            spec_map[px]  = qspec[2]; spec_err[px] = (qspec[3]  - qspec[1]) / 2
            tfrac_map[px] = tf50;     tfrac_err[px] = (tf84 - tf16) / 2
            if is_curved
                beta_map[px] = qbeta[2]
                beta_err[px] = (qbeta[3] - qbeta[1]) / 2
            elseif is_ffa
                tau_map[px] = tau50
                tau_err[px] = (tau84 - tau16) / 2
                em_map[px]  = qem[2]
                em_err[px]  = (qem[3] - qem[1]) / 2
            elseif is_bpl
                nu_b_map[px] = qnu_b[2]
                nu_b_err[px] = (qnu_b[3] - qnu_b[1]) / 2
            end
        end
    end

    hdr = dc.header
    write_fits_map(joinpath(maps_dir, "T.fits"),         T_map,     hdr)
    write_fits_map(joinpath(maps_dir, "T_err.fits"),     T_err,     hdr)
    write_fits_map(joinpath(maps_dir, "nT.fits"),        nT_map,    hdr)
    write_fits_map(joinpath(maps_dir, "nT_err.fits"),    nT_err,    hdr)
    write_fits_map(joinpath(maps_dir, "spec.fits"),      spec_map,  hdr)
    write_fits_map(joinpath(maps_dir, "spec_err.fits"),  spec_err,  hdr)
    write_fits_map(joinpath(maps_dir, "tfrac.fits"),     tfrac_map, hdr)
    write_fits_map(joinpath(maps_dir, "tfrac_err.fits"), tfrac_err, hdr)
    if is_curved
        write_fits_map(joinpath(maps_dir, "beta.fits"),     beta_map, hdr)
        write_fits_map(joinpath(maps_dir, "beta_err.fits"), beta_err, hdr)
    elseif is_ffa
        write_fits_map(joinpath(maps_dir, "tau.fits"),     tau_map, hdr)
        write_fits_map(joinpath(maps_dir, "tau_err.fits"), tau_err, hdr)
        write_fits_map(joinpath(maps_dir, "EM.fits"),      em_map,  hdr)
        write_fits_map(joinpath(maps_dir, "EM_err.fits"),  em_err,  hdr)
    elseif is_bpl
        write_fits_map(joinpath(maps_dir, "nu_b.fits"),     nu_b_map, hdr)
        write_fits_map(joinpath(maps_dir, "nu_b_err.fits"), nu_b_err, hdr)
    end
end

function mag_equip_map(samples_file::String, fd::FluxData, dc::DataCube,
    hg::HexGrid, output_dir::String;
    freq_ghz::Float64=1.0,
    k::Float64=100.0,
    scale_height::Float64=0.1,
    disk_inclination::Float64=33.0,
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset  = read_samples(samples_file)
    n_hex    = size(dataset, 1)
    maps_dir = joinpath(output_dir, "maps")
    mkpath(maps_dir)

    ny      = dc.header["NAXIS2"]
    nx      = dc.header["NAXIS1"]
    mag_map = fill(NaN, ny, nx)
    mag_err = fill(NaN, ny, nx)

    av_spec = median(median(dataset[:, :, 3]; dims=2))

    for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        nt_samples = dataset[i, :, 2]
        B_samples  = bfield_revised(nt_samples, fd.area[i], freq_ghz,
            av_spec, k, scale_height, disk_inclination) .* 1e6
        qB = quantile(B_samples, [0.16, 0.50, 0.84])

        for px in hg.pixel_members[i]
            mag_map[px] = qB[2]
            mag_err[px] = (qB[3] - qB[1]) / 2
        end
    end

    write_fits_map(joinpath(maps_dir, "mag_eq.fits"),     mag_map, dc.header)
    write_fits_map(joinpath(maps_dir, "mag_eq_err.fits"), mag_err, dc.header)
end

function extinction_map(samples_file::String, dc::DataCube, hg::HexGrid,
    output_dir::String;
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset  = read_samples(samples_file)
    n_hex    = size(dataset, 1)
    maps_dir = joinpath(output_dir, "maps")
    mkpath(maps_dir)

    ny      = dc.header["NAXIS2"]
    nx      = dc.header["NAXIS1"]
    ext_map = fill(NaN, ny, nx)
    ext_err = fill(NaN, ny, nx)

    for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        samp    = dataset[i, :, :]
        ext_fin = filter(isfinite, extinction_calc(samp))
        isempty(ext_fin) && continue
        qE = quantile(ext_fin, [0.16, 0.50, 0.84])
        for px in hg.pixel_members[i]
            ext_map[px] = qE[2]
            ext_err[px] = (qE[3] - qE[1]) / 2
        end
    end

    write_fits_map(joinpath(maps_dir, "ext.fits"),     ext_map, dc.header)
    write_fits_map(joinpath(maps_dir, "ext_err.fits"), ext_err, dc.header)
end

function plot_labeled_map(samples_file::String, dc::DataCube, hg::HexGrid,
    output_dir::String;
    parameter::Symbol=:tfrac,
    quality_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    dataset   = read_samples(samples_file)
    model_str = read_model_attr(samples_file)
    n_hex     = size(dataset, 1)
    is_ffa    = model_str == "ffa"
    is_bpl    = model_str == "BrokenPowerLaw"
    ny        = dc.header["NAXIS2"]
    nx        = dc.header["NAXIS1"]

    pmap = fill(NaN, ny, nx)
    for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        samp = dataset[i, :, :]
        val  = if parameter == :tfrac
            thermal_frac_median(samp)
        elseif parameter == :spec
            median(samp[:, 3])
        elseif parameter == :therm
            median(samp[:, 1])
        elseif parameter == :nontherm
            median(samp[:, 2])
        elseif parameter == :beta
            model_str == "CurvedPowerLaw" || error("Beta map requires CurvedPowerLaw model")
            median(samp[:, 4])
        elseif parameter == :tau
            is_ffa || error("Tau map requires ffa model")
            _, tau50, _ = tau0_percentiles(samp)
            tau50
        elseif parameter == :nu_b
            is_bpl || error("nu_b map requires BrokenPowerLaw model")
            median(filter(isfinite, samp[:, 4]))
        else
            error("Unknown parameter $parameter. Choose: :tfrac, :spec, :therm, :nontherm, :beta, :tau, :nu_b")
        end
        for px in hg.pixel_members[i]
            pmap[px] = val
        end
    end

    fig = Figure(size=(700, 700))
    ax  = Axis(fig[1, 1]; aspect=DataAspect(),
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
    model_str = read_model_attr(samples_file)
    n_hex     = size(dataset, 1)
    1 <= bin_idx <= n_hex || error("bin_idx $bin_idx out of range 1:$n_hex")

    samp      = dataset[bin_idx, :, :]
    is_curved = model_str == "CurvedPowerLaw"
    is_ffa    = model_str == "ffa"
    is_bpl    = model_str == "BrokenPowerLaw"

    function pct(v)
        vf = filter(isfinite, v)
        isempty(vf) && return (NaN, NaN, NaN)
        q = quantile(vf, [0.16, 0.50, 0.84])
        return (q[1], q[2], q[3])
    end

    T     = pct(samp[:, 1])
    nT    = pct(samp[:, 2])
    alpha = pct(samp[:, 3])
    beta  = is_curved ? pct(samp[:, 4]) : nothing
    tau0  = is_ffa    ? pct(samp[:, 4]) : nothing
    nu_b  = is_bpl    ? pct(samp[:, 4]) : nothing
    Ha    = pct(samp[:, end])
    tf16, tf50, tf84 = thermal_frac_percentiles(samp)
    tfrac     = (tf16, tf50, tf84)
    ext_samp  = filter(isfinite, extinction_calc(samp))
    EBV       = isempty(ext_samp) ? (NaN, NaN, NaN) :
                Tuple(quantile(ext_samp, [0.16, 0.50, 0.84]))

    sed_path    = joinpath(output_dir, "SED",       "$bin_idx.png")
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
    if is_curved
        @printf("║  %-18s %12.4f  [%10.4f, %10.4f]  ║\n",
            "Curvature (β)", beta[2], beta[1], beta[3])
    elseif is_ffa
        @printf("║  %-18s %12.4f  [%10.4f, %10.4f]  ║\n",
            "τ₀ (FFA)", tau0[2], tau0[1], tau0[3])
    elseif is_bpl
        @printf("║  %-18s %12.4f  [%10.4f, %10.4f]  ║\n",
            "ν_b (GHz)", nu_b[2]/1e9, nu_b[1]/1e9, nu_b[3]/1e9)
    end
    @printf("║  %-18s %12.4f  [%10.4f, %10.4f]  ║\n",
        "Thermal fraction", tfrac[2], tfrac[1], tfrac[3])
    @printf("║  %-18s %12.4g  [%10.4g, %10.4g]  ║\n",
        "Hα (Jy)", Ha[2], Ha[1], Ha[3])
    @printf("║  %-18s %12.4f  [%10.4f, %10.4f]  ║\n",
        "E(B-V)", EBV[2], EBV[1], EBV[3])
    println("╠══════════════════════════════════════════════════════╣")
    println("║  SED:    ", rpad(isfile(sed_path)    ? sed_path    : "(not generated)", 44), "║")
    println("║  Corner: ", rpad(isfile(corner_path) ? corner_path : "(not generated)", 44), "║")
    println("╚══════════════════════════════════════════════════════╝")

    return (; T, nT, alpha, beta, tau0, nu_b, tfrac, Ha, EBV, sed_path, corner_path)
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
    tfrac_width_threshold::Float64=0.4,
    beta_width_threshold::Float64=0.8,
    tau_width_threshold::Float64=1.0,
    nu_b_log_width_threshold::Float64=2.0,
    T_e::Float64=8000.0)
    qmask = mask_poor_bins ?
            compute_quality_mask(samples_file;
                alpha_width_threshold, tfrac_width_threshold,
                beta_width_threshold, tau_width_threshold,
                nu_b_log_width_threshold) :
            nothing
    spec_vs_frac(samples_file, output_dir)
    ext_vs_ha(samples_file, output_dir)
    plot_labeled_map(samples_file, dc, hg, output_dir; quality_mask=qmask)
    plot_sed(fd, dc.freq, samples_file, output_dir; quality_mask=qmask)
    plot_corner(samples_file, output_dir; quality_mask=qmask)
    plot_hex_maps(samples_file, dc, hg, output_dir; T_e, quality_mask=qmask)
    mag_equip_map(samples_file, fd, dc, hg, output_dir; quality_mask=qmask)
    extinction_map(samples_file, dc, hg, output_dir; quality_mask=qmask)
end

function plot_rj_model_maps(rjsamples_file::String, dc::DataCube, hg::HexGrid,
    output_dir::String;
    quality_mask::Union{Nothing,AbstractVector{Bool}} = nothing)

    probs   = model_posterior_probs(rjsamples_file)
    entropy = model_posterior_entropy(rjsamples_file)
    map_idx = map_model_index(rjsamples_file)
    n_hex   = size(probs, 1)
    ny, nx  = dc.header["NAXIS2"], dc.header["NAXIS1"]
    maps_dir = joinpath(output_dir, "maps")
    mkpath(maps_dir)

    prob_maps    = [fill(NaN, ny, nx) for _ in 1:N_MODELS]
    entropy_map  = fill(NaN, ny, nx)
    map_idx_map  = fill(NaN, ny, nx)

    for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        for px in hg.pixel_members[i]
            for m in 1:N_MODELS
                prob_maps[m][px] = probs[i, m]
            end
            entropy_map[px] = entropy[i]
            map_idx_map[px] = Float64(map_idx[i])
        end
    end

    for m in 1:N_MODELS
        write_fits_map(joinpath(maps_dir, "model_prob_$(MODEL_NAMES[m]).fits"),
                        prob_maps[m], dc.header)
    end
    write_fits_map(joinpath(maps_dir, "model_entropy.fits"), entropy_map, dc.header)
    write_fits_map(joinpath(maps_dir, "model_map.fits"),     map_idx_map, dc.header)

    fig = Figure(size=(900, 700))
    cmaps = (:Blues, :Oranges, :Greens, :Purples)
    positions = ((1,1), (1,2), (2,1), (2,2))
    for (m, (row, col)) in zip(1:N_MODELS, positions)
        ax = Axis(fig[row, col]; title=MODEL_NAMES[m],
                  aspect=DataAspect(), xlabel="col", ylabel="row")
        heatmap!(ax, 1:nx, 1:ny, prob_maps[m]';
                 nan_color=(:grey, 0.3), colormap=cmaps[m],
                 colorrange=(0.0, 1.0))
        Colorbar(fig[row, col+2]; colormap=cmaps[m], limits=(0, 1),
                 label="P(model)", width=12)
    end
    save(joinpath(output_dir, "model_probabilities.png"), fig)
    empty!(fig)

    fig2 = Figure(size=(600, 500))
    ax2  = Axis(fig2[1, 1]; title="Model posterior entropy",
                aspect=DataAspect(), xlabel="col", ylabel="row")
    hm = heatmap!(ax2, 1:nx, 1:ny, entropy_map'; nan_color=(:grey, 0.3), colormap=:hot)
    Colorbar(fig2[1, 2]; colormap=:hot, label="H (nats)")
    save(joinpath(output_dir, "model_entropy.png"), fig2)
    empty!(fig2)
    GC.gc()
end

function plot_rj_sed(fd::FluxData, freq::AbstractVector,
    rjsamples_file::String, output_dir::String;
    quality_mask::Union{Nothing,AbstractVector{Bool}} = nothing)

    midx, pshared, p4, _ = read_rj(rjsamples_file)
    map_idx  = map_model_index(rjsamples_file)
    n_hex    = size(midx, 1)
    sed_dir  = joinpath(output_dir, "SED_rj")
    mkpath(sed_dir)

    x_lo = minimum(freq) * 0.3
    x_hi = maximum(freq) * 3.0
    freq_range = 10 .^ range(log10(x_lo), log10(x_hi); length=200)

    all_lo = [fd.dmatrix[i, j] - fd.dnoisematrix[i, j]
              for i in 1:n_hex, j in eachindex(freq)
              if isfinite(fd.dmatrix[i, j]) && fd.dmatrix[i, j] > 0]
    all_hi = [fd.dmatrix[i, j] + fd.dnoisematrix[i, j]
              for i in 1:n_hex, j in eachindex(freq)
              if isfinite(fd.dmatrix[i, j])]
    ha_vals = filter(>(0), fd.dHa)
    y_lo = isempty(all_lo) ? 1e-10 : max(minimum(all_lo), 1e-10) * 0.3
    y_hi = isempty(all_hi) ? 1e-5  : max(maximum(all_hi), maximum(ha_vals; init=0.0)) * 3.0

    @showprogress "Plotting RJ SEDs: " for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue
        m = map_idx[i]

        in_m   = findall(==(m), midx[i, :])
        isempty(in_m) && continue
        A_samp  = pshared[i, in_m, 1]
        Bnt_samp = pshared[i, in_m, 2]
        α_samp  = pshared[i, in_m, 3]
        p4_samp = m != PL_IDX ? p4[i, in_m] : nothing

        fig = Figure()
        ax  = Axis(fig[1, 1];
            xlabel = "Frequency (Hz)", ylabel = "Flux Density (Jy)",
            xscale = log10, yscale = log10,
            limits = (x_lo, x_hi, y_lo, y_hi),
            title  = "Bin $i — MAP model: $(MODEL_NAMES[m])")

        errorbars!(ax, freq, fd.dmatrix[i, :], fd.dnoisematrix[i, :]; color=:black)
        scatter!(ax,  freq, fd.dmatrix[i, :]; color=:black)
        errorbars!(ax, [1e9], [fd.dHa[i]], [0.5 * fd.dHa[i]];
            color=:red, whiskerwidth=8)
        scatter!(ax, [1e9], [fd.dHa[i]]; color=:red)

        draw_idx = rand(1:length(in_m), min(250, length(in_m)))
        for k in draw_idx
            T   = A_samp[k]
            nT  = Bnt_samp[k]
            α   = α_samp[k]
            T > 0 && lines!(ax, freq_range, thermal(T, freq_range);
                             color=(:red, 0.0075), linewidth=10)
            if m == CPL_IDX && nT > 0
                lines!(ax, freq_range,
                    non_thermal_curved(nT, α, p4_samp[k], freq_range);
                    color=(:blue, 0.0075), linewidth=10)
            elseif m == FFA_IDX && nT > 0
                lines!(ax, freq_range,
                    non_thermal_abs(nT, α, p4_samp[k], freq_range);
                    color=(:blue, 0.0075), linewidth=10)
            elseif m == BPL_IDX && nT > 0
                lines!(ax, freq_range,
                    non_thermal_broken(nT, α, p4_samp[k], freq_range);
                    color=(:blue, 0.0075), linewidth=10)
            elseif nT > 0
                lines!(ax, freq_range, non_thermal(nT, α, freq_range);
                    color=(:blue, 0.0075), linewidth=10)
            end
        end

        best_T   = median(A_samp)
        best_nT  = median(Bnt_samp)
        best_α   = median(α_samp)
        if m == CPL_IDX
            lines!(ax, freq_range,
                sed.(best_T, best_nT, best_α, median(p4_samp), freq_range, Ref(CurvedPowerLaw()));
                color=:black)
        elseif m == FFA_IDX
            lines!(ax, freq_range,
                sed.(best_T, best_nT, best_α, median(p4_samp), freq_range, Ref(ffa()));
                color=:black)
        elseif m == BPL_IDX
            lines!(ax, freq_range,
                sed.(best_T, best_nT, best_α, median(p4_samp), freq_range, Ref(BrokenPowerLaw()));
                color=:black)
        else
            lines!(ax, freq_range, sed.(best_T, best_nT, best_α, freq_range); color=:black)
        end

        save(joinpath(sed_dir, "$i.png"), fig)
        empty!(fig)
        GC.gc()
    end
end

function plot_rj_hex_maps(rjsamples_file::String, fd::FluxData,
    dc::DataCube, hg::HexGrid, output_dir::String;
    T_e::Float64 = 8000.0,
    quality_mask::Union{Nothing,AbstractVector{Bool}} = nothing)

    midx, pshared, p4, _ = read_rj(rjsamples_file)
    n_hex    = size(midx, 1)
    ny, nx   = dc.header["NAXIS2"], dc.header["NAXIS1"]
    maps_dir = joinpath(output_dir, "maps")
    mkpath(maps_dir)

    T_map     = fill(NaN, ny, nx);  T_err     = fill(NaN, ny, nx)
    nT_map    = fill(NaN, ny, nx);  nT_err    = fill(NaN, ny, nx)
    spec_map  = fill(NaN, ny, nx);  spec_err  = fill(NaN, ny, nx)
    tfrac_map = fill(NaN, ny, nx);  tfrac_err = fill(NaN, ny, nx)
    tau_map   = fill(NaN, ny, nx);  tau_err   = fill(NaN, ny, nx)
    em_map    = fill(NaN, ny, nx);  em_err    = fill(NaN, ny, nx)
    beta_map  = fill(NaN, ny, nx);  beta_err  = fill(NaN, ny, nx)
    nu_b_map  = fill(NaN, ny, nx);  nu_b_err  = fill(NaN, ny, nx)

    for i in 1:n_hex
        quality_mask !== nothing && !quality_mask[i] && continue

        qT    = quantile(pshared[i, :, 1], [0.16, 0.50, 0.84])
        qnT   = quantile(pshared[i, :, 2], [0.16, 0.50, 0.84])
        qspec = quantile(pshared[i, :, 3], [0.16, 0.50, 0.84])

        frac  = pshared[i, :, 1] ./ (pshared[i, :, 1] .+ pshared[i, :, 2])
        qfrac = quantile(filter(isfinite, frac), [0.16, 0.50, 0.84])

        for m in [CPL_IDX, FFA_IDX, BPL_IDX]
            in_m  = findall(==(m), midx[i, :])
            isempty(in_m) && continue
            p4_m  = filter(isfinite, p4[i, in_m])
            isempty(p4_m) && continue
            qp4   = quantile(p4_m, [0.16, 0.50, 0.84])
            if m == CPL_IDX
                for px in hg.pixel_members[i]
                    beta_map[px] = qp4[2]; beta_err[px] = (qp4[3] - qp4[1]) / 2
                end
            elseif m == FFA_IDX
                em_samples = emission_measure_calc(p4_m; T_e)
                qem = quantile(em_samples, [0.16, 0.50, 0.84])
                for px in hg.pixel_members[i]
                    tau_map[px] = qp4[2]; tau_err[px] = (qp4[3] - qp4[1]) / 2
                    em_map[px]  = qem[2]; em_err[px]  = (qem[3] - qem[1]) / 2
                end
            elseif m == BPL_IDX
                for px in hg.pixel_members[i]
                    nu_b_map[px] = qp4[2]; nu_b_err[px] = (qp4[3] - qp4[1]) / 2
                end
            end
        end

        for px in hg.pixel_members[i]
            T_map[px]     = qT[2];    T_err[px]     = (qT[3]    - qT[1])    / 2
            nT_map[px]    = qnT[2];   nT_err[px]    = (qnT[3]   - qnT[1])   / 2
            spec_map[px]  = qspec[2]; spec_err[px]  = (qspec[3]  - qspec[1]) / 2
            tfrac_map[px] = qfrac[2]; tfrac_err[px] = (qfrac[3]  - qfrac[1]) / 2
        end
    end

    hdr = dc.header
    write_fits_map(joinpath(maps_dir, "T.fits"),         T_map,    hdr)
    write_fits_map(joinpath(maps_dir, "T_err.fits"),     T_err,    hdr)
    write_fits_map(joinpath(maps_dir, "nT.fits"),        nT_map,   hdr)
    write_fits_map(joinpath(maps_dir, "nT_err.fits"),    nT_err,   hdr)
    write_fits_map(joinpath(maps_dir, "spec.fits"),      spec_map, hdr)
    write_fits_map(joinpath(maps_dir, "spec_err.fits"),  spec_err, hdr)
    write_fits_map(joinpath(maps_dir, "tfrac.fits"),     tfrac_map, hdr)
    write_fits_map(joinpath(maps_dir, "tfrac_err.fits"), tfrac_err, hdr)
    write_fits_map(joinpath(maps_dir, "beta.fits"),      beta_map, hdr)
    write_fits_map(joinpath(maps_dir, "beta_err.fits"),  beta_err, hdr)
    write_fits_map(joinpath(maps_dir, "tau.fits"),       tau_map,  hdr)
    write_fits_map(joinpath(maps_dir, "tau_err.fits"),   tau_err,  hdr)
    write_fits_map(joinpath(maps_dir, "EM.fits"),        em_map,   hdr)
    write_fits_map(joinpath(maps_dir, "EM_err.fits"),    em_err,   hdr)
    write_fits_map(joinpath(maps_dir, "nu_b.fits"),      nu_b_map, hdr)
    write_fits_map(joinpath(maps_dir, "nu_b_err.fits"),  nu_b_err, hdr)
end

function all_plots_rj(fd::FluxData, dc::DataCube, hg::HexGrid,
    rjsamples_file::String, output_dir::String;
    mask_poor_bins::Bool          = true,
    min_map_model_prob::Float64   = 0.5,
    alpha_width_threshold::Float64 = 0.5,
    tfrac_width_threshold::Float64 = 0.4,
    T_e::Float64                  = 8000.0)

    qmask = mask_poor_bins ?
            rj_quality_mask(rjsamples_file;
                min_map_model_prob, alpha_width_threshold, tfrac_width_threshold) :
            nothing
    plot_rj_model_maps(rjsamples_file, dc, hg, output_dir; quality_mask=qmask)
    plot_rj_sed(fd, dc.freq, rjsamples_file, output_dir; quality_mask=qmask)
    plot_rj_hex_maps(rjsamples_file, fd, dc, hg, output_dir; T_e, quality_mask=qmask)
end

function write_fits_map(path::String, map::Matrix{Float64}, header)
    data_out = permutedims(map, (2, 1))
    isfile(path) && rm(path)
    hdus = FITSFiles.HDU[FITSFiles.HDU(data_out, header)]
    Base.write(path, hdus)
end
