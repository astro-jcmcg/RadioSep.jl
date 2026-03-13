module RadioSep

using FITSFiles
using Statistics
using Distributions
using HDF5
using AffineInvariantMCMC
using CairoMakie
using PairPlots
using ProgressMeter
using SpecialFunctions

struct DataCube
    data::Array{Float64,3}
    freq::Vector{Float64}
    maps::Vector{String}
    header::Any
    noise::Vector{Float64}
    beam_correction::Vector{Float64}
end

struct HaData
    flux_scaled::Matrix{Float64}
    noise::Float64
    beam_correction::Float64
end

struct HexGrid
    pixel_members::Vector{Vector{CartesianIndex{2}}}
    centers::Vector{Tuple{Float64,Float64}}
    gridsize::Int
    xmin::Float64
    ymin::Float64
    sx::Float64
    sy::Float64
    nx_hex::Int
    ny_hex::Int
    inverse_map::Matrix{Int}
end

struct FluxData
    dmatrix::Matrix{Float64}
    dnoisematrix::Matrix{Float64}
    dHa::Vector{Float64}
    dHa_noise::Vector{Float64}
    area::Vector{Float64}
    hex_pos::Vector{Tuple{Float64,Float64}}
end

include("loader.jl")
include("masking.jl")
include("hexbin.jl")
include("fluxes.jl")
include("model.jl")
include("mcmc.jl")
include("equipartition.jl")
include("postprocess.jl")
include("plotting.jl")

export DataCube, HaData, HexGrid, FluxData
export import_datacube, load_ha, beam_correction_factor
export gen_mask
export build_hex_grid, find_gridsize
export recover_fluxes
export sed, lnlike, lnprior, lnpost
export run_mcmc_hex
export bfield_revised
export thermal_frac_percentiles, thermal_frac_median, extinction_calc, compute_quality_mask
export plot_sed, plot_corner, spec_vs_frac, ext_vs_ha
export plot_hex_maps, mag_equip_map, extinction_map, all_plots
export plot_labeled_map, query_bin

function complete_run(filelocation::String, noise_dict::Dict{String,Float64},
    output_dir::String;
    threshold::Float64=5.0,
    fill_factor::Float64=0.5)
    dc = import_datacube(filelocation, noise_dict)
    ha = load_ha(filelocation, noise_dict["Ha.fits"])
    mask = gen_mask(dc; threshold)
    gs = find_gridsize(dc.header)
    hg = build_hex_grid(mask, gs; fill_factor)
    fd = recover_fluxes(dc, ha, hg, mask)
    run_mcmc_hex(fd, dc.freq; output_path=output_dir)
    return dc, ha, hg, fd
end

export complete_run

function sep_bayes(filelocation::String, noise_dict::Dict{String,Float64},
    output_dir::String;
    threshold::Float64=5.0,
    fill_factor::Float64=0.5,
    n_walkers::Int=100,
    n_steps::Int=2000,
    n_burnin::Int=400,
    mask_poor_bins::Bool=true,
    alpha_width_threshold::Float64=0.5,
    tfrac_width_threshold::Float64=0.4)

    dc = import_datacube(filelocation, noise_dict)
    ha = load_ha(filelocation, noise_dict["Ha.fits"])
    mask = gen_mask(dc; threshold)
    gs = find_gridsize(dc.header)
    hg = build_hex_grid(mask, gs; fill_factor)
    fd = recover_fluxes(dc, ha, hg, mask)

    samples_file = joinpath(output_dir, "samples.h5")
    run_mcmc_hex(fd, dc.freq; n_walkers, n_steps, n_burnin, output_path=output_dir)

    all_plots(fd, dc, hg, samples_file, output_dir;
        mask_poor_bins, alpha_width_threshold, tfrac_width_threshold)

    return dc, ha, hg, fd
end

export sep_bayes

end