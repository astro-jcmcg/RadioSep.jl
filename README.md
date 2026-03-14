# RadioSep.jl

Bayesian separation of thermal and non-thermal radio continuum emission from multi-frequency FITS images. Emission is modelled as a two-component spectral energy distribution (SED): free-free (thermal) and synchrotron (non-thermal). Posterior distributions for each component are sampled per hexagonal aperture using an affine-invariant ensemble MCMC sampler.

This is implementation is based on the work presented in [Westcott et al. (2018)](https://academic.oup.com/mnras/article/475/4/5116/4801202).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Pipeline stages](#3-pipeline-stages)
   - [3.1 Data loading](#31-data-loading)
   - [3.2 Source masking](#32-source-masking)
   - [3.3 Hexagonal binning](#33-hexagonal-binning)
   - [3.4 Flux recovery](#34-flux-recovery)
   - [3.5 Bayesian model](#35-bayesian-model)
   - [3.6 MCMC sampling](#36-mcmc-sampling)
   - [3.7 Post-processing](#37-post-processing)
   - [3.8 Quality masking](#38-quality-masking)
   - [3.9 Output maps and plots](#39-output-maps-and-plots)
4. [Output files](#4-output-files)
5. [Example scripts](#5-example-scripts)
   - [5.1 Full pipeline in one call](#51-full-pipeline-in-one-call)
   - [5.2 Step-by-step pipeline](#52-step-by-step-pipeline)
   - [5.3 Rerunning plots from saved samples](#53-rerunning-plots-from-saved-samples)
   - [5.4 Querying individual bins](#54-querying-individual-bins)
   - [5.5 Running without quality masking](#55-running-without-quality-masking)
   - [5.6 Tuning quality mask thresholds](#56-tuning-quality-mask-thresholds)
   - [5.7 Custom MCMC settings](#57-custom-mcmc-settings)
6. [API reference](#6-api-reference)
7. [Physical background](#7-physical-background)

---

## 1. Overview

RadioSep takes a set of co-registered, single-frequency FITS images at multiple radio frequencies plus an Hα image, and produces spatially-resolved maps of:

| Output | Description |
|--------|-------------|
| `T.fits` | Thermal (free-free) flux density normalisation at 1 GHz |
| `nT.fits` | Non-thermal (synchrotron) flux density normalisation at 1 GHz |
| `spec.fits` | Non-thermal spectral index α |
| `tfrac.fits` | Thermal fraction T / (T + nT) |
| `mag_eq.fits` | Revised equipartition magnetic field strength (μG) |
| `ext.fits` | Dust extinction E(B-V) |

Each output pixel value is the median of the marginalised posterior distribution for the corresponding hex bin. Bins with poorly constrained posteriors are masked to NaN by default.

The workflow is:

```
FITS images → source mask → hexagonal apertures → SED flux densities
    → per-bin MCMC → posterior samples → maps + diagnostic plots
```

---

## 2. Installation

RadioSep.jl requires Julia ≥ 1.10. 

```julia
using Pkg

# Install from local path
Pkg.develop(path="/path/to/RadioSep.jl")

# Or activate directly
Pkg.activate("/path/to/RadioSep.jl")
Pkg.instantiate()
```

---

## 3. Pipeline stages

### 3.1 Data loading

**Functions:** `import_datacube`, `load_ha`, `beam_correction_factor`

All `.fits` files in the data directory (excluding `Ha.fits`) are loaded and stacked into a `DataCube` struct with shape `(ny, nx, n_freq)`. The frequency for each band is read from the `CRVAL3` FITS header keyword.

The beam correction factor converts summed pixel flux densities (Jy/beam) to total flux densities (Jy):

```
beam_correction = beam_area / pixel_area
beam_area = π × BMAJ × BMIN / (4 ln 2)
```

The Hα image is loaded separately and scaled to equivalent radio flux.

**Noise dictionary:** Per-band 1σ noise values (Jy/beam) must be supplied as a `Dict{String,Float64}` keyed by filename. These should represent the true total noise including both thermal noise and any calibration uncertainty — they are used directly in the likelihood without further inflation.

### 3.2 Source masking

**Function:** `gen_mask`

A pixel is included in the source mask only if it exceeds `threshold × noise` (default 5σ) in **every** frequency band simultaneously. This conservative all-band requirement prevents low-SNR pixels from entering the SED fitting where the spectral shape cannot be reliably determined.

Returns a `(ny, nx)` `BitMatrix`.

### 3.3 Hexagonal binning

**Functions:** `find_gridsize`, `build_hex_grid`

Hexagonal apertures are used because they tile the plane efficiently and provide approximately uniform area and shape, unlike rectangular bins or irregular regions. Each hex is sized to contain roughly one synthesised beam area, ensuring the bins are as small as the resolution allows while still averaging over at least one independent resolution element.

`fill_factor` (default 0.5) sets the minimum fraction of a hex's pixels that must be within the source mask for the hex to be accepted. Hexes with too few valid pixels are discarded.

The `HexGrid` struct stores:
- `pixel_members[i]` — list of `CartesianIndex` for every pixel in hex `i`
- `centers[i]` — pixel-space `(col, row)` centre of hex `i`
- `inverse_map[row, col]` — reverse lookup: which hex index owns each pixel (0 = none)

### 3.4 Flux recovery

**Function:** `recover_fluxes`

For each accepted hex bin, flux densities are summed over all pixels that are both inside the hex **and** above the source detection threshold. This intersection ensures edge pixels with low SNR do not degrade the per-bin SED.

Total flux density per bin per band:
```
S_ν = Σ(pixel fluxes) / beam_correction
```

Per-bin noise combines a 5% flux-scale term (systematic floor) in quadrature with the thermal noise:
```
σ_bin = sqrt( (0.05 × S_ν)² + (n_pixels / beam_correction) × σ_image² )
```

The solid angle of each bin is computed from the number of valid pixels and the pixel scale, and is used later for the equipartition magnetic field calculation.

### 3.5 Bayesian model

**Functions:** `sed`, `lnlike`, `lnprior`, `lnpost`

The two-component SED model is:

```
S(ν) = A × (ν/ν₀)^(-0.1)  +  B × (ν/ν₀)^α
         thermal (free-free)       non-thermal (synchrotron)
```

where `ν₀ = 1 GHz`, `A` is the thermal normalisation, `B` is the non-thermal normalisation, and `α` is the synchrotron spectral index (physically negative, typically −0.5 to −1.0).

**Log-likelihood:** Gaussian per-band residuals using the noise values from `recover_fluxes`:

```
ln L = Σ_ν  -½ [ ln(2π σ²) + (S_obs - S_model)² / σ² ]
```

**Log-prior:**

| Parameter | Prior | Notes |
|-----------|-------|-------|
| `A` (thermal norm) | Hard lower bound `A > 0`; one-sided Gaussian below Hα estimate | Prevents unphysical negative thermal flux; Hα constrains upper limit |
| `B` (non-thermal norm) | Hard lower bound `B > 0` | Prevents sign degeneracy |
| `α` (spectral index) | Gaussian, μ = −0.8, σ = 0.4 | Standard synchrotron prior |

The scaled Hα flux gives an estimate of the expected thermal free-free emission. The prior penalises the sampler if `A` falls below this estimate (the thermal contribution cannot be less than what Hα implies), but imposes no penalty above it (dust extinction can hide Hα, so the true thermal flux may exceed the Hα estimate). Formally:

```
ln π(A) = -½ ((A - A_Ha) / σ_Ha)²   if A ≤ A_Ha
         = 0                          if A > A_Ha
```

### 3.6 MCMC sampling

**Function:** `run_mcmc_hex`

Uses the affine-invariant ensemble sampler (`AffineInvariantMCMC.jl`). Default settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_walkers` | 100 | Ensemble size |
| `n_steps` | 2000 | Total steps per walker |
| `n_burnin` | 400 | Steps discarded as burn-in |
| Post-burn samples | 160,000 | Per hex bin |

**Walker initialisation:** Walkers are started in a tight log-normal ball around an analytical estimate rather than with broad uniform draws.

- Thermal init ≈ Hα-derived estimate or 10% of mean flux (whichever is larger)
- Non-thermal init ≈ remaining flux after subtracting thermal estimate
- Spectral index init = −0.8 (prior centre)

**HDF5 output:** Posterior samples are saved to `samples.h5` with dataset shape `(n_hex, n_post, 4)`:

| Column | Content |
|--------|---------|
| 1 | `therm_norm` (Jy at 1 GHz) |
| 2 | `non_therm_norm` (Jy at 1 GHz) |
| 3 | `spec` (spectral index α) |
| 4 | `Ha` (samples drawn from Hα measurement distribution) |

The Hα column stores random draws from `Normal(Ha_meas, Ha_uncert)` and is used for the extinction calculation, propagating Hα measurement uncertainty into the E(B-V) posterior.

### 3.7 Post-processing

**Functions:** `thermal_frac_percentiles`, `thermal_frac_median`, `extinction_calc`

**Thermal fraction:**
```
f_T = A / (A + B)
```
Computed sample-by-sample and summarised as 16th/50th/84th percentiles.

**Dust extinction E(B-V):**
```
E(B-V) = -(2.5 / 2.54) × log₁₀(Ha_obs / A)
```
where `Ha_obs` is the Hα-sampled column and `A` is the thermal normalisation. This ratio compares the observed Hα flux to the radio-derived expectation; any deficit indicates dust absorption.

**Equipartition magnetic field:** Uses the revised Beck & Krause (2005) formula, accounting for the energy ratio between cosmic ray protons and electrons (`k`), the line-of-sight depth through the emitting region (determined by the disk scale height and inclination), and the synchrotron spectral index. The formula is vectorised over non-thermal flux samples to produce a posterior distribution of B, summarised by its median in μG.

### 3.8 Quality masking

**Function:** `compute_quality_mask`

After MCMC, bins are evaluated against two criteria:

1. **Spectral index posterior width:** `(p84_α − p16_α) > alpha_width_threshold` (default 0.5)
   The prior 1σ width is 0.4, so a posterior wider than 0.5 indicates the data are not meaningfully constraining α — the sampler is essentially returning the prior.

2. **Thermal fraction posterior width:** `(p84_tfrac − p16_tfrac) > tfrac_width_threshold` (default 0.4)
   If the thermal/non-thermal split itself spans more than 40 percentage points, the separation is not reliable for that bin.

A bin is masked if **either** criterion is triggered. Masked bins appear as NaN in output FITS maps and are skipped in SED and corner plots. Red bin numbers in `labeled_map.png` indicate masked bins.

Set `mask_poor_bins=false` to bypass masking and inspect all bins — useful for understanding why peripheral or low-SNR bins fail.

### 3.9 Output maps and plots

**FITS maps** (`plot_hex_maps`, `mag_equip_map`, `extinction_map`): Each accepted hex bin's median posterior value is painted onto a 2D pixel grid matching the input image dimensions, producing standard FITS files that can be opened in DS9 or any FITS viewer.

**SED plots** (`plot_sed`): One PNG per bin showing observed flux densities with error bars, 250 posterior draw lines for the thermal (red) and non-thermal (blue) components, and the median best-fit total SED.

**Corner/pair plots** (`plot_corner`): One PNG per bin showing the joint and marginal posterior distributions for all three parameters, rendered using `PairPlots.jl`.

**Labeled map** (`plot_labeled_map`): A single annotated PNG showing one of the output parameters as a colour map with bin index numbers overlaid. White numbers = accepted bins; red numbers = masked bins. Use this to identify bin indices for `query_bin`.

**Scatter plots** (`spec_vs_frac`, `ext_vs_ha`): Summary plots showing spectral index vs. thermal fraction and E(B-V) vs. Hα flux across all bins.

---

## 4. Output files

After a complete run, `output_dir/` contains:

```
output_dir/
├── samples.h5              # MCMC posterior samples — (n_hex, n_post, 4)
├── labeled_map.png         # Annotated bin index map
├── spec_vs_frac.png        # Spectral index vs thermal fraction scatter plot
├── ha_vs_ext.png           # E(B-V) vs Hα scatter plot
├── maps/
│   ├── T.fits              # Thermal normalisation map
│   ├── nT.fits             # Non-thermal normalisation map
│   ├── spec.fits           # Spectral index map
│   ├── tfrac.fits          # Thermal fraction map
│   ├── mag_eq.fits         # Equipartition magnetic field map (μG)
│   └── ext.fits            # Dust extinction E(B-V) map
├── SED/
│   ├── 1.png               # SED plot for bin 1
│   ├── 2.png               # ...
│   └── ...
└── posterior/
    ├── 1.png               # Corner plot for bin 1
    ├── 2.png               # ...
    └── ...
```

---

## 5. Example scripts

### 5.1 Full pipeline in one call

The simplest way to run everything. `sep_bayes` loads data, runs MCMC, and produces all plots and maps.

```julia
using SepBayes

noise = Dict(
    "C.fits"  => 4.7e-6,
    "Ha.fits" => 5.1e-6,
    "K.fits"  => 5.1e-6,
    "L.fits"  => 35.6e-6,
    "S.fits"  => 7.2e-6,
    "U.fits"  => 6.9e-6,
    "X.fits"  => 4.6e-6,
)

dc, ha, hg, fd = sep_bayes(
    "/path/to/data/",
    noise,
    "/path/to/output/",
)
```

The returned objects (`dc`, `ha`, `hg`, `fd`) allow further interactive exploration without reloading data.

---

### 5.2 Step-by-step pipeline

Run each stage individually for full control. This is useful when iterating on MCMC settings or the masking threshold without reloading data.

```julia
using SepBayes

noise = Dict(
    "C.fits"  => 4.7e-6,
    "Ha.fits" => 5.1e-6,
    "K.fits"  => 5.1e-6,
    "L.fits"  => 35.6e-6,
    "S.fits"  => 7.2e-6,
    "U.fits"  => 6.9e-6,
    "X.fits"  => 4.6e-6,
)

datadir = "/path/to/data/"
outdir  = "/path/to/output/"

# 1. Load data
dc   = import_datacube(datadir, noise)
ha   = load_ha(datadir, noise["Ha.fits"])

# 2. Source mask (5σ detection in all bands)
mask = gen_mask(dc; threshold=5.0)

# 3. Hexagonal grid (one beam per hex, ≥50% fill)
gs   = find_gridsize(dc.header)
hg   = build_hex_grid(mask, gs; fill_factor=0.5)

println("Accepted hex bins: ", length(hg.pixel_members))

# 4. Recover per-bin flux densities
fd = recover_fluxes(dc, ha, hg, mask)

# 5. Run MCMC
run_mcmc_hex(fd, dc.freq;
             n_walkers = 100,
             n_steps   = 2000,
             n_burnin  = 400,
             output_path = outdir)

# 6. Compute quality mask
samples = joinpath(outdir, "samples.h5")
qmask   = compute_quality_mask(samples;
                                alpha_width_threshold = 0.5,
                                tfrac_width_threshold = 0.4)

# 7. Generate all plots and FITS maps
all_plots(fd, dc, hg, samples, outdir;
          mask_poor_bins = true)
```

---

### 5.3 Rerunning plots from saved samples

If the MCMC has already been run and only the plots need regenerating (e.g. after adjusting thresholds), reload the data structures and call the plotting functions directly against the existing `samples.h5`.

```julia
using SepBayes

noise = Dict(
    "C.fits"  => 4.7e-6,
    "Ha.fits" => 5.1e-6,
    "K.fits"  => 5.1e-6,
    "L.fits"  => 35.6e-6,
    "S.fits"  => 7.2e-6,
    "U.fits"  => 6.9e-6,
    "X.fits"  => 4.6e-6,
)

datadir = "/path/to/data/"
outdir  = "/path/to/output/"
samples = joinpath(outdir, "samples.h5")

# Reload data structures (fast — no MCMC)
dc   = import_datacube(datadir, noise)
ha   = load_ha(datadir, noise["Ha.fits"])
mask = gen_mask(dc)
hg   = build_hex_grid(mask, find_gridsize(dc.header))
fd   = recover_fluxes(dc, ha, hg, mask)

# Rerun all plots with default quality masking
all_plots(fd, dc, hg, samples, outdir)

# Or regenerate individual outputs:
qmask = compute_quality_mask(samples)

plot_labeled_map(samples, dc, hg, outdir; quality_mask=qmask)
plot_hex_maps(samples, dc, hg, outdir;   quality_mask=qmask)
mag_equip_map(samples, fd, dc, hg, outdir; quality_mask=qmask)
extinction_map(samples, dc, hg, outdir;  quality_mask=qmask)
plot_sed(fd, dc.freq, samples, outdir;   quality_mask=qmask)
plot_corner(samples, outdir;             quality_mask=qmask)
```

---

### 5.4 Querying individual bins

Use `labeled_map.png` to identify which bin index covers a region of interest, then call `query_bin` to retrieve the full posterior summary.

```julia
using SepBayes

# (assume dc, ha, hg, fd already loaded as above)

outdir  = "/path/to/output/"
samples = joinpath(outdir, "samples.h5")

# Look up by bin index
result = query_bin(12, samples, fd, dc, outdir)

# Or by pixel coordinate (col, row) — useful when working from a FITS viewer
result = query_bin(75, 83, samples, fd, dc, hg, outdir)

# The return value is a NamedTuple with (p16, p50, p84) for each parameter
println("Thermal fraction: ", result.tfrac)
println("Spectral index:   ", result.alpha)
println("SED plot path:    ", result.sed_path)
```

**Example output:**
```
╔══════════════════════════════════════════════════════╗
║  Bin 12                                               ║
╠══════════════════════════════════════════════════════╣
║  Thermal (Jy)          5.231e-04  [ 3.812e-04,  6.649e-04]  ║
║  Non-thermal (Jy)      8.104e-04  [ 6.920e-04,  9.287e-04]  ║
║  Spectral index          -0.7341  [   -0.8623,    -0.6059]  ║
║  Thermal fraction         0.3921  [    0.2844,     0.4998]  ║
║  Hα (Jy)               6.112e-04  [ 4.278e-04,  7.946e-04]  ║
║  E(B-V)                   0.1834  [    0.0412,     0.3256]  ║
╠══════════════════════════════════════════════════════╣
║  SED:    /path/to/output/SED/12.png                          ║
║  Corner: /path/to/output/posterior/12.png                    ║
╚══════════════════════════════════════════════════════╝
```

---

### 5.5 Running without quality masking

To inspect all bins including poorly constrained ones — useful for understanding why edge bins are rejected or for sanity-checking the MCMC on a new dataset.

```julia
# Via sep_bayes
dc, ha, hg, fd = sep_bayes(
    "/path/to/data/", noise, "/path/to/output/unmasked/";
    mask_poor_bins = false,
)

# Via all_plots
all_plots(fd, dc, hg, samples, outdir; mask_poor_bins=false)
```

In `labeled_map.png`, all bin numbers will appear in white regardless of quality. Maps will include values for all bins.

---

### 5.6 Tuning quality mask thresholds

The default thresholds may need adjustment depending on the dataset. A source with few frequency bands will have broader posteriors in general; a source with many well-separated bands may warrant tighter thresholds.

```julia
# Check how many bins pass at different thresholds
for α_thresh in [0.4, 0.5, 0.6, 0.7]
    qmask = compute_quality_mask(samples;
                                  alpha_width_threshold = α_thresh,
                                  tfrac_width_threshold = 0.4)
    println("alpha_width ≤ $α_thresh: ", count(qmask), " bins pass")
end

# Apply chosen thresholds
all_plots(fd, dc, hg, samples, outdir;
          mask_poor_bins         = true,
          alpha_width_threshold  = 0.6,
          tfrac_width_threshold  = 0.5)
```

You can also pass a manually constructed mask (e.g. based on external criteria):

```julia
qmask = trues(length(hg.pixel_members))
qmask[32] = false   # exclude isolated outlier bin

plot_hex_maps(samples, dc, hg, outdir; quality_mask=qmask)
mag_equip_map(samples, fd, dc, hg, outdir; quality_mask=qmask)
extinction_map(samples, dc, hg, outdir; quality_mask=qmask)
plot_sed(fd, dc.freq, samples, outdir; quality_mask=qmask)
plot_corner(samples, outdir; quality_mask=qmask)
plot_labeled_map(samples, dc, hg, outdir; quality_mask=qmask)
```

---

### 5.7 Custom MCMC settings

For large datasets or when convergence is uncertain, increase the chain length. For quick tests, reduce it.

```julia
# Fast test run (fewer walkers and steps)
run_mcmc_hex(fd, dc.freq;
             n_walkers   = 50,
             n_steps     = 500,
             n_burnin    = 100,
             output_path = outdir)

# High-fidelity run (longer chains for better-sampled posteriors)
run_mcmc_hex(fd, dc.freq;
             n_walkers   = 200,
             n_steps     = 5000,
             n_burnin    = 1000,
             output_path = outdir)

# Or via sep_bayes
dc, ha, hg, fd = sep_bayes(
    datadir, noise, outdir;
    n_walkers = 200,
    n_steps   = 5000,
    n_burnin  = 1000,
)
```

---

## 6. API reference

### Top-level

| Function | Description |
|----------|-------------|
| `sep_bayes(filelocation, noise_dict, output_dir; kwargs...)` | Run the complete pipeline in one call |
| `complete_run(filelocation, noise_dict, output_dir; kwargs...)` | Load → mask → hexbin → fluxes → MCMC (no plotting) |

### Data loading

| Function | Returns | Description |
|----------|---------|-------------|
| `import_datacube(filelocation, noise_dict)` | `DataCube` | Load all radio FITS files into a stacked array |
| `load_ha(filelocation, ha_noise)` | `HaData` | Load and scale Hα image |
| `beam_correction_factor(header)` | `Float64` | Compute beam_area / pixel_area |

### Masking and binning

| Function | Returns | Description |
|----------|---------|-------------|
| `gen_mask(dc; threshold=5.0)` | `BitMatrix` | All-band SNR source mask |
| `find_gridsize(header)` | `Int` | Compute hex grid size from beam and image size |
| `build_hex_grid(mask, gridsize; fill_factor=0.5)` | `HexGrid` | Build hexagonal aperture grid |
| `recover_fluxes(dc, ha, hg, mask)` | `FluxData` | Per-bin flux densities and noise |

### MCMC

| Function | Returns | Description |
|----------|---------|-------------|
| `run_mcmc_hex(fd, freq; n_walkers, n_steps, n_burnin, output_path)` | nothing | Run MCMC, save `samples.h5` |

### Post-processing

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_quality_mask(samples_file; alpha_width_threshold, tfrac_width_threshold)` | `BitVector` | Flag poorly constrained bins |
| `thermal_frac_percentiles(samples)` | `(p16, p50, p84)` | Thermal fraction percentiles |
| `thermal_frac_median(samples)` | `Float64` | Median thermal fraction |
| `extinction_calc(samples)` | `Vector{Float64}` | E(B-V) posterior samples |
| `bfield_revised(flux_density, area, freq, spec, k, scale_height, inclination)` | array | Beck & Krause (2005) B field |

### Plotting and maps

| Function | Description |
|----------|-------------|
| `all_plots(fd, dc, hg, samples_file, outdir; mask_poor_bins, ...)` | Run all plots and maps |
| `plot_hex_maps(samples_file, dc, hg, outdir; quality_mask)` | Write T, nT, spec, tfrac FITS maps |
| `mag_equip_map(samples_file, fd, dc, hg, outdir; quality_mask)` | Write mag_eq.fits |
| `extinction_map(samples_file, dc, hg, outdir; quality_mask)` | Write ext.fits |
| `plot_sed(fd, freq, samples_file, outdir; quality_mask)` | Write SED PNGs |
| `plot_corner(samples_file, outdir; quality_mask)` | Write corner plot PNGs |
| `plot_labeled_map(samples_file, dc, hg, outdir; parameter, quality_mask)` | Write annotated bin index map |
| `spec_vs_frac(samples_file, outdir)` | Spectral index vs thermal fraction scatter plot |
| `ext_vs_ha(samples_file, outdir)` | E(B-V) vs Hα scatter plot |

### Bin lookup

| Function | Returns | Description |
|----------|---------|-------------|
| `query_bin(bin_idx, samples_file, fd, dc, outdir)` | `NamedTuple` | Print and return posterior summary for a bin |
| `query_bin(col, row, samples_file, fd, dc, hg, outdir)` | `NamedTuple` | Same, identified by pixel coordinate |

---

## 7. Physical background

### Free-free (thermal) emission

Free-free radiation from ionised gas (HII regions) has a nearly flat spectrum with spectral index α ≈ −0.1 in the optically thin regime.

```
S_thermal(ν) ∝ ν^(-0.1)
```

The Hα recombination line emission is produced by the same ionised gas and provides an independent constraint on the thermal radio flux, used here as a prior.

### Synchrotron (non-thermal) emission

Relativistic electrons spiralling in magnetic fields produce synchrotron radiation with a power-law spectrum:

```
S_synchrotron(ν) ∝ ν^α,   α typically −0.5 to −1.0
```

The spectral index encodes information about the cosmic ray electron energy spectrum. Steeper values indicate older electron populations that have lost energy to synchrotron and inverse Compton losses.

### Equipartition magnetic field

Under the assumption of minimum energy (approximate equipartition between cosmic ray and magnetic field energy densities), the revised Beck & Krause (2005) formula gives the magnetic field strength from the synchrotron surface brightness. Key inputs are the non-thermal flux density, the line-of-sight path length through the emitting region (inferred from the disk scale height and inclination), the proton-to-electron energy ratio `k` (typically assumed 100 for a normal ISM), and the observing frequency.

### Dust extinction

The ratio of observed Hα flux to the radio-inferred thermal free-free flux measures dust absorption of the Hα line:

```
E(B-V) = -(2.5 / 2.54) × log₁₀(Ha_obs / S_thermal)
```

A positive E(B-V) indicates absorbed Hα (the observed Hα underestimates the true ionised gas content). Negative values can arise from measurement scatter or when the Hα flux exceeds the radio thermal estimate (suggesting the simple spectral model overestimates the thermal component at that location).
