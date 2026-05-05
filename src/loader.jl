using FITSFiles

function beam_correction_factor(header)::Float64
    beam_area = π * header["BMAJ"] * header["BMIN"] / (4 * log(2))
    pixel_area = header["CDELT2"]^2
    return beam_area / pixel_area
end

function import_datacube(filelocation::String, noise_dict::Dict{String,Float64})::DataCube
    file_list = readdir(filelocation)
    maps = sort([f for f in file_list if endswith(f, ".fits") && !startswith(f, "Ha")])
    isempty(maps) && error("No non-Ha FITS files found in $filelocation")
    first_hdu = fits(joinpath(filelocation, maps[1]))[1]
    header = first_hdu.cards

    planes = map(maps) do f
        raw = fits(joinpath(filelocation, f))[1].data
        squeeze_to_2d(raw)
    end

    dcube = cat(planes...; dims=3)
    freq = [fits(joinpath(filelocation, f))[1].cards["CRVAL3"] for f in maps]
    noise = [noise_dict[f] for f in maps]
    bcf = [beam_correction_factor(fits(joinpath(filelocation, f))[1].cards) for f in maps]
    return DataCube(Float64.(dcube), Float64.(freq), maps, header, noise, bcf)
end

function load_ha(filelocation::String, ha_noise::Float64)::HaData
    path = joinpath(filelocation, "Ha.fits")
    hdu = fits(path)[1]
    raw = squeeze_to_2d(hdu.data)
    scaled = Float64.(raw) .* (1.0 / 1.5)^(-0.1)
    bcf = beam_correction_factor(hdu.cards)
    return HaData(scaled, ha_noise, bcf)
end

function squeeze_to_2d(arr::AbstractArray)
    d = ndims(arr)
    while d > 2 && size(arr, d) == 1
        arr = selectdim(arr, d, 1)
        d -= 1
    end

    while ndims(arr) > 2
        arr = selectdim(arr, ndims(arr), 1)
    end

    return permutedims(arr, (2, 1))
end