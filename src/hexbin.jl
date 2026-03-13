function find_gridsize(header)::Int
    image_length = header["NAXIS1"] * header["CDELT2"]
    image_beam_a = π * header["BMAJ"] * header["BMIN"] / (4 * log(2))
    hex_width = sqrt((2 * image_beam_a) / sqrt(3))
    return Int(ceil(image_length / hex_width))
end

function _hex_geometry_vertex(nx_pixels::Int, ny_pixels::Int, gridsize::Int)
    nx_hex = gridsize

    xmin, xmax = 0.0, Float64(nx_pixels)
    ymin = 0.0

    padding = 1e-9 * (xmax - xmin)
    xmin -= padding
    xmax += padding

    sx = (xmax - xmin) / nx_hex
    sy = sx * sqrt(3)
    ny_hex = ceil(Int, ny_pixels / sy)

    return xmin, ymin, sx, sy, nx_hex, ny_hex
end

function compute_hex_membership(mask::BitMatrix, gridsize::Int)
    ny, nx = size(mask)

    xmin, ymin, sx, sy, nx_hex, ny_hex = _hex_geometry_vertex(nx, ny, gridsize)

    nx1, ny1 = nx_hex + 1, ny_hex + 1
    nx2, ny2 = nx_hex, ny_hex
    n_primary = nx1 * ny1

    n_bins = n_primary + nx2 * ny2
    bin_vals = [Float64[] for _ in 1:n_bins]
    bin_pixels = [CartesianIndex{2}[] for _ in 1:n_bins]

    for row in 1:ny
        for col in 1:nx
            val = Float64(mask[row, col])

            x = col - 1.0
            y = row - 1.0

            ix = (x - xmin) / sx
            iy = (y - ymin) / sy

            ix1 = round(Int, ix)
            iy1 = round(Int, iy)
            ix2 = floor(Int, ix)
            iy2 = floor(Int, iy)

            d1 = (ix - ix1)^2 + 3 * (iy - iy1)^2
            d2 = (ix - ix2 - 0.5)^2 + 3 * (iy - iy2 - 0.5)^2

            if d1 < d2
                if 0 ≤ ix1 < nx1 && 0 ≤ iy1 < ny1
                    idx = ix1 * ny1 + iy1 + 1
                    push!(bin_vals[idx], val)
                    push!(bin_pixels[idx], CartesianIndex(row, col))
                end
            else
                if 0 ≤ ix2 < nx2 && 0 ≤ iy2 < ny2
                    idx = n_primary + ix2 * ny2 + iy2 + 1
                    push!(bin_vals[idx], val)
                    push!(bin_pixels[idx], CartesianIndex(row, col))
                end
            end
        end
    end

    mean_values = [isempty(b) ? NaN : mean(b) for b in bin_vals]

    geom = (xmin=xmin, ymin=ymin, sx=sx, sy=sy,
        nx_hex=nx_hex, ny_hex=ny_hex,
        nx1=nx1, ny1=ny1, nx2=nx2, ny2=ny2,
        n_primary=n_primary)

    return bin_pixels, mean_values, geom
end

function build_hex_grid(mask::BitMatrix, gridsize::Int; fill_factor::Float64=0.5)::HexGrid
    ny, nx = size(mask)

    bin_pixels, mean_values, geom = compute_hex_membership(mask, gridsize)

    n_bins = length(mean_values)
    accepted = findall(i -> !isnan(mean_values[i]) && mean_values[i] >= fill_factor,
        1:n_bins)

    pixel_members = bin_pixels[accepted]

    centers = Vector{Tuple{Float64,Float64}}(undef, length(accepted))
    for (j, idx) in enumerate(accepted)
        if idx <= geom.n_primary
            zero_idx = idx - 1
            ix1 = zero_idx ÷ geom.ny1
            iy1 = zero_idx % geom.ny1
            cx = ix1 * geom.sx + geom.xmin + 1.0
            cy = iy1 * geom.sy + geom.ymin + 1.0
        else
            zero_idx = idx - geom.n_primary - 1
            ix2 = zero_idx ÷ geom.ny2
            iy2 = zero_idx % geom.ny2
            cx = (ix2 + 0.5) * geom.sx + geom.xmin + 1.0
            cy = (iy2 + 0.5) * geom.sy + geom.ymin + 1.0
        end
        centers[j] = (cx, cy)
    end

    inverse_map = zeros(Int, ny, nx)
    for (j, members) in enumerate(pixel_members)
        for px in members
            inverse_map[px] = j
        end
    end

    return HexGrid(pixel_members, centers, gridsize,
        geom.xmin, geom.ymin, geom.sx, geom.sy,
        geom.nx_hex, geom.ny_hex, inverse_map)
end