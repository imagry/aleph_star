# taken from: http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html#algo2

function _draw_hline!(img, xi::Int64, xf::Int64, y::Int64, color)
    y < 1 && return
    y > size(img)[1] && return
    if xi>xf
        xi,xf = xf,xi
    end
    for x in max(xi,1):min(size(img)[2],xf)
        img[y,x,:] .= color
    end
end

function _fill_bottom_flat_triangle!(img, v1x::Int64,v1y::Int64, v2x::Int64,v2y::Int64, v3x::Int64,v3y::Int64, color)
    invslope1::Float64 = (v2x - v1x) / (v2y - v1y);    
    invslope2::Float64 = (v3x - v1x) / (v3y - v1y);
    curx1::Float64 = v1x;
    curx2::Float64 = v1x;
    scanlineY::Int64 = v1y;
    while scanlineY <= v2y
        _draw_hline!(img, floor(Int64, curx1), floor(Int64, curx2), scanlineY, color);
        curx1 += invslope1;
        curx2 += invslope2;
        scanlineY += 1
    end
end

function _fill_top_flat_triangle!(img, v1x::Int64,v1y::Int64, v2x::Int64,v2y::Int64, v3x::Int64,v3y::Int64, color)
    invslope1::Float64 = (v3x - v1x) / (v3y - v1y);
    invslope2::Float64 = (v3x - v2x) / (v3y - v2y);
    curx1::Float64 = v3x;
    curx2::Float64 = v3x;
    scanlineY::Int64 = v3y;
    while scanlineY > v1y
        _draw_hline!(img, floor(Int64, curx1), floor(Int64, curx2), scanlineY, color);
        curx1 -= invslope1;
        curx2 -= invslope2;
        scanlineY -= 1
    end
end

function draw_triangle!(img, v1x::Int64,v1y::Int64, v2x::Int64,v2y::Int64, v3x::Int64,v3y::Int64, color)
    # at first sort the three vertices by y-coordinate ascending so v1 is the topmost vertice
    if v1y > v2y
        v1x,v2x = v2x,v1x
        v1y,v2y = v2y,v1y
    end
    if v2y > v3y
        v2x,v3x = v3x,v2x
        v2y,v3y = v3y,v2y
    end
    if v1y > v2y
        v1x,v2x = v2x,v1x
        v1y,v2y = v2y,v1y
    end
    v1y > size(img)[1] && return
    v3y < 1 && return
    max(v1x,v2x,v3x) < 1 && return
    min(v1x,v2x,v3x) > size(img)[2] && return

    # here we know that v1.y <= v2.y <= v3.y
    # check for trivial case of bottom-flat triangle
    if v2y == v3y
        _fill_bottom_flat_triangle!(img, v1x,v1y, v2x,v2y, v3x,v3y, color);
    elseif v1y == v2y
        # check for trivial case of top-flat triangle
        _fill_top_flat_triangle!(img, v1x,v1y, v2x,v2y, v3x,v3y, color);
    else
        # general case - split the triangle in a topflat and bottom-flat one
        v4x::Int64 = floor(Int64, (v1x + ((v2y - v1y) / (v3y - v1y)) * (v3x - v1x)))
        v4y::Int64 = v2y
        _fill_bottom_flat_triangle!(img, v1x,v1y, v2x,v2y, v4x,v4y, color);
        _fill_top_flat_triangle!(img, v2x,v2y, v4x,v4y, v3x,v3y, color);
    end
end

# line from: https://github.com/JuliaImages/ImageDraw.jl/blob/master/src/line2d.jl
function line!(img, x0::Int, y0::Int, x1::Int, y1::Int, color)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = x0 < x1 ? 1 : -1
    sy = y0 < y1 ? 1 : -1;

    err = (dx > dy ? dx : -dy) / 2

    while true
        if y0 >= 1 && y0 <= size(img)[1] && x0 >= 1 && x0 <= size(img)[2]
            img[y0, x0] = color
        end
        (x0 != x1 || y0 != y1) || break
        e2 = err
        if e2 > -dx
            err -= dy
            x0 += sx
        end
        if e2 < dy
            err += dx
            y0 += sy
        end
    end

    img
end
