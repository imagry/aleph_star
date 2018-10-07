struct Segment
    x1::Float32
    y1::Float32
    x2::Float32
    y2::Float32
    dx::Float32
    dy::Float32
    d2::Float32
    minx::Float32
    maxx::Float32
    miny::Float32
    maxy::Float32
    ix::Int32
    function Segment(x1,y1,x2,y2,ix=-1)
        dx = x2-x1
        dy = y2-y1
        minx, maxx = if x1 < x2 x1, x2 else x2, x1 end
        miny, maxy = if y1 < y2 y1, y2 else y2, y1 end
        new(
            x1,y1,x2,y2,
            dx,
            dy,
            dx*dx+dy*dy,
            minx, maxx,
            miny, maxy,
            ix
        )
    end
end

_get_cos_alpha(s1,s2) = (s1.dx*s2.dx + s1.dy*s2.dy)/sqrt(s1.d2)/sqrt(s2.d2)
_get_sin_alpha(s1,s2) = (s1.dx*s2.dy - s1.dy*s2.dx)/sqrt(s1.d2)/sqrt(s2.d2)

function  _on_same_side(seg1::Segment, seg2::Segment)
    a2x = seg1.dx
    a2y = seg1.dy
    b1x = seg2.x1 - seg1.x1
    b1y = seg2.y1 - seg1.y1
    b2x = seg2.x2 - seg1.x1
    b2y = seg2.y2 - seg1.y1
    s1 = a2x * b1y - a2y * b1x
    s2 = a2x * b2y - a2y * b2x
    return s1 * s2 > 0.0
end

_cross(seg1, seg2) =
    !(_on_same_side(seg1, seg2) || _on_same_side(seg2, seg1))

function test_coll(c::Car, seg::Segment, safety_fac=1.2)
    msx(x) = c.cx+(x-c.cx)*safety_fac
    msy(y) = c.cy+(y-c.cy)*safety_fac
    if (msx(c.maxx) < seg.minx || msy(c.maxy) < seg.miny ||
       msx(c.minx) > seg.maxx || msy(c.miny) > seg.maxy)
        return false
    end
    seg1 = Segment(msx(c.rrx), msy(c.rry), msx(c.rlx), msy(c.rly))
    seg2 = Segment(msx(c.frx), msy(c.fry), msx(c.flx), msy(c.fly))
    seg3 = Segment(msx(c.rrx), msy(c.rry), msx(c.frx), msy(c.fry))
    seg4 = Segment(msx(c.rlx), msy(c.rly), msx(c.flx), msy(c.fly))
    return _cross(seg, seg1) || _cross(seg, seg2) || _cross(seg, seg3) || _cross(seg, seg4)
end

function test_coll(c1::Car, c2::Car, safety_fac=1.2)
    msx(x) = c1.cx+(x-c1.cx)*safety_fac # to be only used on car1 !!!
    msy(y) = c1.cy+(y-c1.cy)*safety_fac # to be only used on car1 !!!
    if (msx(c1.maxx) < c2.minx || msy(c1.maxy) < c2.miny ||
        msx(c1.minx) > c2.maxx || msy(c1.miny) > c2.maxy)
        return false
    end
    seg1 = Segment(msx(c1.rrx), msy(c1.rry), msx(c1.rlx), msy(c1.rly))
    seg2 = Segment(msx(c1.frx), msy(c1.fry), msx(c1.flx), msy(c1.fly))
    seg3 = Segment(msx(c1.rrx), msy(c1.rry), msx(c1.frx), msy(c1.fry))
    seg4 = Segment(msx(c1.rlx), msy(c1.rly), msx(c1.flx), msy(c1.fly))
    sega = Segment(c2.rrx, c2.rry, c2.rlx, c2.rly)
    segb = Segment(c2.frx, c2.fry, c2.flx, c2.fly)
    segc = Segment(c2.rrx, c2.rry, c2.frx, c2.fry)
    segd = Segment(c2.rlx, c2.rly, c2.flx, c2.fly)
    return (
        _cross(sega, seg1) || _cross(sega, seg2) || _cross(sega, seg3) || _cross(sega, seg4) ||
        _cross(segb, seg1) || _cross(segb, seg2) || _cross(segb, seg3) || _cross(segb, seg4) ||
        _cross(segc, seg1) || _cross(segc, seg2) || _cross(segc, seg3) || _cross(segc, seg4) ||
        _cross(segd, seg1) || _cross(segd, seg2) || _cross(segd, seg3) || _cross(segd, seg4)
        )
end

function get_beam_coll(origin_x,origin_y, segments::Vector{Segment}, ang, max_sensor_length)
    cosa = cos(ang)
    sina = sin(ang)
    dseg = Segment(
        origin_x, origin_y,
        origin_x + max_sensor_length * cosa, origin_y + max_sensor_length * sina
    )

    best_d2 = dseg.d2
    best_px = dseg.x2
    best_py = dseg.y2
    best_sa = -1.0
    best_ix::Int32 = -1
    for seg in segments
        _on_same_side(dseg, seg) && continue
        # we have a collision, lets find where exactly:
        v1x = dseg.x1 - seg.x1
        v1y = dseg.y1 - seg.y1

        v2x = seg.dx
        v2y = seg.dy

        v3x = -dseg.dy                                  
        v3y = dseg.dx

        v1v3 = v1x * v3x + v1y * v3y
        v2v3 = v2x * v3x + v2y * v3y
        t2 = v1v3 / v2v3

        px = seg.x1 + t2 * v2x
        py = seg.y1 + t2 * v2y
        dx = px - dseg.x1
        dy = py - dseg.y1
        if sign(dx)!=sign(cosa) || sign(dy)!=sign(sina)
            # it's a backwards ray
            continue
        end
        d2 = dx * dx + dy * dy

        if d2 < best_d2
            best_ix = seg.ix
            best_d2 = d2
            best_px = px
            best_py = py
            best_sa = _get_sin_alpha(dseg, seg)
        end
    end
    return best_px, best_py, best_d2, best_sa, best_ix
end

mutable struct Collider
    cell_dx::Float64
    d::Dict{Tuple{Int64,Int64},Vector{Segment}}
    hasmem::Bool                 # for caching result of last query
    memkey::Tuple{Int64,Int64}   #     |
    mem::Vector{Segment}         # ____|
    Collider(cell_dx) = new(cell_dx, Dict{Tuple{Int64,Int64},Vector{Segment}}(), false, (0,0), Segment[])
end

function get_plot_all(c::Collider)
    x = Float64[]
    y = Float64[]
    for segs in values(c.d)
        for seg in segs
            push!(x,seg.x1)
            push!(x,seg.x2)
            push!(x,NaN)
            push!(y,seg.y1)
            push!(y,seg.y2)
            push!(y,NaN)
        end
    end
    x,y
end

float_to_key(dx,x) = floor(Int64, x/dx)

function addp!(c::Collider,seg::Segment)
    k1 = (float_to_key(c.cell_dx,seg.x1), float_to_key(c.cell_dx,seg.y1))
    k2 = (float_to_key(c.cell_dx,seg.x2), float_to_key(c.cell_dx,seg.y2))
    if !haskey(c.d,k1)
        c.d[k1] = Segment[]
    end
    push!( c.d[k1], seg)
    if k1 != k2
        if !haskey(c.d,k2)
            c.d[k2] = Segment[]
        end
        push!(c.d[k2], seg)
    end
end
function addp!(c::Collider,segs::Vector{Segment})
    for seg in segs
        addp!(c,seg)
    end
end
function getsegs(c::Collider,x,y)
    kxc = float_to_key(c.cell_dx, x)
    kyc = float_to_key(c.cell_dx, y)
    if c.hasmem && c.memkey==(kxc,kyc)
        return c.mem
    end
    all_segs = Segment[]
    for kx in (kxc-1):(kxc+1), ky in (kyc-1):(kyc+1)
        append!(all_segs, get(c.d, (kx,ky), Segment[]))
    end
    c.hasmem = true
    c.memkey = (kxc,kyc)
    c.mem = unique(all_segs)
    return c.mem
end



