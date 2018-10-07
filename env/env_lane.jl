import Dierckx
import DSP: conv

const actionc_steer = 7
const actionc_accel = 5
const MINSPEED = -0.9
const MAXACCEL = 4.0 # meter / second^2
const MAXSTEER = pi/30  # in radians. This is 90 deg of steerwheel asumming steer factor of 15
const MAXSPEED = 14.0 # in meter / second
const YOLOCAR_MAX_VEL = 5.0 # meter / second
const YOLOCAR_MIN_VEL = 0.9 # meter / second
const YOLOCAR_MAX_ACCEL = 6.0 # meter / second^2
const DRAWING_SCALE = 0.3
const SAFETY = 1.3


include("draw.jl")
include("car.jl")
include("collisions.jl")

function gen_simple_road(n, width, len, curvy, smth, seed=1)
    #rand(seed)
    smth = smth*(1000/n)
    da = (rand(n + 102) .- 0.5) .* 2.0 .* pi .* curvy
    a = Float64[da[1]]
    for i in 2:length(da)
        v = a[end] + da[i]
        if abs(v)>pi/2
            v -= 2.0*da[i]
        end
        push!(a, v)
    end
    l = range(-100, stop=100, length=n)
    g = exp.(-l .* l / smth / smth)
    a = conv(g, a)[div(n,2):end-div(n,2)+1]
    a /= maximum(abs.(a))
    a *= pi/2
    
    a .-= a[50]
    dl = len / n
    dx = dl * cos.(a)
    dy = dl * sin.(a)
    xl = cumsum(dx)
    yl = cumsum(dy)
    a .+= pi / 2
    dx = width .* cos.(a)
    dy = width .* sin.(a)
    xr = xl + dx
    yr = yl + dy
    xr = xr[50:end-50]
    yr = yr[50:end-50]
    xl = xl[50:end-50]
    yl = yl[50:end-50]
    a = a[50:end-50]
    x0 = xr[1]
    y0 = 0.5*(yr[1].+yl[1])
    return xr.-x0, yr.-y0, xl.-x0, yl.-y0, 0.5*(xr.+xl).-x0, 0.5*(yr.+yl).-y0, a
end

struct MyState
    rcx::Float32
    rcy::Float32
    angle::Float32
    steer::Float32
    accel::Float32
    speed::Float32
    yolocar_dist::Float32
    yolocar_timestep::Int32
    times_passed_yolocar::Int32
end

MyState(car::Car, yolocar_dist, yolocar_timestep, times_passed) =
    MyState(car.rcx,car.rcy,car.angle,car.steer,car.accel,car.speed,yolocar_dist, yolocar_timestep, times_passed)

mutable struct MyEnv
    coll::Collider
    max_dist::Float64
    splx::Dierckx.Spline1D
    sply::Dierckx.Spline1D
    yolocar_width::Float64
    yolocar_height::Float64
    yolocar_vels::Vector{Float32}
    triangle_points::Vector{Tuple{Float32,Float32}}
end

function action_ix_to_action(::MyEnv, ix::Int32)
    R = div(ix-1, actionc_accel) + 1
    C = ix-(R-1)*actionc_accel
    steer = range(-MAXSTEER, stop=MAXSTEER, length=actionc_steer)[R]
    accel = range(-MAXACCEL, stop=MAXACCEL, length=actionc_accel)[C]
    (steer, accel)
end

function initialize_simple_road(xl,yl,xr,yr,xc,yc,v=-1.0)
    segs = Segment[]
    triangle_points = Tuple{Float64,Float64}[]
    ix = 1
    for i in 1:(length(xr)-1)
        push!(segs,Segment(xl[i],yl[i],xl[i+1],yl[i+1],ix))
        push!(triangle_points, (xr[i+1],yr[i+1]))
        ix +=1
        push!(segs,Segment(xr[i],yr[i],xr[i+1],yr[i+1],ix))
        push!(triangle_points, (xl[i],yl[i]))
        ix += 1
    end
    push!(segs,Segment(xr[1],yr[1],xl[1],yl[1],ix))
    speed = if v<0.0 0.01 + rand()*MAXSPEED else v end
    av = if v<0.0
        Car(2.0 + 2.0*(rand()-0.5),2.0*(rand()-0.5), randn()*0.08,0.0,0.0,speed)
    else
        Car(2.0,0.0, 0.0,0.0,0.0,speed)
    end
    col = Collider(71.0)
    addp!(col, segs)

    yolocar_dfac = 0.1 + rand()*0.8 # right vs left weighting
    path_x = xl.*yolocar_dfac + xr.*(1.0-yolocar_dfac)
    path_y = yl.*yolocar_dfac + yr.*(1.0-yolocar_dfac)
    dx = diff(path_x)
    dy = diff(path_y)
    d = [0; cumsum(sqrt.(dx.*dx + dy.*dy))]
    splx = Dierckx.Spline1D(d, path_x)
    sply = Dierckx.Spline1D(d, path_y)

    yolocar_dist = 40.0 + randn()*7.0
    yolocar_width = 1.6 + randn()*0.1
    yolocar_height = 1.5 + randn()*0.1

    yolocar_vels = Float64[rand()*YOLOCAR_MAX_VEL]
    for _ in 1:50000
        a = 2*(rand()-0.5)*YOLOCAR_MAX_ACCEL
        dv = 0.1*a
        if yolocar_vels[end]+dv>YOLOCAR_MAX_VEL || yolocar_vels[end]+dv<YOLOCAR_MIN_VEL
            dv = -dv
        end
        push!(yolocar_vels, yolocar_vels[end]+dv)
    end
    l = range(-120,stop=120,length=210)
    yolocar_vels = conv(yolocar_vels,exp.(-l.*l/1000))/200.0*18.0/3.7
    yolocar_vels = yolocar_vels[1000:end-1000]

    # scaling yolo car vel between max and min:
    yolocar_vels .-= minimum(yolocar_vels)
    yolocar_vels ./= maximum(yolocar_vels)
    yolocar_vels .*= (YOLOCAR_MAX_VEL - YOLOCAR_MIN_VEL)
    yolocar_vels .+= YOLOCAR_MIN_VEL

    yolocar_timestep = 1
    times_passed::Int32 = 0
    return MyState(av,yolocar_dist,yolocar_timestep, times_passed), MyEnv(col,d[end],splx,sply,yolocar_width,yolocar_height,yolocar_vels, triangle_points)
end

function initialize_simple_road(v=-1.0)
    width = 4.9 + rand()*2.1
    # a very long route (100km)
    xr,yr,xl,yl, xc,yc, a = gen_simple_road(round(Int64,100000/5), width, 500000.0/5, 0.1, 5.7, -1.0);
    return initialize_simple_road(xl,yl,xr,yr,xc,yc,v)
end

function iscolliding(av::Car, borders::Collider, safety_fac=1.3)
    for seg in getsegs(borders,av.cx,av.cy)
        test_coll(av, seg, safety_fac) && return true
    end
    return false
end

function my_calc_reward(av::Car, borders::Collider, yolo_x, yolo_y, safety=1.3)
    av.rcx < 0.0 && return 0.0, 1 # don't let it escape the road by reversing
    av.speed < MINSPEED && return 0.0, 1
    av.speed > MAXSPEED && return 0.0, 1
    iscolliding(av, borders, safety) && return 0.0, 2
    dx = av.cx - yolo_x
    dy = av.cy - yolo_y
    d2 = dx*dx + dy*dy
    d2 < 1.5*1.5*safety*safety && return 0.0, 3 # colliding with yolo car
    _,_,d2_1,_,_ = get_beam_coll(av.fcx,av.fcy, getsegs(borders,av.cx,av.cy), -pi/2+av.angle, 50.0)
    _,_,d2_2,_,_ = get_beam_coll(av.fcx,av.fcy, getsegs(borders,av.cx,av.cy),  pi/2+av.angle, 50.0)
    l = sqrt(d2_1)
    r = sqrt(d2_2)
    return (2.0*(av.speed>0.0) + max(0.0, av.speed)^1.2)*(min(l,r)/max(l,r))^1.5, 0
end

function calc_reward(av::Car, borders::Collider, yolo_x, yolo_y, safety=1.3)
    reward, result = my_calc_reward(av, borders, yolo_x, yolo_y, safety)
    return reward, result>0
end

getcar(state::MyState) =
    Car(state.rcx,state.rcy,state.angle,state.steer,state.accel,state.speed)

function get_sensors(env::MyEnv, state::MyState)
    car = getcar(state)
    spd = max(0.0, car.speed)
    bkg = floor(UInt8, 200.0*spd/MAXSPEED)
    img = ones(UInt8, 84,84) * bkg
    c = cos(-car.angle+pi/2)
    s = sin(-car.angle+pi/2)
    # plotting the road
    coll = env.coll
    segs = getsegs(coll, car.cx, car.cy)
    for seg in segs
        seg.ix > length(env.triangle_points) && continue
        # move and rotate the segment...
        x1 = seg.x1 - car.cx
        y1 = seg.y1 - car.cy
        x1,y1 = x1*c-y1*s, x1*s+y1*c
        x2 = seg.x2 - car.cx
        y2 = seg.y2 - car.cy
        x2,y2 = x2*c-y2*s, x2*s+y2*c
        xt = env.triangle_points[seg.ix][1] - car.cx
        yt = env.triangle_points[seg.ix][2] - car.cy
        xt,yt = xt*c-yt*s, xt*s+yt*c

        # to image coordinates:
        x1 = floor(Int64, x1/DRAWING_SCALE) + 42
        x2 = floor(Int64, x2/DRAWING_SCALE) + 42
        xt = floor(Int64, xt/DRAWING_SCALE) + 42
        y1 = floor(Int64, y1/DRAWING_SCALE)
        y2 = floor(Int64, y2/DRAWING_SCALE)
        yt = floor(Int64, yt/DRAWING_SCALE)

        # plot!
        draw_triangle!(img, xt,yt, x2,y2, x1,y1, 255)
        line!(img, x1,y1, x2,y2, 255)
    end

    # drawing the yolo-car
    xc = env.splx(state.yolocar_dist)
    yc = env.sply(state.yolocar_dist)

    # move and rotate
    xc -= car.cx
    yc -= car.cy
    xc,yc = xc*c-yc*s, xc*s+yc*c

    # to image coordinates:
    xc = floor(Int64, xc/DRAWING_SCALE) + 42
    yc = floor(Int64, yc/DRAWING_SCALE)

    # plot!
    maxdv = YOLOCAR_MAX_VEL - MINSPEED
    mindv = YOLOCAR_MIN_VEL - MAXSPEED

    spd = env.yolocar_vels[state.yolocar_timestep]-car.speed
    spd -= mindv
    spd /= maxdv - mindv
    spd += 0.1
    spd *= 200.0
    if spd < 0.0
        @show spd
        @show mindv
        @show maxdv
        @show env.yolocar_vels[state.yolocar_timestep]
        @show car.speed
    end
    yolo_color = floor(UInt8, spd)
    for x in (xc-2):(xc+2), y in (yc-2):(yc+2)
        x < 1 && continue
        y < 1 && continue
        x > 84 && continue
        y > 84 && continue
        img[y,x] = yolo_color
    end
    
    img[1,39:45] .= bkg
    img[2,40:44] .= bkg
    img[3,41:43] .= bkg
    img[4,42:42] .= bkg

    # plot current steering
    steer_color = floor(UInt8, (0.05+pi/30 + car.steer)/(pi/15) * 200.0)
    for y in 1:84
        img[y,1:4] .= steer_color
        img[y,5] = 228
        img[y,81:84] .= steer_color
        img[y,80] = 228
    end
    
    return img
end

function sim!(env::MyEnv, state::MyState, action::Tuple{Float64,Float64}, stepc=5, safety=SAFETY)
    reward = 0.0
    done = false
    steer, accel = action
    av = getcar(state)
    yolocar_dist::Float32 = state.yolocar_dist
    yolocar_timestep::Int32 = state.yolocar_timestep
    times_passed::Int32 = state.times_passed_yolocar
    dt = 0.1
    for j in 1:stepc
        # moving the yolo car!
        yolocar_timestep += 1
        yolocar_dist += dt*env.yolocar_vels[yolocar_timestep]

        av = move(av,dt,steer,accel)

        # check if we passed the yolo car
        # (if yes, we should spawn a new car)
        c = cos(-av.angle+pi/2)
        s = sin(-av.angle+pi/2)    
        xc = env.splx(state.yolocar_dist)
        yc = env.sply(state.yolocar_dist)
        xc -= av.cx
        yc -= av.cy
        if xc*xc + yc*yc < 100.0*100.0
            # we are cloose to the car, passing it was possible within a short timescale
            xc,yc = xc*c-yc*s, xc*s+yc*c
            xc = floor(Int64, xc/DRAWING_SCALE) + 42
            yc = floor(Int64, yc/DRAWING_SCALE)
            if yc < 2
                # passed, will no be drawn on screen
                yolocar_dist += 70 + randn()*7
                times_passed += 1
            end
        end

        # calculating reward
        yolo_x::Float32 = env.splx(yolocar_dist)
        yolo_y::Float32 = env.sply(yolocar_dist)
        imre, done = calc_reward(av, env.coll, yolo_x, yolo_y, safety)
        reward += imre
        done && break
    end
    if done
        reward = 0.0
    end
    state = MyState(av,yolocar_dist,yolocar_timestep, times_passed)
    return state, reward, done
end







