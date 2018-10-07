const G = 9.8
const CARMAXSTEER = pi/6

struct Car
    # coordinates
    rlx::Float32
    rly::Float32
    rrx::Float32
    rry::Float32
    flx::Float32
    fly::Float32
    frx::Float32
    fry::Float32
    minx::Float32
    miny::Float32
    maxx::Float32
    maxy::Float32
    cx::Float32
    cy::Float32
    fcx::Float32 # front center
    fcy::Float32
    rcx::Float32 # rear center
    rcy::Float32

    # physical properties
    full_steering_dt::Float32
    maxsteer::Float32
    wheelbase::Float32
    fronttrack::Float32
    steering_speed::Float32
    wheel_static_friction_coefficient::Float32
    
    # state (other than coordinates)
    angle::Float32
    speed::Float32
    steer::Float32
    accel::Float32

    function Car(rcx,rcy,angle,steer,accel,speed)
        full_steering_dt = 3.0
        maxsteer = CARMAXSTEER
        wheelbase = 2.7
        fronttrack = 1.65
        steering_speed = 2.0*maxsteer / full_steering_dt
        wheel_static_friction_coefficient = 0.7

        ca9 = cos(pi / 2 - angle)
        sa9 = sin(pi / 2 - angle)
        ca = cos(angle)
        sa = sin(angle)

        fa2 = fronttrack / 2.0

        rlx = rcx - fa2 * ca9
        rly = rcy + fa2 * sa9
        rrx = rcx + fa2 * ca9
        rry = rcy - fa2 * sa9

        fx = rcx + wheelbase * ca
        fy = rcy + wheelbase * sa

        flx = fx - fa2 * ca9
        fly = fy + fa2 * sa9
        frx = fx + fa2 * ca9
        fry = fy - fa2 * sa9

        minx = min(rlx, rrx, flx, frx)
        maxx = max(rlx, rrx, flx, frx)
        miny = min(rly, rry, fly, fry)
        maxy = max(rly, rry, fly, fry)

        new(
            rlx, rly, rrx, rry, flx, fly, frx, fry,
            minx,
            miny,
            maxx,
            maxy,
            0.5 * (minx + maxx),
            0.5 * (miny + maxy),
            0.5 * (flx + frx),
            0.5 * (fly + fry),
            rcx, rcy,

            full_steering_dt,
            maxsteer,
            wheelbase,
            fronttrack,
            steering_speed,
            wheel_static_friction_coefficient,

            angle, speed, steer, accel,
        )
    end
end

function get_steer(c::Car, Rw::Float64)
    Rr = Rw - c.fronttrack / 2.0
    steer = pi/2.0 - atan(Rr/c.wheelbase)
    return steer    
end

function get_max_safe_steer(c::Car)
    max_safe_steer = c.maxsteer
    if abs(c.speed) > 1.0e-2
        max_safe_steer = abs(atan(c.wheel_static_friction_coefficient*G*c.wheelbase/c.speed/c.speed))
    end
    return min(max_safe_steer, c.maxsteer)
end

function move(c::Car, dt::Float64, steering::Float64, accel::Float64)
    @assert dt > 1.0e-3

    maxsteer = get_max_safe_steer(c)
    if steering > maxsteer
        steering = maxsteer
    elseif steering < -maxsteer
        steering = -maxsteer
    end

    if steering - c.steer < -c.steering_speed*dt
        steering = c.steer - c.steering_speed*dt
    elseif steering - c.steer > c.steering_speed*dt
        steering = c.steer + c.steering_speed*dt
    end

    angle = c.angle
    rcx = c.rcx
    rcy = c.rcy

    arclen = c.speed * dt + 0.5 * accel * dt * dt
    if abs(steering) > 1.0e-4
        Rr = abs(c.wheelbase * tan(pi / 2.0 - steering))

        Rw = (Rr + c.fronttrack / 2.0)

        rx = c.rcx - Rw * cos(pi / 2.0 - c.angle) * sign(steering)
        ry = c.rcy + Rw * sin(pi / 2.0 - c.angle) * sign(steering)

        dangle = arclen / Rr * sign(steering)

        # translate to origin
        ox = c.rcx - rx
        oy = c.rcy - ry
        # rotate
        cda = cos(dangle)
        sda = sin(dangle)
        rcx = rx + ox * cda - oy * sda
        rcy = ry + ox * sda + oy * cda
        angle += dangle
    else
        # asumming steer of 0.0
        rcx += arclen * cos(c.angle)
        rcy += arclen * sin(c.angle)
    end

    speed = c.speed + accel * dt
    if angle < -pi
        angle += 2.0*pi
    elseif angle > pi
        angle -= 2.0*pi
    end

    Car(rcx,rcy,angle,steering,accel,speed)
end

function get_plot(c::Car)
    [c.flx,c.frx,c.rrx,c.rlx,c.flx],
    [c.fly,c.fry,c.rry,c.rly,c.fly]
end

getvxvy(c::Car) = 
    cos(c.angle)*c.speed, sin(c.angle)*c.speed



