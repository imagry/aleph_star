# see good tutorial here: http://ksimek.github.io/2013/08/13/intrinsic/
# the camera here is initialized to look towards -z, and up is y, just like the default Blender camera

function _getIntrinsic(FOVH)
    # focal length, asumming some detector size which is irrelevant
    fl = 1.0/tan(FOVH/2.0)
    [fl   0     0   0;
      0  fl     0   0;
      0   0     1   0;
      0   0     0   1]
end

function _getExtrinsic(ROT_XYZ, TRANS_XYZ)
    R = (rotXYZ(ROT_XYZ) * rotXYZ([pi,0.0,pi]))'
    ex = zeros(4,4)
    ex[1:3,1:3] = R
    ex[1:3,4] = -R * TRANS_XYZ
    ex[4,4] = 1.0
    ex
end

function rotX(theta)
    ct = cos(theta)
    st = sin(theta)
    [1   0    0;
     0   ct  -st;
     0   st   ct]
end

function rotY(theta)
    ct = cos(theta)
    st = sin(theta)
    [ ct   0    st;
      0    1     0;
     -st   0    ct]
end

function rotZ(theta)
    ct = cos(theta)
    st = sin(theta)
    [ct   -st    0;
     st    ct    0;
     0     0     1]
end

rotXYZ(XYZ::Vector{Float64}) = rotZ(XYZ[3]) * rotY(XYZ[2]) * rotX(XYZ[1])

Camera(ROT_XYZ::Vector{Float64}, TRANS_XYZ::Vector{Float64}, FOVH::Float64) =
    _getIntrinsic(FOVH) * _getExtrinsic(ROT_XYZ, TRANS_XYZ)

@inline function xyz_to_image_coords(CAM::Array{Float64,2}, P_XYZ::Vector{Float64}, image_width::Int64, image_height::Int64)
    i = CAM * [P_XYZ..., 1.0]
    imx = i[1] / i[3] * image_width / 2+image_width/2
    imy = i[2] / i[3] * image_height / 2*image_width/image_height+image_height/2
    image_width-imx, image_height-imy, i[3]
end

function line_to_2d(CAM, xl,yl,W=256,H=192)
    x2d_l = zeros(length(xl))
    y2d_l = zeros(length(yl))
    z2d_l = zeros(length(yl))
    for i in eachindex(xl)
        if isnan(xl[i])
            x2d_l[i] = NaN
            y2d_l[i] = NaN
            z2d_l[i] = NaN
            continue
        end
        v = xyz_to_image_coords(CAM, [xl[i],yl[i],0.0], W, H)
        x2d_l[i] = v[1]
        y2d_l[i] = v[2]
        z2d_l[i] = v[3]
    end
    ixs = find((z2d_l.>0.0) .| (isnan.(z2d_l)))
    x2d_l[ixs],y2d_l[ixs]
end

