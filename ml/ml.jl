using JLD2
using Knet
using Distributions

const EPSILON = 1.0f-5

function normit(x)
    d = ndims(x) == 4 ? [1,2,3] : [1,]
    s = prod(size(x)[d])
    mu = sum(x, dims=d) ./ s
    x = x .- mu
    sigma = sqrt.(EPSILON .+ (sum(x .* x, dims=d)) ./ s)
    return x ./ sigma
end

function predict(w,x)
    x = conv4(w[1],x; stride=4) .+ w[2]
    x = normit(x)
    x = max.(0.3f0.*x,x)
    x = conv4(w[3],x; stride=2) .+ w[4]
    x = normit(x)
    x = max.(0.3f0.*x,x)
    x = mat(x)    
    x = w[5]*x .+ w[6]
    x = normit(x)
    x = max.(0.3f0.*x,x)
    return abs.(w[7]*x .+ w[8])
end

function initialize_weights(number_of_actions)
    [
        0.01f0*randn(Float32, 8, 8, 1, 16),
        0.01f0*randn(Float32, 20, 20, 16),
        0.01f0*randn(Float32, 4,4,16,32),
        0.01f0*randn(Float32, 9,9,32),
        0.01f0*randn(Float32, 256,2592),
        0.01f0*randn(Float32, 256,1),
        0.01f0*randn(Float32, number_of_actions,256),
        0.01f0*randn(Float32, number_of_actions,1),
    ]
end

loss(w,x,y) = sum(abs2, (y.-predict(w,x))) ./ length(y)
lossgradient = gradloss(loss)

function train(w, x,y; lr=.0001f0)
    dw, ll = lossgradient(w, x, y)
    for i in 1:length(w)
        w[i] -= lr .* dw[i]
    end
    return w, ll
end

function trainit(N,w,sensors,mqs, lr, batchsize, priorities)
    h = sort(priorities)[floor(Int64, 0.9*length(priorities))]    
    _p = [max(h, p) for p in priorities]
    _at = Distributions.AliasTable(_p)
    tll = 0.0
    for j in 1:N
        if j % 100 == 0
            print('.'); flush(stdout)
        end
        x,y, bix = batchit(sensors, mqs, batchsize, _at);
        w,ll = train(w, x, y; lr=lr)
        priorities[bix] .= ll
        tll += ll
    end
    w, tll/N
end

function batchit(sensors, qs, batchsize, alias_table)
    @assert length(qs) > 0
    @assert length(sensors) == length(qs)
    NACT = length(qs[1])
    x = zeros(Float32, size(sensors[1])[1],size(sensors[1])[2],1,batchsize)
    y = zeros(Float32, NACT,batchsize)
    bix = rand(alias_table, batchsize)
    for (bnum, ix) in enumerate(bix)
        y[:,bnum] .= qs[ix]
        x[:,:,1,bnum] .= sensors[ix] ./ 255.0f0
    end
    KnetArray{Float32}(x), KnetArray{Float32}(y), bix
end

function savew(w)
    ww = [Array(wi) for wi in w]
    @save "w.jld2" ww
end

function loadw()
    @load "w.jld2" ww
    [KnetArray(wi) for wi in ww]
end

function network_predict(env::MyEnv, w, sensors)
    x = zeros(Float32, size(sensors)[1],size(sensors)[2],1,1)
    x[:,:,1,1] .= sensors./255.0f0
    v = Knet.Array(predict(w, Knet.KnetArray(x)))
    return reshape(v,length(v))
end

function moving_avg(y,w)
    ay = Float64[]
    for i in 1:length(y)
        push!(ay,mean(y[max(i-w,1):i]))
    end
    ay
end
