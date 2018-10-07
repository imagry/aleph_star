
mutable struct DemoTrainingDQN
    # network weights
    w::Vector{Array{Float32,N} where N}
    
    # tree params
    max_stepc::Int64
    epsilon::Float64
    MIN_EPSILON::Float64
    fac::Float64
    gamma::Float64

    # training params
    LR::Float64
    batchsize::Int64
    train_epochs::Int64
    max_states::Int64
    train_when_min::Int64

    # for validation
    val_safety::Float64

    # experience buffer
    sensors::Vector{Array{UInt8,2}}
    qs::Vector{Vector{Float32}}
    priorities::Vector{Float64}
    
    # for stats
    rewards::Vector{Float64}
    ranks::Vector{Int64}
    stepscum::Vector{Int64}
    episode_mean_speed::Vector{Float64}
    episode_std_speed::Vector{Float64}
    val_mean_speed::Vector{Float64}
    val_std_speed::Vector{Float64}
    val_steps::Vector{Float64}
    val_rewards::Vector{Float64}
    val_stepscum::Vector{Int64}
    avg_window::Int64
    _stepscum::Int64
    iter_of_training_start::Int64
    epsilon0::Float64
    function DemoTrainingDQN()
        new(
            Vector{Array{Float32,N} where N}(),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            Vector{Array{UInt8,2}}(), Vector{Vector{Float32}}(),
            Vector{Float64}(), Vector{Float64}(),
            Vector{Int64}(), Vector{Int64}(),
            Vector{Float64}(), Vector{Float64}(),
            Vector{Float64}(), Vector{Float64}(),
            Vector{Float64}(), Vector{Float64}(),
            Vector{Int64}(),
            0, 0, 0, 0,
        )
    end
end

function InitializeDTDQN()
    dtdqn = DemoTrainingDQN()
    dtdqn.w = initialize_weights(actionc_steer*actionc_accel)
    dtdqn.max_stepc = 5500
    dtdqn.epsilon = 0.5
    dtdqn.MIN_EPSILON = 0.01
    dtdqn.fac = 0.995
    dtdqn.gamma = 0.98
    dtdqn.LR = 0.01
    dtdqn.batchsize = 64
    dtdqn.train_epochs = 20
    dtdqn.max_states = 80000
    dtdqn.train_when_min = 50000
    dtdqn.val_safety = 1.0
    dtdqn.avg_window = 50
    dtdqn._stepscum = 0
    dtdqn.iter_of_training_start = -1
    dtdqn.epsilon0 = dtdqn.epsilon
    dtdqn
end

function traindtdqn(dtdqn::DemoTrainingDQN, iters::Integer, fname)

    w = map(KnetArray, dtdqn.w)
    for i in 1:iters
        #### Accumulating episode ###################################################
        
        println("--------------------------------------------------------------- i="*string(i))
        flush(stdout)
        
        added_states = 0
        episodsc = 0
        tot_episode_len = 0
        tot_passed = 0
        while true
            # accumulate episodes
            _speeds = Float64[]
            _sensors = Array{UInt8,2}[]
            _qs = Vector{Float32}[]        
            _rewards = Float64[]
            rank = 0
                    
            episodsc += 1
            state, env = initialize_simple_road();
            sensor = get_sensors(env, state)
            reward = 0.0f0
            
            while true
                # single episode loop
                dtdqn._stepscum += 1
                rank += 1
                
                sensor = get_sensors(env, state)
                q = network_predict(env, w, sensor)
                
                push!(_speeds, state.speed)
                push!(_sensors, sensor)
                push!(_qs, q)
                push!(_rewards, reward)
                
                action_ix = argmax(q)
                action = action_ix_to_action(env, Int32(action_ix))
                state, reward, done = sim!(env, state, action, 5, dtdqn.val_safety)
                if done
                    sensor = get_sensors(env, state)
                    q = network_predict(env, w, sensor)
                    q .= 0.0f0

                    push!(_speeds, state.speed)
                    push!(_sensors, sensor)
                    push!(_qs, q)
                    push!(_rewards, reward)
                end    
                done && break
                rank > dtdqn.max_stepc && break
            end
            # n-step backprop
            for i in length(_qs):-1:2
                _qs[i-1][argmax(_qs[i-1])] = _rewards[i] + dtdqn.gamma*maximum(_qs[i])
            end
            append!(dtdqn.qs, _qs)
            append!(dtdqn.sensors, _sensors)
            f = if length(dtdqn.priorities)==0
                1.0
            else 
                median(dtdqn.priorities)
            end
            append!(dtdqn.priorities, f.*ones(Float32, length(_qs)))
            
            push!(dtdqn.stepscum, dtdqn._stepscum)
            push!(dtdqn.ranks, rank)
            push!(dtdqn.episode_mean_speed, mean(_speeds))
            push!(dtdqn.episode_std_speed, std(_speeds))
            push!(dtdqn.rewards, sum(_rewards))
            if length(dtdqn.qs) > dtdqn.max_states
                dtdqn.qs = dtdqn.qs[(end-dtdqn.max_states+1):end]
                dtdqn.sensors = dtdqn.sensors[(end-dtdqn.max_states+1):end]
                dtdqn.priorities = dtdqn.priorities[(end-dtdqn.max_states+1):end]
            end
            added_states += length(_qs)
            added_states >= dtdqn.max_stepc-1 && break
            tot_episode_len += rank
            tot_passed += state.times_passed_yolocar
            print("."); flush(stdout)
        end
        mean_episode_len = tot_episode_len / episodsc
        mean_passed = tot_passed / episodsc
        
        #### Training #######################################################
        
        if length(dtdqn.qs) >= dtdqn.train_when_min
            if dtdqn.iter_of_training_start<0
                dtdqn.iter_of_training_start = div(dtdqn._stepscum, dtdqn.max_stepc)
            end
            dtdqn.epsilon = max(dtdqn.MIN_EPSILON, dtdqn.epsilon0 * dtdqn.fac^(div(dtdqn._stepscum, dtdqn.max_stepc)-dtdqn.iter_of_training_start))
            N = round(Int64, added_states * dtdqn.train_epochs / dtdqn.batchsize)
            println("\n---- training for "*string(N)*" iterations"); flush(stdout)
            w,_ = trainit(N, w, dtdqn.sensors,dtdqn.qs,dtdqn.LR,dtdqn.batchsize, dtdqn.priorities)
            dtdqn.w = map(Array, w)
        end
    
        #### Validating #####################################################
        
        valv = Float64[]
        valr = 0.0
        spd = mean(dtdqn.episode_mean_speed[max(1,end-dtdqn.avg_window):end])
        if isnan(spd)
            spd = 0.8
        end
        state, val_env = initialize_simple_road(spd);
        for _ in 1:dtdqn.max_stepc
            push!(valv, state.speed)
            _sensors = get_sensors(val_env, state)
            _vqs = network_predict(val_env, w, _sensors)
            action = action_ix_to_action(val_env,Int32(argmax(_vqs)))
            state, reward, done = sim!(val_env, state, action, 5, dtdqn.val_safety)
            valr += reward
            done && break
        end
        vpassed = state.times_passed_yolocar
        push!(dtdqn.val_rewards, valr)
        push!(dtdqn.val_mean_speed, mean(valv))
        push!(dtdqn.val_std_speed, std(valv))
        push!(dtdqn.val_steps, length(valv))
        push!(dtdqn.val_stepscum, dtdqn._stepscum)
            
        #### Reporting #####################################################
        
        println()
        @show length(dtdqn.qs), dtdqn.rewards[end], div(dtdqn._stepscum, dtdqn.max_stepc)
        println("         eps = "*string(round(dtdqn.epsilon, digits=4))*"    gamma ="*string(round(dtdqn.gamma, digits=4)))
        println("         rnk = "*string(mean_episode_len)*             "      mvel = "*string(round(dtdqn.episode_mean_speed[end], digits=4))*"    svel = "*string(round(dtdqn.episode_std_speed[end], digits=4)))
        println("        vrnk = "*string(dtdqn.val_steps[end])*         "     vmvel = "*string(round(dtdqn.val_mean_speed[end], digits=4))*"   vsvel = "*string(round(dtdqn.val_std_speed[end], digits=4)))
        println(" mean_passed = "*string(floor(Int64, mean_passed))*    "   vpassed = "*string(vpassed))
        flush(stdout)
                            
        if i%10 == 0
            IJulia.clear_output();
        end

        if i%100 == 0
            @save fname dtdqn
        end
                            
    end
end

function plotdtdqn(dtdqn)
    p1 = plot(dtdqn.stepscum/5500, moving_avg(dtdqn.rewards, dtdqn.avg_window), label="train reward")
    plot!(p1, dtdqn.val_stepscum/5500, moving_avg(dtdqn.val_rewards, dtdqn.avg_window), lw=3, label="val reward")
    
    p2 = plot(dtdqn.stepscum/5500, moving_avg(dtdqn.ranks/dtdqn.max_stepc, dtdqn.avg_window), label="train rank (perc.)")
    plot!(p2, dtdqn.val_stepscum/5500, moving_avg(dtdqn.val_steps/dtdqn.max_stepc, dtdqn.avg_window), lw=3, label="max tree rank (perc.)")
    
    p3 = plot(dtdqn.stepscum/5500, moving_avg(dtdqn.episode_mean_speed, dtdqn.avg_window), label="train mean speed")
    plot!(p3, dtdqn.val_stepscum/5500, moving_avg(dtdqn.val_mean_speed, dtdqn.avg_window), lw=3, label="val mean speed")
    
    plot(p1,p2,p3, layout=(3,1), size=(900,1100))
end
