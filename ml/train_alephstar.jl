mutable struct DemoTrainingAlephStar
    # network weights
    w::Vector{Array{Float32,N} where N}
    
    # tree params
    stepc::Int64
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
    num_of_done_leafs::Vector{Int64}
    num_of_leafs::Vector{Int64}
    num_of_car_crash::Vector{Int64}
    num_of_wall_crash::Vector{Int64}
    num_of_vel_crash::Vector{Int64}
    tree_mean_speed::Vector{Float64}
    tree_std_speed::Vector{Float64}
    val_mean_speed::Vector{Float64}
    val_std_speed::Vector{Float64}
    val_steps::Vector{Float64}
    val_rewards::Vector{Float64}
    avg_window::Int64
    weighted_nodes_threshold::Int64
    function DemoTrainingAlephStar()
        new(
            Vector{Array{Float32,N} where N}(),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            Vector{Array{UInt8,2}}(),
            Vector{Vector{Float32}}(),
            Vector{Float64}(),
            Vector{Float64}(), Vector{Int64}(),
            Vector{Int64}(), Vector{Int64}(),
            Vector{Int64}(), Vector{Int64}(),
            Vector{Int64}(), Vector{Float64}(),
            Vector{Float64}(), Vector{Float64}(),
            Vector{Float64}(), Vector{Float64}(),
            Vector{Float64}(), 0, 0,
        )
    end
end

function InitializeDTAS()
    dtas = DemoTrainingAlephStar()
    dtas.w = initialize_weights(actionc_steer*actionc_accel)
    dtas.stepc = 5500
    dtas.epsilon = 0.5
    dtas.MIN_EPSILON = 0.01
    dtas.fac = 0.995
    dtas.gamma = 0.98
    dtas.LR = 0.01
    dtas.batchsize = 64
    dtas.train_epochs = 20
    dtas.max_states = 80000
    dtas.train_when_min = 50000
    dtas.val_safety = 1.0
    dtas.avg_window = 50
    dtas.weighted_nodes_threshold = 200
    dtas
end

function traindtas(dtas::DemoTrainingAlephStar, iters::Integer, fname)
    w = map(KnetArray, dtas.w)
    for i in 1:iters
        #### Accumulating tree ###################################################
        
        println("--------------------------------------------------------------- i="*string(i))
        flush(stdout)
        
        state, env = initialize_simple_road();
        tree = build_tree(w, env, state, dtas.stepc, dtas.epsilon, dtas.gamma)
        backprop_weighted_q!(tree, dtas.gamma, dtas.weighted_nodes_threshold)
        
        # gather some stats
        push!(dtas.num_of_car_crash, 0)
        push!(dtas.num_of_wall_crash, 0)
        push!(dtas.num_of_vel_crash, 0)
        # some crash stats
        for node in tree.nodes
            !is_leaf(node) && continue
            sim_state = tree.states[node.id]
            yolo_x = env.splx(sim_state.yolocar_dist)
            yolo_y = env.sply(sim_state.yolocar_dist)    
            _, result = my_calc_reward(getcar(sim_state), env.coll, yolo_x, yolo_y, SAFETY)
            if result==1
                dtas.num_of_vel_crash[end] += 1
            elseif result==2
                dtas.num_of_wall_crash[end] += 1
            elseif result==3
                dtas.num_of_car_crash[end] += 1
            end
            @assert is_leaf(node)
            # this is possible due to floating point issues
            # result==0 && @assert !node.done
        end
        # calculate mean speed
        treev = Float64[]
        n = get_best_leaf(tree, dtas.gamma)
        times_passed = tree.states[n.id].times_passed_yolocar
        while !is_root(n)
            sim_state = tree.states[n.id]
            push!(treev, sim_state.speed)
            n = get(n.parent)
        end
        max_rank = get_tree_rank(tree, dtas.gamma)
        push!(dtas.ranks, max_rank)
        push!(dtas.tree_mean_speed, mean(treev))
        push!(dtas.tree_std_speed, std(treev))
        push!(dtas.rewards, get_accumulated_reward(tree))
        push!(dtas.num_of_done_leafs, sum([tree.dones[n.id] for n in tree.nodes if is_leaf(n)]))
        push!(dtas.num_of_leafs, sum([is_leaf(n) for n in tree.nodes]))
        # these could be slightly different due to floating point issues:
        # @assert num_of_car_crash[end] + num_of_vel_crash[end] + num_of_wall_crash[end] == num_of_done_leafs[end]
        
        ixs = findall([!is_leaf(n) || tree.dones[n.id] for n in tree.nodes])
        append!(dtas.qs, tree.children_qs[ixs])
        append!(dtas.sensors, tree.sensors[ixs])
        f = if length(dtas.priorities)==0
            1.0
        else 
            median(dtas.priorities)
        end
        append!(dtas.priorities, f.*ones(Float32, length(ixs)))
        if length(dtas.qs) > dtas.max_states
            dtas.qs = dtas.qs[(end-dtas.max_states+1):end]
            dtas.sensors = dtas.sensors[(end-dtas.max_states+1):end]
            dtas.priorities = dtas.priorities[(end-dtas.max_states+1):end]
        end
        
        #### Training #######################################################
                            
        if length(dtas.qs) >= dtas.train_when_min
            N = round(Int64, length(ixs) * dtas.train_epochs / dtas.batchsize)
            println("\n---- training for "*string(N)*" iterations"); flush(stdout)
            w,ll = trainit(N, w, dtas.sensors, dtas.qs, dtas.LR, dtas.batchsize, dtas.priorities)
            dtas.w = map(Array, w)
            if dtas.epsilon > dtas.MIN_EPSILON
                dtas.epsilon *= dtas.fac
            end
        end
      
        #### Validating #####################################################
        
        valv = Float64[]
        valr = 0.0
        spd = mean(dtas.tree_mean_speed[max(1,end-dtas.avg_window):end])
        if isnan(spd)
            spd = 0.8
        end
        state, val_env = initialize_simple_road(spd);
        for _ in 1:dtas.stepc
            push!(valv, state.speed)
            
            _sensors = get_sensors(val_env, state)
            _vqs = Vector{Float32}(network_predict(val_env, w, _sensors))
            action = action_ix_to_action(val_env,Int32(argmax(_vqs)))
            state, reward, done = sim!(val_env, state, action, 5, dtas.val_safety)
            valr += reward
            done && break
        end
        vpassed = state.times_passed_yolocar
        push!(dtas.val_rewards, valr)
        push!(dtas.val_mean_speed, mean(valv))
        push!(dtas.val_std_speed, std(valv))
        push!(dtas.val_steps, length(valv))
        
        #### Reporting #####################################################
        
        println()
        @show length(dtas.qs), length(ixs)
        println("      eps   = "*string(round(dtas.epsilon, digits=4))*"    gamma="*string(round(dtas.gamma, digits=4)))
        println(" tree_rnk   = "*string(max_rank)*    "    mvel    = "*string(round(mean(treev), digits=4))*"    svel = "*string(round(std(treev), digits=4)))
        println("val_steps   = "*string(length(valv))*"    mvel    = "*string(round(mean(valv),digits=4))*"    svel = "*string(round(std(valv),digits=4)))
        println("tree_passed = "*string(times_passed)*"    vpassed = "*string(vpassed))
        flush(stdout)
                            
        if i%10 == 0
            IJulia.clear_output();
        end
        if i%100 == 0
            @save fname dtas
        end
                            
    end
end

function plotdtas(dtas)
    p1 = plot(moving_avg(dtas.rewards, dtas.avg_window), label="max tree reward")
    plot!(p1, moving_avg(dtas.val_rewards, dtas.avg_window), lw=3, label="network rewards")

    p2 = plot(moving_avg(dtas.ranks/dtas.stepc, dtas.avg_window), label="max tree rank (perc.)")
    plot!(p2, moving_avg(dtas.val_steps/dtas.stepc, dtas.avg_window), lw=3, label="network rank")
    plot!(p2, moving_avg(dtas.num_of_done_leafs/dtas.stepc, dtas.avg_window), label="done perc.")
    plot!(p2, moving_avg(dtas.num_of_leafs/dtas.stepc, dtas.avg_window), label="leafs perc.")

    p3 = plot(moving_avg(dtas.num_of_vel_crash./dtas.num_of_done_leafs, dtas.avg_window), label="vel crash perc.", lw=3)
    plot!(p3, moving_avg(dtas.num_of_wall_crash./dtas.num_of_done_leafs, dtas.avg_window), label="wall crash perc.", lw=3)
    plot!(p3, moving_avg(dtas.num_of_car_crash./dtas.num_of_done_leafs, dtas.avg_window), label="car crash perc.", lw=3)

    p4 = plot(moving_avg(dtas.tree_mean_speed, dtas.avg_window), label="tree mean speed")
    plot!(p4, moving_avg(dtas.tree_std_speed, dtas.avg_window), label="tree std speed")
    plot!(p4, moving_avg(dtas.val_std_speed, dtas.avg_window), label="network std speed")
    plot!(p4, moving_avg(dtas.val_mean_speed, dtas.avg_window), label="network mean speed")

    plot(p1,p2,p3,p4, layout=(4,1), size=(900,1300))
end
