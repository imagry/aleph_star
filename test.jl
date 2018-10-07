function test()
    # some testing
    stepc = 5500
    epsilon = 0.4
    gamma = 0.9
    w = map(KnetArray, initialize_weights(actionc_accel*actionc_steer))
    state, env = initialize_simple_road();
    tree = build_tree(w, env, state, stepc, epsilon,gamma);
    backprop_weighted_q!(tree, gamma)
    @test (stepc+1)==length(tree.nodes)==length(tree.sensors)==length(tree.children_qs)==length(tree.states)==length(tree.accumulated_rewards)==length(tree.dones)
    @test is_root(tree.nodes[1])
    @test is_leaf(tree.nodes[end])
    visitedc = zeros(Int32, length(tree.nodes))
    for nix in length(tree.nodes):-1:1
        node = tree.nodes[nix]
        for ch in values(node.children)
            visitedc[nix] += visitedc[ch.id]
        end
        visitedc[nix] += 1
    end
    @test visitedc[1] == stepc+1
    for (ix,n) in enumerate(tree.nodes)
        tree.dones[n.id] && @test length(n.children)==0 || all_children_done(tree, n)
        if !is_leaf(n)
            @test visitedc[n.id] == 1+sum([visitedc[c.id] for c in values(n.children)])
        end
        if is_leaf(n)
            @test visitedc[n.id] == 1
            tree.dones[n.id] && @test abs(sum(tree.children_qs[n.id])) < 7.0e-5        
            parent = get(n.parent)
            mn = Float32(mean(tree.children_qs[n.id]))
            reward = tree.accumulated_rewards[n.id] - tree.accumulated_rewards[parent.id]
            q = reward + gamma*mn
            @test abs(tree.children_qs[parent.id][n.action_ix] - q) < 7.0e-5
        elseif !is_root(n) 
            parent = get(n.parent)
            q::Float32 = 0.0
            for (i,c) in n.children
                q += visitedc[c.id] * tree.children_qs[n.id][i]
            end
            q /= visitedc[n.id]-1
            q *= gamma
            reward = tree.accumulated_rewards[n.id] - tree.accumulated_rewards[parent.id]
            q += reward
            @test abs(tree.children_qs[parent.id][n.action_ix] - q) < 7.0e-5        
        end
    end
    println("[V] all tests passed!")
end
