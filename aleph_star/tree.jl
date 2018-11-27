is_root(node) = isnull(node.parent)

is_leaf(node) = length(node.children) == 0

all_children_explored(node::Node{ACTIONC}) where ACTIONC =
    length(node.children)==ACTIONC

all_children_done(tree, node) =
    all_children_explored(node) && all(tree.dones[c.id] for c in values(node.children))

function get_rank(node)
    rank=1
    while true
        isnull(node.parent) && break
        node = get(node.parent)
        rank += 1
    end
    rank
end

get_best_leaf(tree, gamma) = tree.nodes[argmax(tree.accumulated_rewards + gamma*[maximum(qs) for qs in tree.children_qs])]
  
get_tree_rank(tree, gamma) = get_rank(get_best_leaf(tree, gamma))

get_accumulated_reward(tree) = maximum(tree.accumulated_rewards)

function calc_visitedc(tree, maxval=-1)
    # calculate number of times each node was visited
    # iterate in reverse, so parents have all children
    # already updated when we get to them
    visitedc = zeros(Int32, length(tree.nodes))
    for nix in length(tree.nodes):-1:1
        node = tree.nodes[nix]
        for ch in values(node.children)
            visitedc[nix] += visitedc[ch.id]
            if maxval > 0 && visitedc[nix] > maxval
                visitedc[nix] = maxval
            end
        end
        visitedc[nix] += 1
    end
    visitedc
end

function calc_action_visitedc(tree, visited_threshold=-1, nonexplored_value=0)
    visitedc = calc_visitedc(tree, visited_threshold)
    avc = Vector{Int32}[]
    for node in tree.nodes
        _avc = Int32(nonexplored_value)*ones(Int32, length(tree.children_qs[1]))
        for (ix,ch) in node.children
            _avc[ix] = visitedc[ch.id]
        end
        push!(avc, _avc)
    end
    avc
end

function backprop_weighted_q!(tree, gamma, visited_threshold=-1)
    avisitedc = calc_action_visitedc(tree, visited_threshold)
    # calculate weighted Qs, root is done separately
    # iterae in reverse, so parents have all children
    # already updated when we get to them
    for nix in length(tree.nodes):-1:2
        node = tree.nodes[nix]
        tree.dones[node.id] = all_children_done(tree, node)
        # update Q at parent
        parent = get(node.parent)
        node_reward = tree.accumulated_rewards[node.id] - tree.accumulated_rewards[parent.id]
        mn = if is_leaf(node)
            mean(tree.children_qs[nix])
        else
            sum(avisitedc[nix] .* tree.children_qs[nix]) ./ sum(avisitedc[nix])
        end
        tree.children_qs[parent.id][node.action_ix] = node_reward + gamma*mn
    end
    # update root
    tree.dones[1] = all_children_done(tree, tree.nodes[1])
end

function backprop_max_q!(tree, gamma)
    # backprop everybody in reverse, so parents have all children
    # already updated when we get to them
    for nix in length(tree.nodes):-1:2
        node = tree.nodes[nix]
        parent = get(node.parent)
        tree.dones[node.id] = all_children_done(tree, node)        
        # update Q at parent
        mx::Float32 = maximum(tree.children_qs[node.id])
        node_reward = tree.accumulated_rewards[node.id] - tree.accumulated_rewards[parent.id]
        tree.children_qs[parent.id][node.action_ix] = node_reward + gamma*mx
    end
    # update root
    tree.dones[1] = all_children_done(tree, tree.nodes[1])
end

function expand!(heap, tree, parent_node::Node{ACTIONC}, w, action_ix, env, gamma) where ACTIONC
    actual_action = action_ix_to_action(env, action_ix)
    new_sim_state, reward, done = sim!(env, tree.states[parent_node.id], actual_action)
    new_sensors = get_sensors(env, new_sim_state)
    children_qs = if done
        zeros(Float32, ACTIONC)
    else
        network_predict(env, w,new_sensors)
    end
    id::Int32 = length(tree.states) + 1
    new_node = Node(Int32(action_ix), Nullable(parent_node), id)
    parent_node.children[action_ix] = new_node
    accumulated_reward::Float32 = tree.accumulated_rewards[parent_node.id] + Float32(reward)
    push!(tree.nodes, new_node)
    push!(tree.sensors, new_sensors)
    push!(tree.children_qs, children_qs)
    push!(tree.states, new_sim_state)
    push!(tree.accumulated_rewards, accumulated_reward)
    push!(tree.dones, done)
    if !done
        for (aix, q) in enumerate(children_qs)
            score::Float32 = accumulated_reward + Float32(gamma)*q
            push!(heap, Int32(aix), new_node.id, score)
        end
    end
end

function build_tree(w, env, root_state, stepc, epsilon, gamma)
    # build the root
    root_sensors = get_sensors(env, root_state)
    root_children_qs = network_predict(env, w, root_sensors)
    ACTIONC = length(root_children_qs)
    null_parent = Nullable{Node{ACTIONC}}()
    root_action_ix::Int32 = -1 # nonsensical action
    root_id::Int32 = 1
    root_accumulated_reward::Float32 = 0.0f0
    root_done::Bool = false
    root = Node(root_action_ix, null_parent, root_id)

    # build the tree
    nodes = Node{ACTIONC}[]
    sensors = typeof(root_sensors)[]
    children_qs = Vector{Float32}[]
    states = typeof(root_state)[]
    accumulated_rewards = Float32[]
    dones = Bool[]
    heap = Heap()
    tree = Tree(env, heap, nodes, sensors, children_qs, states, accumulated_rewards, dones)

    # update the tree with root
    push!(tree.nodes, root)
    push!(tree.sensors, root_sensors)
    push!(tree.children_qs, root_children_qs)
    push!(tree.states, root_state)
    push!(tree.accumulated_rewards, root_accumulated_reward)
    push!(tree.dones, root_done)
    for (aix, q) in enumerate(tree.children_qs[1])
        score::Float32 = Float32(gamma)*q
        push!(tree.heap, Int32(aix), root.id, score)
    end

    # add new nodes in a loop
    for i in 1:stepc
        if i % 200 == 0
            print(".")
            flush(stdout)
        end

        # choose parent and action
        tree.dones[1] && return tree # root is done
        length(heap) == 0 && return tree
        action_ix, parent_id = if rand()>epsilon
            pop_max!(heap)
        else
            pop_rand!(heap)
        end
        @assert !tree.dones[parent_id]
        parent = tree.nodes[parent_id]
        @assert !haskey(parent.children, action_ix)

        # add a new node to the tree
        expand!(heap, tree, parent, w, action_ix, env, gamma)
    end
    println()
    return tree
end



