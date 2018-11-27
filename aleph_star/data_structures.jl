# a Node contains the minimum needed to hold the tree structure
struct Node{ACTIONC}
    action_ix::Int32 # action index leading to this state
    parent::Nullable{Node{ACTIONC}}
    children::Dict{Int32, Node{ACTIONC}}
    id::Int32
    function Node(
                  action_ix::Int32,
                  parent::Nullable{Node{ACTIONC}},
                  id::Int32) where ACTIONC
        children = Dict{Int32, Node{ACTIONC}}()
        return new{ACTIONC}(action_ix, parent, children, id)
    end
end

struct HeapCell
    is_used::Bool # because we cannot efficiently pop a random element from a heap
    score::Float32
    action_ix::Int32
    parent_id::Int32
end

mutable struct Heap
    cells::Vector{HeapCell}
    total_used::Int64
    Heap() = new(Vector{HeapCell}(), 0)
end

# The tree contains the nodes (describing the structure)
# and any additional data per node
struct Tree{STATE, SENSOR, ENV, ACTIONC}
    env::ENV
    heap::Heap
    # these 6 vectors are indexed by node.id (SOA style)
    nodes::Vector{Node{ACTIONC}}
    sensors::Vector{SENSOR}
    children_qs::Vector{Vector{Float32}}
    states::Vector{STATE}
    accumulated_rewards::Vector{Float32}
    dones::Vector{Bool}
end



