# TODO: make a nice package
# some docs

using Nullables
import Statistics: mean, std
import DataStructures: heappop!, heappush!
import Base: isless, push!

const HEAP_GC_FAC = 0.2 # fraction of used cells before gc'ing the heap

include("data_structures.jl")
include("heap.jl")
include("tree.jl")
