import Base:length

length(heap::Heap) = length(heap.cells) - heap.total_used

isless(hc1::HeapCell, hc2::HeapCell)  =
    isless(hc2.score, hc1.score) # because we want maximum and not minimum

function push!(heap::Heap, action_ix::Int32, parent_id::Int32, score::Float32)
    heap_cell = HeapCell(
        false, # not used
        score,
        action_ix,
        parent_id,
    )
    heappush!(heap.cells, heap_cell)
end

function pop_max!(heap::Heap)
    length(heap) == 0 && error("empty heap in pop_max!")
    if heap.total_used / length(heap) > HEAP_GC_FAC
        garbage_collect!(heap)
    end
    while true
        hc = heappop!(heap.cells)
        if hc.is_used
            heap.total_used -= 1
            continue
        end
        return hc.action_ix, hc.parent_id
    end
end

function pop_rand!(heap::Heap) 
    length(heap) == 0 && error("empty heap in pop_rand!")
    if heap.total_used / length(heap) > HEAP_GC_FAC
        garbage_collect!(heap)
    end
    while true
        ix = rand(1:length(heap.cells))
        hc = heap.cells[ix]
        hc.is_used && continue
        # mark as used:
        heap.cells[ix] = HeapCell(
            true,
            hc.score,
            hc.action_ix,
            hc.parent_id,
        )
        heap.total_used += 1
        return hc.action_ix, hc.parent_id
    end
end

function garbage_collect!(heap::Heap)
    tmp = HeapCell[]
    for hc in heap.cells
        hc.is_used && continue
        push!(tmp, hc)
    end
    empty!(heap.cells)
    for hc in tmp
        heappush!(heap.cells, hc)
    end
    heap.total_used = 0
end

