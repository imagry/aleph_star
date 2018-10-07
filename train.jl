function runningmean(qs)
    t = 0.0
    n = 0
    for q in qs
        t += sum(q)
        n += length(q)
    end
    t/n
end

function runninstd(qs, mn)
    t = 0.0
    n = 0
    for q in qs
        for qi in q
            t += (qi - mn).^2
            n += 1
        end
    end
    t/n
end

