local argcheck = require 'argcheck'

-- display source, target, hypothesis and attention
local displayResults = argcheck{
    {name='srcdict', type='Dictionary'},
    {name='dict', type='Dictionary'},
    {name='nbest', type='number'},
    {name='beam', type='number'},
    call = function(dict, srcdict, nbest, beam)
        local eos = dict:getSymbol(dict:getEosIndex())
        local unk = dict:getSymbol(dict:getUnkIndex())
        local seos = srcdict:getSymbol(srcdict:getEosIndex())
        local runk = unk
        repeat -- select unk token for reference different from hypothesis
            runk = string.format('<%s>', runk)
        until dict:getIndex(runk) == dict:getUnkIndex()

        return function(sample, hypos, scores, attns)
            local src, tgt = sample.source:t(), sample.target:t()
            for i = 1, sample.bsz do
                local sourceString = srcdict:getString(src[i]):gsub(seos, '')
                print('S-' .. sample.index[i], sourceString)

                local ref = dict:getString(tgt[i])
                    :gsub(eos .. '.*', ''):gsub(unk, runk) --ref may contain pad
                print('T-' .. sample.index[i], ref)

                for j = 1, math.min(nbest, beam) do
                    local idx = (i - 1) * beam + j
                    local hypo = dict:getString(hypos[idx]):gsub(eos, '')
                    print('H-' .. sample.index[i], scores[idx], hypo)
                    -- NOTE: This will print #hypo + 1 attention maxima. The
                    -- last one is the attention that was used to generate the
                    -- <eos> symbol.
                    local _, maxattns = torch.max(attns[idx], 2)
                    print('A-' .. sample.index[i],
                    table.concat(maxattns:squeeze(2):totable(), ' '))
                end
            end
        end
    end
}

return {
    displayResults = displayResults,
}
