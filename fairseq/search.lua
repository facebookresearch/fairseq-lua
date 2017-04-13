-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Common search functions used for sentence generation. A search function is
-- expected to return a table containing the following functions, all operating
-- on mini-batches:
--   - `init(bsz, source)` will be called prior to generation
--   - `prune(step, ldist)` is called at every generation step with the model
--      output, which are scores for all words in the dictionary after
--      LogSoftMax. This function
--      is expected to return a table with the following entries:
--      - `nextIn`: model input for the next generation step (dictionary
--        indices).
--      - `nextHid`: index for hidden state propagation
--      - `eos`: boolean flag indicating if generation should be stopped.
--      Except for `eos`, these are all tensors of size (bsz * beam).
--  - `results()`: returns three items: a (bsz * beam) table of hypotheses, each
--     being a tensor with word indices, a (bsz * beam) tensor with
--     corresponding scores (e.g. average log-probabilities), and a
--     (bsz * beam) table of attention scores, each being a 2D tensor
--     with (targetlength X sourcelength) entries.
--
-- The methods above will be called from a model's generate() function.
--
--]]

local visdom = require 'visdom'
local argcheck = require 'argcheck'
local plstringx = require 'pl.stringx'
local tds = require 'tds'
local mutils = require 'fairseq.models.utils'
local clib = require 'fairseq.clib'

local search = {}

function search.greedy(ttype, dict, maxlen)
    -- Greedy search: at each step, select the symbol with the highest
    -- log-probabilities.
    local logprobs = torch.Tensor():type(ttype)
    local lengths = torch.Tensor():type(ttype)
    local notEos = torch.Tensor():type(ttype)
    local nextHid = torch.LongTensor()
    local outs = torch.Tensor():type(ttype)
    local ascores = {}
    local sourcelen = nil
    local f = {}

    f.init = function(bsz, sample)
        logprobs:resize(bsz):fill(0)
        lengths:resize(bsz):fill(0)
        notEos:resize(bsz):fill(1)
        outs:resize(maxlen + 1, bsz)
        ascores = {}
        sourcelen = sample.source:size(1)
        -- Propagation of hidden states is fixed since there's only one active
        -- hypothesis per sentence.
        nextHid:resize(bsz):copy(torch.range(1, bsz))
    end

    local maxScores = torch.Tensor():type(ttype)
    local maxIndices, isEos
    if not ttype:match('torch.Cuda.*') then
        maxIndices = torch.LongTensor()
        isEos = torch.ByteTensor()
    else
        maxIndices = torch.CudaLongTensor()
        isEos = torch.CudaByteTensor()
    end
    f.prune = function(step, ldist, attnscores)
        maxScores, maxIndices = torch.topk(maxScores, maxIndices, ldist, 1, 2,
            true)
        local maxScoresV = maxScores:view(-1)
        local maxIndicesV = maxIndices:view(-1)

        logprobs:add(torch.cmul(notEos, maxScoresV))
        lengths:add(notEos)

        isEos = maxIndicesV.eq(isEos, maxIndicesV, dict:getEosIndex())
        notEos:maskedFill(isEos, 0)

        outs:narrow(1, step, 1):copy(maxIndices)
        local as = torch.FloatTensor(attnscores:size()):copy(attnscores)
        table.insert(ascores, as)
        return {
            nextIn = maxIndices,
            nextHid = nextHid,
            eos = notEos:sum() < 1,
        }
    end

    f.results = function()
        local bsz = logprobs:nElement()
        local hypotheses = {}
        local attentions = {}
        for i = 1, bsz do
            hypotheses[i] = torch.LongTensor(lengths[i])
            hypotheses[i]:copy(outs:sub(1, lengths[i], i, i))
            attentions[i] = torch.FloatTensor(lengths[i], sourcelen)
            for j = 1, lengths[i] do
                attentions[i]:narrow(1, j, 1):copy(ascores[j]:narrow(1, i, 1))
            end
        end
        return hypotheses, torch.cdiv(logprobs, lengths), attentions
    end

    return f
end

search.beam = argcheck{
    {name='ttype', type='string'},
    {name='dict', type='Dictionary'},
    {name='srcdict', type='Dictionary'},
    {name='beam', type='number'},
    {name='lenPenalty', type='number', default=1.0},
    {name='unkPenalty', type='number', default=0},
    {name='subwordPenalty', type='number', default=0},
    {name='coveragePenalty', type='number', default=0},
    {name='vocab', type='tds.Hash', opt=true},
    {name='subwordContSuffix', type='string', default='|'},
    call = function(ttype, dict, srcdict, beam, lenPenalty,
        unkPenalty, subwordPenalty, coveragePenalty, vocab, subwordContSuffix)
    -- Beam search: Keep track of `beam` hypotheses per sentence, move
    -- finished hypothesis (EOS symol) out of the beam (`finalized` table)
    -- and stop once there are `beam` finished hypotheses per sentence.
    local hidOffsets = nil
    local hscores = torch.Tensor():type(ttype)
    local ones = torch.Tensor(dict:size()):type(ttype):fill(1)
    local notEos = 0
    -- Previous outputs and attention distributions
    local outs = {}
    local ascores = {}
    -- Backpointers
    local backp = {}
    local finalized = {}
    local sourcelen = nil
    local f = {}

    local swcLen = subwordContSuffix:len()

    -- The penalty tensor is added to the log-probs produced by the model.
    local penalties = torch.Tensor(dict:size()):type(ttype):fill(0)
    penalties[dict:getEosIndex()] = 0
    penalties[dict:getUnkIndex()] = -unkPenalty
    for i = 1, dict:size() do
        if dict:getSymbol(i):sub(-swcLen) == subwordContSuffix then
            penalties[i] = -subwordPenalty
        end
    end
    local bpenalties = nil
    local srcVocab = {}

    f.init = function(bsz, sample)
        hscores:resize(bsz * beam):fill(0)
        -- e.g. for beam = 4: 111155559999...
        hidOffsets = torch.range(0, (bsz * beam) - 1)
            :long():div(beam):mul(beam) + 1
        outs, ascores = {}, {}
        backp, finalized = {}, {}, {}
        for i = 1, bsz do finalized[i] = {} end
        notEos = bsz
        sourcelen = sample.source:size(1)
        bpenalties = torch.expand(penalties:view(1, -1), bsz * beam,
            dict:size())

        if vocab then
            -- Add source sentence words to the vocabulary
            local sourceT = sample.source:t()
            for i = 1, sourceT:size(1) do
                srcVocab[i] = tds.Hash()
                -- narrow() is used to skip the end-of-sentence token.
                local str = srcdict:getString(
                    sourceT[i]:narrow(1, 1, sourceT[i]:size(1) - 1)
                )
                -- TODO(jgehring) Implement a dictionary for sub-words that
                -- takes care of proper string assembly.
                str = str:gsub(subwordContSuffix .. ' ', '')
                for _, w in ipairs(plstringx.split(str)) do
                    -- Add this word and all all prefixes
                    for j = 1, w:len() do
                        srcVocab[i][w:sub(1, j)] = 1
                    end
                end
            end
        end
    end

    local function backtraceWord(step, bp, cur)
        local w = dict:getSymbol(cur)
        if w:sub(-swcLen) == subwordContSuffix then
            w = w:sub(1, -swcLen - 1)
        end
        for l = step - 1, 1, -1 do
            local prev = dict:getSymbol(outs[l][bp])
            if prev:sub(-swcLen) ~= subwordContSuffix then
                break
            end
            w = prev:sub(1, -swcLen - 1) .. w
            bp = backp[l][bp]
        end
        return w
    end

    local function coverageP(attn)
        -- Coverage penalty according to https://arxiv.org/abs/1609.08144
        -- attn is a hypo X source tensor.
        return attn:sum(1):clamp(0, 1):log():sum()
    end

    local function selectHypos(step, bsz, scores, indices, ds)
        local selScores = torch.FloatTensor(bsz, beam)
        local selIndices = torch.LongTensor(bsz, beam)
        local eos = dict:getEosIndex()
        for i = 1, bsz do
            local scoresI, indicesI = scores[i], indices[i]
            local selScoresI, selIndicesI = selScores[i], selIndices[i]
            local maxk = scoresI:size(1)
            local j, k = 1, 1
            while j <= beam and k <= maxk do
                if indicesI[k] % ds + 1 ~= eos then
                    -- Not eos: select for next round if word is in vocab
                    if vocab then
                        local bp = math.floor(indicesI[k] / ds) + 1 + ((i-1) * beam)
                        local bidx = math.floor(i / beam) + 1
                        local word = backtraceWord(step, bp, (indicesI[k] % ds) + 1)
                        if vocab[word] or srcVocab[bidx][word] then
                            selScoresI[j], selIndicesI[j] = scoresI[k], indicesI[k]
                            j = j + 1
                        end
                    else
                        selScoresI[j], selIndicesI[j] = scoresI[k], indicesI[k]
                        j = j + 1
                    end
                elseif #finalized[i] < beam then
                    -- Eos: backtrace and store in finalized
                    local bp = math.floor(indicesI[k] / ds) + 1 + ((i-1) * beam)
                    -- scores contains the sum of logprobs for all words in the
                    -- hypothesis.
                    local score = scoresI[k] / math.pow(step, lenPenalty)
                    local hypo = torch.LongTensor(step)
                    local attn = torch.FloatTensor(step, sourcelen)
                    hypo[step] = eos
                    attn:narrow(1, step, 1):copy(
                        ascores[step]:narrow(1, (i-1) * beam + j, 1)
                    )
                    for l = step - 1, 1, -1 do
                        hypo[l] = outs[l][bp]
                        attn:narrow(1, l, 1):copy(ascores[l]:narrow(1, bp, 1))
                        bp = backp[l][bp]
                    end
                    score = score + coveragePenalty * coverageP(attn)
                    table.insert(finalized[i], {
                        hypo = hypo,
                        score = score,
                        attn = attn,
                    })

                    if #finalized[i] == beam then
                        -- The list of finalized hypotheses for this sentence is
                        -- full, so consider this sentence as done. It will
                        -- still be part of future mini-batches, though.
                        notEos = notEos - 1
                    end
                end
                k = k + 1
            end

            -- Not enough non-finalized hypotheses to fill the search beam
            -- or there's a sufficient number of finalized hypotheses already:
            -- simply clone the worst candidate hypothesis (hack).
            while j <= beam do
                selScoresI[j], selIndicesI[j] = scoresI[k-1], indicesI[k-1]
                j = j + 1
            end
        end
        return selScores, selIndices
    end

    local topScores = torch.Tensor():type(ttype)
    local topIndices
    if not ttype:match('torch.Cuda.*') then
        topIndices = torch.LongTensor()
    else
        topIndices = torch.CudaLongTensor()
    end
    local scoresBuf = torch.Tensor():type(ttype)
    f.prune = function(step, ldist, attnscores)
        local as = torch.FloatTensor(attnscores:size()):copy(attnscores)
        table.insert(ascores, as)

        local vocabsize = ldist:size(2)
        local ldistp = ldist:add(bpenalties:narrow(2, 1, vocabsize)):t()
        
        -- Add log-probs of hypotheses at the previous time-step so ldistp will
        -- represent the total log-probability for each new hypothesis.
        ldistp:addr(1, ones:narrow(1, 1, vocabsize), hscores)
        ldistp = ldistp:t()
        local bsz = ldistp:size(1) / beam

        -- Select candidate hypotheses.
        -- The model produces a (bsz * beam X vocabsize) tensor, but for
        -- pruning we'll work with a (bsz X beam * vocabsize) tensor. This
        -- makes it possible to use a single topk() call.  Whenever we work
        -- with the top indices later, it's important to remember that they
        -- refer to a beam * vocabsize slice, i.e.  the actual symbol
        -- index is (index-1 % vocabsize) + 1, while the candidate in the
        -- beam the new hypothesis was produced from is floor(index /
        -- vocabsize + 1.
        local bdist
        if step == 1 then
            -- For the first step, all hypotheses are equal (they start from the
            -- same token) so we simply select candidates from the first one in
            -- the beam.
            bdist = ldistp:unfold(1, 1, beam):squeeze(3)
        else
            bdist = ldistp:view(bsz, ldistp:size(2) * beam)
        end
        topScores, topIndices = clib.topk(
            topScores, topIndices, bdist, beam * 2)
        topIndices:add(-1)
        local selScores, selIndices = selectHypos(step, bsz,
            topScores:float(), topIndices:float(), vocabsize)

        -- Determine actual dictionary indices and hidden state propagation
        -- indices.
        local selIndices1 = selIndices:view(-1)
        local nextIn = torch.remainder(selIndices1, vocabsize) + 1
        local nextHid = torch.div(selIndices1, vocabsize)
        nextHid = nextHid + hidOffsets
        hscores = mutils.sendtobuf(selScores:view(-1), scoresBuf)

        table.insert(outs, nextIn)
        table.insert(backp, nextHid)
        return {
            nextIn = nextIn,
            nextHid = nextHid,
            eos = notEos <= 0,
        }
    end

    f.results = function()
        local hypos = {}
        local scores = {}
        local attns = {}
        for i, v in ipairs(finalized) do
            assert(#v == beam, string.format(
                "beam search didn't return enough hypotheses: %d", #v)
            )

            table.sort(v, function(a, b) return a.score > b.score end)
            for j = 1, beam do
                table.insert(hypos, v[j].hypo)
                table.insert(scores, v[j].score)
                table.insert(attns, v[j].attn)
            end
        end

        return hypos, torch.FloatTensor(scores), attns
    end

    return f
end}

search.visualize = argcheck{
    {name='sf', type='table'},
    {name='dict', type='Dictionary'},
    {name='sourceDict', type='Dictionary'},
    {name='host', type='string'},
    {name='port', type='number'},
    call = function(sf, dict, sourceDict, host, port)
    -- Wrapper for search functions that visualizes attention scores using
    -- visdom.
    local plot = visdom{server = 'http://' .. host, port = port}
    plot.ipv6 = false
    local batchsize, source, remapFn = nil, nil, nil
    local f = {}

    f.init = function(bsz, sample)
        batchsize = bsz
        source = sample.source:t()
        if sample.targetVocab then
            remapFn = function(idx) return sample.targetVocab[idx] end
        else
            remapFn = nil
        end
        return sf.init(bsz, sample)
    end

    f.prune = sf.prune

    f.results = function()
        local res = table.pack(sf.results())

        -- Plot attention scores for best hypothesis
        local attn = res[3]
        local beam = #res[1] / batchsize
        for i = 1, batchsize do
            local idx = (i - 1) * beam + 1

            -- Categorical labels are supposed to be unique, so each word will
            -- be prefixed with its index
            local ssrc = plstringx.split(sourceDict:getString(source[idx]))
            for j = 1, #ssrc do
                ssrc[j] = string.format('[%d] %s', j, ssrc[j])
            end
            local rest
            if remapFn then
                rest = res[1][idx]:clone():apply(remapFn)
            else
                rest = res[i][idx]
            end
            local shyp = plstringx.split(dict:getString(rest))
            for j = 1, #shyp do
                shyp[j] = string.format('[%d] %s', j, shyp[j])
            end

            plot:heatmap{
                X = attn[idx]:t(),
                options = {
                    columnnames = shyp,
                    rownames = ssrc,
                },
            }
        end

        return table.unpack(res)
    end

    return f
end}

return search
