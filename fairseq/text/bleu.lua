-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- BLEU scoring
--
--]]

local bleu = {}

local function countNGrams(tokens, order)
    local ngramCounts = {}
    local orderString = tostring(order)
    local len = #tokens
    for i = 1, len - order + 1 do
        local ngram = orderString
        for j = 1, order do
            ngram = ngram .. ' ' .. tostring(tokens[i + j - 1])
        end
        if ngramCounts[ngram] == nil then
            ngramCounts[ngram] = 1
        else
            ngramCounts[ngram] = ngramCounts[ngram] + 1
        end
    end
    return ngramCounts
end

function bleu.scorer(maxOrder)
    local maxOrder = maxOrder or 4
    local totalSys, totalRef = 0, 0
    local allCounts, correctCounts = {}, {}
    local numSents = 0

    for i = 1, maxOrder do
        allCounts[i], correctCounts[i] = 0, 0
    end

    local f = {}

    f.update = function(sys, ref)
        local refNGrams = {}
        for i = 1, maxOrder do
            local ngramCounts = countNGrams(ref, i)
            for ngram, count in pairs(ngramCounts) do
                refNGrams[ngram] = count
            end
        end

        for i = 1, maxOrder do
            local ngramCounts = countNGrams(sys, i)
            for ngram, count in pairs(ngramCounts) do
                allCounts[i] = allCounts[i] + count
                if refNGrams[ngram] ~= nil then
                    if refNGrams[ngram] >= count then
                        correctCounts[i] = correctCounts[i] + count
                    else
                        correctCounts[i] = correctCounts[i] + refNGrams[ngram]
                    end
                end
            end
        end

        totalSys = totalSys + #sys
        totalRef = totalRef + #ref
        numSents = numSents + 1
    end

    local results = function()
        local precision = {}
        local psum = 0
        for i = 1, maxOrder do
            precision[i] = allCounts[i] > 0 and
                (correctCounts[i] / allCounts[i]) or 0
            psum = psum + math.log(precision[i])
        end

        local brevPenalty = 1
        if totalSys < totalRef then
            brevPenalty = math.exp(1 - totalRef / totalSys)
        end

        local bleu = brevPenalty * math.exp(psum / maxOrder) * 100

        return {
            bleu = bleu,
            precision = precision,
            brevPenalty = brevPenalty,
            totalSys = totalSys,
            totalRef = totalRef,
        }
    end
    f.results = results

    f.resultString = function()
        local r = results()
        local str = string.format('BLEU%d = %.2f, ', maxOrder, r.bleu)
        local precs = {}
        for i = 1, maxOrder do
            precs[i] = string.format('%.1f', r.precision[i] * 100)
        end
        str = str .. table.concat(precs, '/')
        str = str .. string.format(
        ' (BP=%.3f, ratio=%.3f, sys_len=%d, ref_len=%d)',
        r.brevPenalty, r.totalRef / r.totalSys, r.totalSys, r.totalRef
        )
        return str
    end

    return f
end

return bleu
