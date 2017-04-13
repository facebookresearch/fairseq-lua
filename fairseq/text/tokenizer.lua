-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Build the word based dataset for a text corpus.
--
--]]


local argcheck = require 'argcheck'
local plutils = require 'pl.utils'
local tds = require 'tds'
local tnt = require 'torchnet'

local tokenizer = {}

tokenizer.pad = '<pad>'
tokenizer.eos = '</s>'
tokenizer.unk = '<unk>'

tokenizer.tokenize = function(line)
    -- Remove extra whitespace
    local s = line:gsub("\t", ""):gsub("^%s+", ""):gsub("%s+$", ""):gsub("%s+", " ")
    return plutils.split(s, ' ')
end

local addFileToDictionary = argcheck{
    {name='filename', type='string'},
    {name='dict', type='Dictionary'},
    {name='tokenize', type='function', default=tokenizer.tokenize},
    call = function(filename, dict, tokenize)
        for s in io.lines(filename) do
            for i, word in pairs(tokenize(s)) do
                dict:addSymbol(word)
            end
            dict:addSymbol(dict.eos)
        end
    end
}

local lineCount = argcheck{
    {name = 'fpath', type = 'string'},
    call = function(fpath)
        local nlines = 0
        for _ in io.lines(fpath) do
            nlines = nlines + 1
        end
        return nlines
    end
}

tokenizer.makeDictionary = argcheck{
    {name='threshold', type='number', default=1},
    call = function(threshold)
        return Dictionary{
            threshold = threshold,
            unk = tokenizer.unk,
            pad = tokenizer.pad,
            eos = tokenizer.eos,
        }
    end
}

tokenizer.buildDictionary = argcheck{
    {name='filename', type='string'},
    {name='threshold', type='number', default=1},
    {name='tokenize', type='function', default=tokenizer.tokenize},
    noordered = true,
    call = function(filename, threshold, tokenize)
        local dict = tokenizer.makeDictionary{
            threshold = threshold,
        }
        addFileToDictionary(filename, dict, tokenize)
        dict:finalize()
        return dict
    end
}

tokenizer.buildDictionary = argcheck{
    {name='filenames', type='table'},
    {name='threshold', type='number', default=1},
    {name='tokenize', type='function', default=tokenizer.tokenize},
    overload = tokenizer.buildDictionary,
    noordered = true,
    call = function(filenames, threshold, tokenize)
        local dict = tokenizer.makeDictionary{
            threshold = threshold,
        }
        for i, filename in pairs(filenames) do
            addFileToDictionary(filename, dict, tokenize)
        end
        dict:finalize()
        return dict
    end
}

tokenizer.buildAlignFreqMap = argcheck{
    {name='alignfile', type='string'},
    {name='srcfile', type='string'},
    {name='tgtfile', type='string'},
    {name='srcdict', type='Dictionary'},
    {name='tgtdict', type='Dictionary'},
    {name='tokenize', type='function', default=tokenizer.tokenize},
    call = function(alignfile, srcfile, tgtfile, srcdict,
        tgtdict, tokenize)
        local freqmap = tds.Vec()
        local srccorp = io.lines(srcfile)
        local tgtcorp = io.lines(tgtfile)

        local function addalignment(alignment, src, tgt)
            if alignment:dim() == 0 then
                return
            end

            -- Compute src-tgt pair frequencies
            for i = 1, alignment:size(1) do
                local srcidx = src[alignment[i][1]]
                assert(srcidx ~= srcdict:getEosIndex())
                assert(srcidx ~= srcdict:getPadIndex())

                local tgtidx = tgt[alignment[i][2]]
                assert(tgtidx ~= tgtdict:getEosIndex())
                assert(tgtidx ~= tgtdict:getPadIndex())

                if srcidx ~= srcdict:getUnkIndex() and
                    tgtidx ~= tgtdict:getUnkIndex() then
                    if not freqmap[srcidx] then
                        freqmap[srcidx] = tds.Hash()
                    end
                    if not freqmap[srcidx][tgtidx] then
                        freqmap[srcidx][tgtidx] = 1
                    else
                        freqmap[srcidx][tgtidx] = freqmap[srcidx][tgtidx] + 1
                    end
                end
            end

        end

        -- TODO: If we modify Dictionary to work better with variable cutoffs
        -- this should be replaced with a proper function
        freqmap:resize(#srcdict.index_to_symbol)
        for line in io.lines(alignfile) do
            addalignment(
                tokenizer.tensorizeAlignment(line, tokenize),
                tokenizer.tensorizeString(srccorp(), srcdict, tokenize),
                tokenizer.tensorizeString(tgtcorp(), tgtdict, tokenize))
        end
        return freqmap
    end
}

tokenizer.tensorizeString = argcheck{
    {name='text', type='string'},
    {name='dict', type='Dictionary'},
    {name='tokenize', type='function', default=tokenizer.tokenize},
    call = function(text, dict, tokenize)
        local words = tokenize(text)
        local ids = torch.LongTensor(#words + 1)
        for i, word in pairs(words) do
            ids[i] = dict:getIndex(word)
        end
        ids[#words + 1] = dict:getEosIndex()
        return ids
    end
}

tokenizer.tensorizeAlignment = argcheck{
    {name='text', type='string'},
    {name='tokenize', type='function', default=tokenizer.tokenize},
    call = function(text, tokenize)
        local tokens = tokenize(text)
        local alignment = torch.IntTensor(#tokens, 2)
        -- Note that alignments are zero-based
        for i, token in ipairs(tokens) do
            local pair = plutils.split(token, '-')
            for j, id in ipairs(pair) do
                alignment[i][j] = tonumber(id) + 1
            end
        end
        return alignment
    end
}

tokenizer.tensorize = argcheck{
    {name='filename', type='string'},
    {name='dict', type='Dictionary'},
    {name='tokenize', type='function', default=tokenizer.tokenize},
    call = function(filename, dict, tokenize)
        local nSequence = lineCount(filename)
        local smap = torch.IntTensor(nSequence, 2)
        local ids = torch.LongTensor(nSequence * 20)
        local nseq, nunk = 0, 0
        local woffset = 1
        local replaced = tds.Hash()
        for s in io.lines(filename) do
            local words = tokenize(s)
            local nwords = #words
            nseq = nseq + 1
            smap[nseq][1] = woffset
            smap[nseq][2] = nwords + 1 -- +1 for the additional </s> character

            while woffset + nwords + 1 > ids:nElement() do
                ids:resize(math.floor(ids:nElement() * 1.5))
            end

            for i, word in pairs(words) do
                local idx = dict:getIndex(word)
                if idx == dict.unk_index and word ~= dict.unk then
                    nunk = nunk + 1
                    if not replaced[word] then
                        replaced[word] = 1
                    else
                        replaced[word] = replaced[word] + 1
                    end
                end
                ids[woffset] = idx
                woffset = woffset + 1
            end
            ids[woffset] = dict.eos_index
            woffset = woffset + 1
        end
        smap = smap:narrow(1, 1, nseq):clone()
        ids = ids:narrow(1, 1, woffset - 1):clone()

        return {smap = smap, words = ids}, {
            nseq = nseq,
            nunk = nunk,
            ntok = ids:nElement(),
            replaced = replaced,
        }
    end
}

tokenizer.binarize = argcheck{
    {name='filename', type='string'},
    {name='dict', type='Dictionary'},
    {name='indexfilename', type='string'},
    {name='datafilename', type='string'},
    {name='tokenize', type='function', default=tokenizer.tokenize},
    call = function(filename, dict, indexfilename, datafilename, tokenize)
        local writer = tnt.IndexedDatasetWriter{
            indexfilename = indexfilename,
            datafilename = datafilename,
            type = 'int',
        }
        local nseq, ntok, nunk = 0, 0, 0
        local ids = torch.IntTensor()
        local replaced = tds.Hash()
        for s in io.lines(filename) do
            local words = tokenize(s)
            local nwords = #words
            ids:resize(nwords + 1)
            nseq = nseq + 1
            for i, word in pairs(words) do
                local idx = dict:getIndex(word)
                if idx == dict.unk_index and word ~= dict.unk then
                    nunk = nunk + 1
                    if not replaced[word] then
                        replaced[word] = 1
                    else
                        replaced[word] = replaced[word] + 1
                    end
                end
                ids[i] = idx
            end
            ids[nwords + 1] = dict.eos_index
            writer:add(ids)
            ntok = ntok + ids:nElement()
        end
        writer:close()

        return {
            nseq = nseq,
            nunk = nunk,
            ntok = ntok,
            replaced = replaced,
        }
    end
}

tokenizer.binarizeAlignFreqMap = argcheck{
    {name='freqmap', type='tds.Vec'},
    {name='srcdict', type='Dictionary'},
    {name='indexfilename', type='string'},
    {name='datafilename', type='string'},
    {name='ncandidates', type='number'},
    {name='tokenize', type='function', default=tokenizer.tokenize},
    call = function(freqmap, srcdict, indexfilename, datafilename,
        ncandidates, tokenize)
        local writer = tnt.IndexedDatasetWriter{
            indexfilename = indexfilename,
            datafilename = datafilename,
            type = 'int',
        }
        local empty = torch.IntTensor()
        local cands = torch.IntTensor()
        local npairs = 0
        for srcidx = 1, #srcdict.index_to_symbol do
            local ncands = freqmap[srcidx] and #freqmap[srcidx] or 0
            if ncands > 0 then
                cands:resize(ncands, 2)
                local j = 1
                for tgtidx, freq in pairs(freqmap[srcidx]) do
                    cands[j][1] = tgtidx
                    cands[j][2] = freq
                    j = j + 1
                end
                ncands = math.min(ncands, ncandidates)
                npairs = npairs + ncands
                local _, indices = torch.topk(cands:narrow(2, 2, 1),
                    ncands, 1, true, true)
                writer:add(cands:index(1, indices:squeeze(2)))
            else
                -- Add empty tensor if there are no candidates for given srcidx
                writer:add(empty)
            end
        end
        writer:close()

        return {
            npairs = npairs
        }
    end
}

return tokenizer
