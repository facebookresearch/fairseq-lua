-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- A dictionary class that manages symbol (e.g. words) to index mapping and vice
-- versa. Building a dictionary is done by repeatedly calling addSymbol() for
-- all symbols in a given corpus and then calling finalize().
--
--]]

local tds = require 'tds'
local argcheck = require 'argcheck'

local Dictionary = torch.class('Dictionary')


Dictionary.__init = argcheck{
    {name='self', type='Dictionary'},
    {name='threshold', type='number', default=0},
    {name='unk', type='string', default='<unk>'},
    {name='pad', type='string', default='<pad>'},
    {name='eos', type='string', default='</s>'},
    call = function(self, threshold, unk, pad, eos)
        self.symbol_to_index = tds.Hash()
        self.index_to_symbol = tds.Vec()
        self.index_to_freq = tds.Vec()
        self.cutoff = math.huge

        -- Pre-populate with unk/pad/eos
        self.unk, self.pad, self.eos = unk, pad, eos
        self:addSymbol(self.unk)
        self.unk_index = self:getIndex(self.unk)
        self:addSymbol(self.pad)
        self.pad_index = self:getIndex(self.pad)
        self:addSymbol(self.eos)
        self.eos_index = self:getIndex(self.eos)
        self.threshold = threshold
        -- It's assumed that indices until and including to self.nspecial are
        -- occupied by special symbols.
        self.nspecial = 3
    end
}

function Dictionary:addSymbol(symbol)
    if self.symbol_to_index[symbol] == nil then
        local index = #self.index_to_symbol + 1
        self.symbol_to_index[symbol] = index
        self.index_to_symbol[index] = symbol
        self.index_to_freq[index] = 1
    else
        local index = self.symbol_to_index[symbol]
        self.index_to_freq[index] = self.index_to_freq[index] + 1
    end
end

function Dictionary:_applyFrequencyThreshold()
    local cutoff = math.huge
    for idx, freq in ipairs(self.index_to_freq) do
        if idx > self.nspecial and freq < self.threshold then
            cutoff = idx - 1
            break
        end
    end

    if cutoff == math.huge then
        -- No regular symbols above threshold, retain special symbols only
        cutoff = self.nspecial
    end
    return cutoff
end

function Dictionary:finalize()
    -- Sort symbols by frequency in descending order, ignoring special ones.
    self.index_to_symbol:sort(function(i, j)
        local idxi = self.symbol_to_index[i]
        local idxj = self.symbol_to_index[j]
        if idxi <= self.nspecial or idxj <= self.nspecial then
            return idxi < idxj
        end
        return self.index_to_freq[idxi] > self.index_to_freq[idxj]
    end)

    -- Update symbol_to_index and index_to_freq mappings
    local new_freq = tds.Vec()
    for idx, sym in ipairs(self.index_to_symbol) do
        local prev = self.symbol_to_index[sym]
        new_freq[idx] = self.index_to_freq[prev]
        self.symbol_to_index[sym] = idx
    end
    self.index_to_freq = new_freq

    collectgarbage()
    if self.threshold > 0 then
        self.cutoff = self:_applyFrequencyThreshold()
    else
        self.cutoff = #self.index_to_symbol
    end
end

function Dictionary:getIndex(symbol)
    local idx = self.symbol_to_index[symbol]
    if idx and idx <= self.cutoff then
        return idx
    end
    return self.unk_index
end

function Dictionary:getSymbol(idx)
    return self.index_to_symbol[idx]
end

function Dictionary:size()
    assert(self.cutoff ~= math.huge, 'Dictionary not finalized')
    return self.cutoff
end

function Dictionary:getUnkIndex()
    return self.unk_index
end

function Dictionary:getPadIndex()
    return self.pad_index
end

function Dictionary:getEosIndex()
    return self.eos_index
end

-- Returns the string of symbols whose indices are provided in vec
function Dictionary:getString(vec)
    local out_tbl = {}
    for i = 1, vec:size(1) do
        table.insert(out_tbl, self:getSymbol(vec[i]))
    end
    local str = table.concat(out_tbl, ' ')
    return str
end
