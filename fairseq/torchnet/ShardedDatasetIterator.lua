-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Iterator that prepares the input data for data parallel training.
--
--]]

local tnt = require 'torchnet'
local argcheck = require 'argcheck'

local ShardedDatasetIterator, DatasetIterator =
    torch.class('tnt.ShardedDatasetIterator', 'tnt.DatasetIterator', tnt)

ShardedDatasetIterator.__init = argcheck{
    doc = [[
<a name="ShardedDatasetIterator">
#### tnt.ShardedDatasetIterator(@ARGP)
@ARGT

This iterator is useful when you want to split out your input batch of samples
into evenly sized sub batches across a specified `dimension`, so you can later
use those for your dataparallel training.

You can specify the exact set of fields that you want to split out by passing
the`fields` argument, the remaining fields will be duplicated.

The output will be in a form of a table of size `nshards` that contains all
the sub batches.

    ]],
    {name='self', type='tnt.ShardedDatasetIterator'},
    {name='dataset', type='tnt.Dataset'},
    {name='nshards', type='number'},
    {name='dimension', type='number', default=1},
    {name='fields', type='table', default={}},
    call = function(self, dataset, nshards, dimension, fields)
        DatasetIterator.__init(self, dataset)
        self:_setup(nshards, dimension, fields)
    end
}

ShardedDatasetIterator.__init = argcheck{
    {name='self', type='tnt.ShardedDatasetIterator'},
    {name='iterator', type='tnt.DatasetIterator'},
    {name='nshards', type='number'},
    {name='dimension', type='number', default=1},
    {name='fields', type='table', default={}},
    overload =  ShardedDatasetIterator.__init,
    call = function(self, iterator, nshards, dimension, fields)
        DatasetIterator.__init(self, iterator)
        self:_setup(nshards, dimension, fields)
    end
}

ShardedDatasetIterator._setup = argcheck{
    {name='self', type='tnt.ShardedDatasetIterator'},
    {name='nshards', type='number'},
    {name='dimension', type='number', default=1},
    {name='fields', type='table', default={}},
    call = function(self, nshards, dimension, fields)
        self.nshards = nshards
        self.dimension = dimension
        self.fields_map = {}
        for _, v in ipairs(fields) do
            self.fields_map[v] = true
        end

        self.base_run = self.run
        self.run = self:_run()
    end
}

local function shouldSplit(k, v, fields)
    local typename = torch.typename(v)
    return fields[k] and typename and typename:match('Tensor')
end

local function inferSize(sample, dimension, fields)
    local size = -1
    for k, v in pairs(sample) do
        if shouldSplit(k, v, fields) then
            local cursize = v:size(dimension)
            assert(size == -1 or size == cursize)
            size = cursize
        end
    end
    assert(size ~= -1)
    return size
end

function ShardedDatasetIterator:_run()
    return function()
        local next_from_base = self.base_run()

        return function()
            local sample = next_from_base()
            if not sample then
                return sample
            end
            local size = inferSize(sample, self.dimension, self.fields_map)
            local shardsz = math.ceil(size / self.nshards)
            local offset = 1
            local result = {}
            for shardid = 1, self.nshards do
                if offset > size then
                    break
                end
                local curshardsz = math.min(shardsz, size - offset + 1)
                result[shardid] = {}
                local shard = result[shardid]
                for k, v in pairs(sample) do
                    if shouldSplit(k, v, self.fields_map) then
                        shard[k] = v:narrow(self.dimension, offset, curshardsz)
                    else
                        shard[k] = v
                    end
                end
                offset = offset + curshardsz
            end
            return result
        end
    end
end
