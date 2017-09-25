-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- MaxBatchDataset builds batches of up to and including maxbatch tokens.
--
--]]


local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local transform = require 'torchnet.transform'
local vector = require 'vector'

local MaxBatchDataset, _ =
   torch.class('tnt.MaxBatchDataset', 'tnt.Dataset', tnt)

MaxBatchDataset.__init = argcheck{
   doc = [[
<a name="MaxBatchDataset">
#### tnt.MaxBatchDataset(@ARGP)
@ARGT
Given a `dataset`, `tnt.MaxBatchDataset` merges samples from this dataset into
a batch such that the total size of the batch does not exceed maxbatch.
]],
    {name='self', type='tnt.MaxBatchDataset'},
    {name='dataset', type='tnt.Dataset'},
    {name='maxbatch', type='number'},
    {name='samplesize', type='function'},
    {name='merge', type='function', opt=true},
    call =
        function(self, dataset, maxbatch, samplesize, merge)
            assert(maxbatch > 0 and math.floor(maxbatch) == maxbatch,
                'maxbatch should be a positive integer number')
            self.dataset = dataset
            self.maxbatch = maxbatch
            self.samplesize = samplesize
            self.makebatch = transform.makebatch{merge=merge}
            self:_buildIndex()
        end
}

MaxBatchDataset._buildIndex  = argcheck{
    {name='self', type='tnt.MaxBatchDataset'},
    call = function(self)
        self.offset = vector.tensor.new_long()
        self.offset[1] = 1
        local size = self.dataset:size()
        local maxssz, maxtsz = 0, 0
        local nstok, nttok = 0, 0

        for i = 1, size do
            local _, ssz, tsz = self.samplesize(self.dataset, i)
            if math.max(ssz, tsz) > self.maxbatch then
                print("warning: found sample that exceeds maxbatch size")
            end

            maxtsz = math.max(maxtsz, tsz)
            maxssz = math.max(maxssz, ssz)
            local nsamples = i - self.offset[#self.offset] + 1
            local tottsz = nsamples * maxtsz
            local totssz = nsamples * maxssz
            nstok = nstok + ssz
            nttok = nttok + tsz

            if i > 1 and math.max(tottsz, totssz) > self.maxbatch then
                self.offset[#self.offset + 1] = i
                maxssz = ssz
                maxtsz = tsz
                nstok = ssz
                nttok = tsz
            end
        end
        self.offset = self.offset:getTensor()
    end
}

MaxBatchDataset.size = argcheck{
    {name='self', type='tnt.MaxBatchDataset'},
    call =
        function(self)
            return self.offset:size(1)
       end
}

MaxBatchDataset.get = argcheck{
   {name='self', type='tnt.MaxBatchDataset'},
   {name='idx', type='number'},
   call =
       function(self, idx)
           assert(idx >= 1 and idx <= self:size(), 'index out of bound')
           local samples = {}
           local first = self.offset[idx]
           local last = idx < self:size() and
               self.offset[idx + 1] - 1 or self.dataset:size()
           for i = first, last do
               local sample = self.dataset:get(i)
               table.insert(samples, sample)
           end
           samples = self.makebatch(samples)
           collectgarbage()
           collectgarbage()
           return samples
       end
}
