-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Ensemble model.
-- Can only be used for inference.
--
--]]

require 'nn'
require 'rnnlib'
local argcheck = require 'argcheck'
local plstringx = require 'pl.stringx'
local utils = require 'fairseq.utils'

local cuda = utils.loadCuda()

local EnsembleModel = torch.class('EnsembleModel', 'Model')

EnsembleModel.__init = argcheck{
    {name='self', type='EnsembleModel'},
    {name='config', type='table'},
    call = function(self, config)
        local paths = plstringx.split(config.path, ',')
        self.models = {}
        for i, path in pairs(paths) do
            self.models[i] = torch.load(path)
        end
    end
}

EnsembleModel.type = argcheck{
    doc=[[
Shorthand for network():type()
]],
    {name='self', type='EnsembleModel'},
    {name='type', type='string', opt=true},
    {name='tensorCache', type='table', opt=true},
    call = function(self, type, tensorCache)
        local ret = nil
        for _, model in ipairs(self.models) do
            ret = model:type(type, tensorCache)
        end
        return not type and ret or self
    end
}

EnsembleModel.make = argcheck{
    {name='self', type='EnsembleModel'},
    {name='config', type='table'},
    call = function(self, config)
        error('Cannot construct a nn.Module instance for ensemble modules')
    end
}

EnsembleModel.generate = argcheck{
    {name='self', type='EnsembleModel'},
    {name='config', type='table'},
    {name='sample', type='table'},
    {name='search', type='table'},
    call = function(self, config, sample, search)
        local dict = config.dict
        local minlen = config.minlen
        local maxlen = config.maxlen
        local sourceLen = sample.source:size(1)
        local bsz = sample.source:size(2)
        local bbsz = config.beam * bsz

        local timers = {
            setup = torch.Timer(),
            encoder = torch.Timer(),
            decoder = torch.Timer(),
            search_prune = torch.Timer(),
            search_results = torch.Timer(),
        }

        local callbacks = {}
        for _, model in ipairs(self.models) do
            table.insert(callbacks, model:generationCallbacks(config, bsz))
        end

        for _, timer in pairs(timers) do
            timer:stop()
            timer:reset()
        end

        local states = {}
        for i = 1, #self.models do
            states[i] = {
                sample = sample,
            }
        end

        timers.setup:resume()
        local states = {}
        for i = 1, #self.models do
            states[i] = callbacks[i].setup(sample)
        end
        timers.setup:stop()

        timers.encoder:resume()
        for i = 1, #self.models do
            callbacks[i].encode(states[i])
        end
        if cuda.cutorch then
            cuda.cutorch.synchronize()
        end
        timers.encoder:stop()

        -- <eos> is used as a start-of-sentence marker
        local targetIns = {}
        for i = 1, #self.models do
            targetIns[i] = torch.Tensor(bbsz):type(self:type())
            targetIns[i]:fill(dict:getEosIndex())
        end

        search.init(bsz, sample)
        local vocabsize =
            sample.targetVocab and sample.targetVocab:size(1) or dict:size()
        local aggSoftmax = torch.zeros(bbsz, vocabsize):type(self:type())
        local aggAttnScores = torch.zeros(bbsz, sourceLen):type(self:type())
        -- We do maxlen + 1 steps to give model a chance to predict EOS
        for step = 1, maxlen + 1 do
            timers.decoder:resume()
            aggSoftmax:zero()
            aggAttnScores:zero()
            for i = 1, #self.models do
                local softmax = callbacks[i].decode(states[i], targetIns[i])
                aggSoftmax:add(softmax)
                if callbacks[i].attention then
                    aggAttnScores:add(callbacks[i].attention(states[i]))
                end
            end
            -- Average softmax and attention scores.
            aggSoftmax:div(#self.models)
            aggAttnScores:div(#self.models)
            if cuda.cutorch then
                cuda.cutorch.synchronize()
            end
            timers.decoder:stop()

            local aggLogSoftmax = aggSoftmax:log()
            self:updateMinMaxLenProb(aggLogSoftmax, dict, step, minlen, maxlen)

            timers.search_prune:resume()
            local pruned = search.prune(step, aggLogSoftmax, aggAttnScores)
            timers.search_prune:stop()

            for i = 1, #self.models do
                targetIns[i]:copy(pruned.nextIn)
                callbacks[i].update(states[i], pruned.nextHid)
            end

            if pruned.eos then
                break
            end
        end

        timers.search_results:resume()
        local results = table.pack(search.results())
        -- This is pretty hacky, but basically we can't run finalize for
        -- the selection models many times, because it will remap ids many times
        -- TODO: refactor this
        callbacks[1].finalize(states[1], sample, results)
        timers.search_results:stop()

        local times = {}
        for k, v in pairs(timers) do
            times[k] = v:time()
        end
        table.insert(results, times)
        return table.unpack(results)
    end
}

EnsembleModel.extend = argcheck{
    {name='self', type='EnsembleModel'},
    {name='n', type='number'},
    call = function(self, n)
        for _, model in ipairs(self.models) do
            model:extend(n)
        end
    end
}

return EnsembleModel
