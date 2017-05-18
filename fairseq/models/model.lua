-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Base class for models, outlining the basic interface.
--
--]]

local argcheck = require 'argcheck'
local utils = require 'fairseq.utils'

local cuda = utils.loadCuda()

local Model = torch.class('Model')

Model.__init = argcheck{
    doc=[[
Default constructor. This will construct a network by calling `make()`.
]],
    {name='self', type='Model'},
    {name='config', type='table', opt=true},
    call = function(self, config)
        self.module = self:make(config)
    end
}

Model.network = argcheck{
    doc=[[
Returns the encapsulated nn.Module instance.
]],
    {name='self', type='Model'},
    call = function(self)
        return self.module
    end
}

Model.type = argcheck{
    doc=[[
Shorthand for network():type()
]],
    {name='self', type='Model'},
    {name='type', type='string', opt=true},
    {name='tensorCache', type='table', opt=true},
    call = function(self, type, tensorCache)
        local ret = self.module:type(type, tensorCache)
        if ret == self.module then
            return self
        end
        return ret
    end
}

function Model:float(...)
    return self:type('torch.FloatTensor',...)
end

function Model:double(...)
    return self:type('torch.DoubleTensor',...)
end

function Model:cuda(...)
    return self:type('torch.CudaTensor',...)
end

Model.make = argcheck{
    doc=[[
Constructs a new network as a nn.Module instance.
]],
    {name='self', type='Model'},
    call = function(self, config)
        error('Implementation expected')
    end
}

Model.resizeCriterionWeights = argcheck{
    doc=[[
Resize criterion weights to accomadate per sample target vocabulary.

The default implementation is a no-op.
]],
    {name='self', type='Model'},
    {name='criterion', type='nn.Criterion'},
    {name='critweights', type='torch.CudaTensor'},
    {name='sample', type='table'},
    call = function(self, criterion, critweights, sample)
    end
}

Model.prepareSample = argcheck{
    doc=[[
Returns a function that prepares a data sample inside a tochnet engine state.
The nn.Module returned by `network()` is expected to be able to compute a
forward pass on `state.sample.input`.

The default implementation is a no-op.
]],
    {name='self', type='Model'},
    call = function(self)
        return function(sample)
        end
    end
}

Model.generationCallbacks = argcheck{
    doc=[[
Returns 5 callback functions to be used during the generation step:
  - `setup` will be called before the generation step in order
    to prepare a source data sample, extract the attention scores and
    initialize the decoder hidden state
  - `encode` will be called before the generation step to run
    the encoder forward pass and duplicate the output for each beam hypotheses
  - `decode` will be called at each step of the generation,
    it performs the decoder forward pass, and then applies nn.SoftMax
  - `attention` will be called at each step of the generation to
    aquire attention scores
  - `update` will be called at each step of the generation to
    update the decoder hidden state
]],
    {name='self', type='Model'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        return {
            setup = self:generationSetup(config, bsz),
            encode = self:generationEncode(config, bsz),
            decode = self:generationDecode(config, bsz),
            attention = self:generationAttention(config, bsz),
            update = self:generationUpdate(config, bsz),
            finalize = self:generationFinalize(config),
        }
    end
}

Model.generationSetup = argcheck{
    doc=[[
Returns a function that does some preparation before the generation step.
It converts a data sample into required format, extracts the attention
scores node from the graph and initializes hidden state for the decoder.
]],
    {name='self', type='Model'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        error('Implementation expected')
    end
}

Model.generationEncode = argcheck{
    doc=[[
Returns a function that performs the encoder forward pass.
After that it duplicates the encoder output for each
beam hypotheses.
]],
    {name='self', type='Model'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        error('Implementation expected')
    end
}

Model.generationDecode = argcheck{
    doc=[[
Returns a function that performs the decoder forward pass, and then
applies nn.SoftMax on the result.
]],
    {name='self', type='Model'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        error('Implementation expected')
    end
}

Model.generationAttention = argcheck{
    doc=[[
Returns a function that returns attention scores over the source sentences.
Called after the decode callback. This function can be nil.
]],
    {name='self', type='Model'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        return nil
    end
}

Model.generationUpdate = argcheck{
    doc=[[
Returns a function that updates the decoder hidden state.
]],
    {name='self', type='Model'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        error('Implementation expected')
    end
}

Model.generationFinalize = argcheck{
    doc=[[
Returns a function that finalizes generation by performing some transformations.
]],
    {name='self', type='Model'},
    {name='config', type='table'},
    call = function(self, config)
        return function(state, sample, results)
            -- Do nothing
        end
    end
}

Model.generate = argcheck{
    doc=[[
Sentence generation. See search.lua for a description of search functions.
]],
    {name='self', type='Model'},
    {name='config', type='table'},
    {name='sample', type='table'},
    {name='search', type='table'},
    call = function(self, config, sample, search)
        local dict = config.dict
        local minlen = config.minlen
        local maxlen = config.maxlen
        local bsz = sample.source:size(2)
        local bbsz = config.beam * bsz
        local callbacks = self:generationCallbacks(config, bsz)

        local timers = {
            setup = torch.Timer(),
            encoder = torch.Timer(),
            decoder = torch.Timer(),
            search_prune = torch.Timer(),
            search_results = torch.Timer(),
        }

        for k, v in pairs(timers) do
            v:stop()
            v:reset()
        end

        timers.setup:resume()
        local state = callbacks.setup(sample)
        if cuda.cutorch then
            cuda.cutorch.synchronize()
        end
        timers.setup:stop()

        timers.encoder:resume()
        callbacks.encode(state)
        timers.encoder:stop()

        -- <eos> is used as a start-of-sentence marker
        local targetIn = torch.Tensor(bbsz):type(self:type())
        targetIn:fill(dict:getEosIndex())
        local sourceLen = sample.source:size(1)
        local attnscores = torch.zeros(bbsz, sourceLen):type(self:type())

        search.init(bsz, sample)
        -- We do maxlen + 1 steps to give model a chance to
        -- predict EOS
        for step = 1, maxlen + 1 do
            timers.decoder:resume()
            local softmax = callbacks.decode(state, targetIn)
            local logsoftmax = softmax:log()
            if cuda.cutorch then
                cuda.cutorch.synchronize()
            end
            timers.decoder:stop()

            if callbacks.attention then
                attnscores:copy(callbacks.attention(state))
            end

            self:updateMinMaxLenProb(logsoftmax, dict, step, minlen, maxlen)

            timers.search_prune:resume()
            local pruned = search.prune(step, logsoftmax, attnscores)
            targetIn:copy(pruned.nextIn)
            callbacks.update(state, pruned.nextHid)
            timers.search_prune:stop()

            if pruned.eos then
                break
            end
        end

        timers.search_results:resume()
        local results = table.pack(search.results())
        callbacks.finalize(state, sample, results)
        timers.search_results:stop()

        local times = {}
        for k, v in pairs(timers) do
            times[k] = v:time()
        end
        table.insert(results, times)
        return table.unpack(results)
    end
}

Model.extend = argcheck{
    doc=[[
Ensures that recurrent parts of the model are unrolled for a given
number of time-steps.
]],
    {name='self', type='Model'},
    {name='n', type='number'},
    call = function(self, n)
        self:network():apply(function(module)
            if torch.isTypeOf(module, 'nn.Recurrent') then
                module:extend(n)
            elseif torch.isTypeOf(module, 'nn.MapTable') then
                module:resize(n)
            end
        end)
    end
}

function Model:updateMinMaxLenProb(ldist, dict, step, minlen, maxlen)
    local eos = dict:getEosIndex()
    -- Up until we reach minlen, EOS should never be selected
    -- Here we make the probability of chosing EOS -inf
    if step <= minlen then
        ldist:narrow(2, eos, 1):fill(-math.huge)
    end

    -- After reaching maxlen, we need to make sure EOS is selected
    -- so, we make probabilities of everything else -inf
    if step > maxlen then
        local eos = dict:getEosIndex()
        local vocabsize = ldist:size(2)
        ldist:narrow(2, 1, eos - 1):fill(-math.huge)
        ldist:narrow(2, eos + 1, vocabsize - eos):fill(-math.huge)
    end
end

return Model
