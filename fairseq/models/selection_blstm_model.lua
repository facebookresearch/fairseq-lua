-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- The BLSTM model that uses words alignment to reduce the target
-- vocabulary size.
--
--]]

require 'nn'
require 'rnnlib'
local argcheck = require 'argcheck'
local mutils = require 'fairseq.models.utils'

local SelectionBLSTMModel = torch.class('SelectionBLSTMModel', 'BLSTMModel')

SelectionBLSTMModel.make = argcheck{
    {name='self', type='SelectionBLSTMModel'},
    {name='config', type='table'},
    call = function(self, config)
        local encoder = self:makeEncoder(config)
        local decoder = self:makeDecoder(config)

        -- Wire up encoder and decoder
        local input = nn.Identity()()
        local prevhIn, targetIn, targetVocab, sourceIn = input:split(4)
        local output = decoder({
            prevhIn,
            targetIn,
            targetVocab,
            encoder(sourceIn):annotate{name = 'encoder'},
        }):annotate{name = 'decoder'}

        return nn.gModule({input}, {output})
    end
}

SelectionBLSTMModel.makeDecoder = argcheck{
    doc=[[
Constructs a conditional LSTM decoder with soft attention.
It also takes an additional input targetVocab to reduce the
target vocabulary size.
]],
    {name='self', type='SelectionBLSTMModel'},
    {name='config', type='table'},
    call = function(self, config)
        local input = nn.Identity()()
        local prevhIn, targetIn, targetVocab, encoderOut = input:split(4)
        local decoderRNNOut = self:makeDecoderRNN(
            config, prevhIn, targetIn, encoderOut)
        local output = mutils.makeTargetMappingWithSelection(
            config, config.dict:size(), decoderRNNOut, targetVocab)
        return nn.gModule({input}, {output})
    end
}

SelectionBLSTMModel.resizeCriterionWeights = argcheck{
    {name='self', type='SelectionBLSTMModel'},
    {name='criterion', type='nn.Criterion'},
    {name='critweights', type='torch.CudaTensor'},
    {name='sample', type='table'},
    call = function(self, criterion, critweights, sample)
        local size = sample.targetVocab:size(1)
        -- Resize criterion weights to match target vocab size
        -- Note: we only use special weights (different from 1.0)
        -- for just a few symbols (like pad), and also we guarantee
        -- that those symbols will have the same ids from batch to batch.
        -- Thus we don't have to remap anything here.
        criterion.nll.weights = critweights:narrow(1, 1, size)
    end
}

SelectionBLSTMModel.prepareSample = argcheck{
    {name='self', type='SelectionBLSTMModel'},
    call = function(self)
        local buffers = {
            targetVocab = torch.Tensor():type(self:type()),
        }

        local prepareSource = self:prepareSource()
        local prepareHidden = self:prepareHidden()
        local prepareInput = self:prepareInput()
        local prepareTarget = self:prepareTarget()

        return function(sample)
            local source = prepareSource(sample)
            local hid = prepareHidden(sample)
            local input = prepareInput(sample)
            local target = prepareTarget(sample)
            local targetVocab = mutils.sendtobuf(
                sample.targetVocab, buffers.targetVocab)

            sample.target = target
            sample.input = {hid, input, targetVocab, source}
        end
    end
}

SelectionBLSTMModel.generationSetup = argcheck{
    {name='self', type='SelectionBLSTMModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local beam = config.beam
        local bbsz = beam * bsz
        local m = self:network()
        local prepareSource = self:prepareSource()
        local decoderRNN = mutils.findAnnotatedNode(m, 'decoderRNN')
        assert(decoderRNN ~= nil)
        local targetVocabBuffer = torch.Tensor():type(self:type())

        return function(sample)
            m:evaluate()

            local state = {
                remapFn = function(idx) return sample.targetVocab[idx] end,
                sourceIn = prepareSource(sample),
                prevhIn = decoderRNN:initializeHidden(bbsz),
                targetVocab = mutils.sendtobuf(sample.targetVocab,
                    targetVocabBuffer),
            }
            return state
        end
    end
}

SelectionBLSTMModel.generationDecode = argcheck{
    {name='self', type='SelectionBLSTMModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local softmax = nn.SoftMax():type(self:type())
        local m = self:network()
        local decoder = mutils.findAnnotatedNode(m, 'decoder')
        return function(state, targetIn)
            targetIn:apply(state.remapFn)
            local out = decoder:forward({
                state.prevhIn, {targetIn}, state.targetVocab, state.encoderOut})
            return softmax:forward(out)
        end
    end
}

SelectionBLSTMModel.generationFinalize = argcheck{
    {name='self', type='SelectionBLSTMModel'},
    {name='config', type='table'},
    call = function(self, config)
        return function(state, sample, results)
            local hypos, _, _ = unpack(results)
            for _, h in ipairs(hypos) do
                h:apply(state.remapFn)
            end
            sample.target:apply(state.remapFn)
        end
    end
}

return SelectionBLSTMModel
