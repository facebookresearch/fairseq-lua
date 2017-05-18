-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- This model closely follows the conditional setup of rnn-lib v1, with -name
-- clstm and -aux conv_attn. See the individual functions (makeEncoder,
-- makeDecoder) for detailed comments regarding the model architecture.
--
--]]

require 'nn'
require 'nngraph'
require 'rnnlib'
local argcheck = require 'argcheck'
local mutils = require 'fairseq.models.utils'
local utils = require 'fairseq.utils'
local rmutils = require 'rnnlib.mutils'

local cuda = utils.loadCuda()

local AvgpoolModel, parent = torch.class('AvgpoolModel', 'Model')

AvgpoolModel.make = argcheck{
    {name='self', type='AvgpoolModel'},
    {name='config', type='table'},
    call = function(self, config)
        local encoder = self:makeEncoder(config)
        local decoder = self:makeDecoder(config)

        -- Wire up encoder and decoder
        local input = nn.Identity()()
        local prevhIn, targetIn, sourceIn = input:split(3)
        local output = decoder({
            prevhIn,
            targetIn,
            encoder(sourceIn):annotate{name = 'encoder'},
        }):annotate{name = 'decoder'}

        return nn.gModule({input}, {output})
    end
}

AvgpoolModel.makeEncoder = argcheck{
    doc=[[
This encoder computes embeddings of source language words and their positions.
It produces two outputs, which are later utilized in the soft attention module:
the sum of the embeddings and a windowed version of the embeddings (average
pooling over a small context).
]],
    {name='self', type='AvgpoolModel'},
    {name='config', type='table'},
    call = function(self, config)
        local sourceIn = nn.Identity()()

        -- This aims to follow conv_attn_aux in rnn-lib/condition
        local tokens, positions = sourceIn:split(2)
        local dict = config.srcdict
        local embedToken = mutils.makeLookupTable(config, dict:size(),
            dict:getPadIndex())
        -- XXX Assumes source sentence length < 1024
        local embedPosition =
            mutils.makeLookupTable(config, 1024, dict:getPadIndex())
        local embed =
            nn.CAddTable()({embedToken(tokens), embedPosition(positions)})
        if config.dropout_src > 0 then
            embed = nn.Dropout(config.dropout_src)(embed)
        end

        -- win_type == 1
        local apool = 5
        local pad = (apool - 1) / 2
        local window_model = nn.Sequential()
        window_model:add(nn.View(1, -1, config.nembed):setNumInputDims(2))
        window_model:add(nn.SpatialZeroPadding(0, 0, pad, pad))
        window_model:add(nn.SpatialAveragePooling(1, apool))
        window_model:add(nn.View(-1, config.nembed):setNumInputDims(3))

        -- agg_type == 1
        return nn.gModule({sourceIn}, {window_model(embed), embed})
    end
}

AvgpoolModel.makeAttention = argcheck{
    {name='self', type='AvgpoolModel'},
    {name='config', type='table'},
    {name='attnIn', type='nngraph.Node'},
    call = function(self, config, attnIn)
        -- The attention model uses the windowed encoder output
        -- (encoderOutPooled) for attention score computation, and the
        -- non-windowed output (encoderOutSingle) for computing the conditional
        -- input for the recurrent model.
        -- Furthermore, it computes a separate embedding of the recurrent
        -- model's input, which is combined with the model's previous hidden
        -- state.
        local input, prevh, encoderOut = attnIn:split(3)
        local encoderOutPooled, encoderOutSingle = encoderOut:split(2)

        -- Projection of previous hidden state onto source word space
        local prevhProj = nn.Linear(config.nhid, config.nembed)(prevh)
        local decoderRep = nn.CAddTable()({prevhProj, input})

        -- Compute scores (usually denoted with alpha) using a simple dot
        -- product between encoder output and previous decoder state
        local scores = nn.SoftMax()(
            nn.View(-1):setNumInputDims(2)(
                nn.MM()({
                    encoderOutPooled,
                    nn.View(-1, config.nembed, 1)(decoderRep),
                })
            )
        ):annotate{name = 'attentionScores'}

        -- Compute final attentional vector as a weighted sum of the encoder
        -- output and remove the resulting singleton dimension with nn.Squeeze.
        local attnOut = nn.Squeeze(2, 2)(
            nn.MM(true, false)({
                encoderOutSingle,
                nn.View(-1, 1):setNumInputDims(1)(
                    scores
                ),
            })
        )

        return attnOut
    end
}

AvgpoolModel.makeDecoderRNN = argcheck{
    {name='self', type='AvgpoolModel'},
    {name='config', type='table'},
    {name='prevhIn', type='nngraph.Node'},
    {name='targetIn', type='nngraph.Node'},
    {name='encoderOut', type='nngraph.Node'},
    call = function(self, config, prevhIn, targetIn, encoderOut)
        local attnIn = nn.Identity()()
        local attnOut = self:makeAttention(config, attnIn)
        local attnmodule = nn.gModule({attnIn}, {attnOut})

        -- Decoder network: a conditional LSTM. The attention module defined
        -- above integrates the source language representation produced by the
        -- encoder. We'll take care of the input below.
        local dict = config.dict
        local rnn = nn.CLSTM{
            attention = attnmodule,
            inputsize = config.nembed,
            hidsize = config.nhid,
            nlayer = config.nlayer,
            winitfun = function(network)
                rmutils.defwinitfun(network, config.init_range)
            end,
            dropout = config.dropout_hid,
            usecudnn = cuda.cudnn ~= nil,
        }

        -- Lookup table for embedding the previous target word. The embedding is
        -- used for the decoder and the attention model.
        local targetLut = mutils.makeLookupTable(config, dict:size(),
            dict:getPadIndex())
        local targetEmbed
        if config.dropout_tgt > 0 then
            targetEmbed = nn.MapTable(nn.Sequential()
                :add(targetLut)
                :add(nn.Dropout(config.dropout_tgt)))(targetIn)
        else
            targetEmbed = nn.MapTable(targetLut)(targetIn)
        end

        local scaleHidden = nn.Identity()
        if config.nhid ~= config.nembed then
            scaleHidden = nn.Linear(config.nhid, config.nembed)
        end

        local decoderRNNOut = scaleHidden(
            -- Join over time steps to produce a (bsz*bptt) X nhid tensor
            nn.JoinTable(1)(
                nn.SelectTable(-1)(nn.SelectTable(2)(
                    rnn({
                        -- ZipAlong pairs the encoder output up with every
                        -- entry in targetEmbed (a table of length bptt), so
                        -- it can be fed to the attention model at every
                        -- time step.
                        prevhIn,
                        nn.ZipAlong()({targetEmbed, encoderOut}),
                    }):annotate{name = 'decoderRNN'}
                ))
            )
        )

        if config.dropout_out > 0 then
            decoderRNNOut = nn.Dropout(config.dropout_out)(decoderRNNOut)
        end

        return decoderRNNOut
    end
}

AvgpoolModel.makeDecoder = argcheck{
    doc=[[
Constructs a conditional LSTM decoder with soft attention.
]],
    {name='self', type='AvgpoolModel'},
    {name='config', type='table'},
    call = function(self, config)
        local input = nn.Identity()()
        local prevhIn, targetIn, encoderOut = input:split(3)
        local decoderRNNOut = self:makeDecoderRNN(
            config, prevhIn, targetIn, encoderOut)
        local output = mutils.makeTargetMapping(config, config.dict:size())(
            decoderRNNOut
        )
        return nn.gModule({input}, {output})
    end
}

AvgpoolModel.prepareSource = argcheck{
    {name='self', type='AvgpoolModel'},
    call = function(self)
        -- Device buffers for samples
        local buffers = {
            source = torch.Tensor():type(self:type()),
            sourcePos = torch.Tensor():type(self:type()),
        }

        return function(sample)
            -- The encoder is non-recurrent so it can operate on 2D tensors
            -- directly. The dataset produces bptt X bsz tensors, so they'll
            -- need to be transposed here.
            return {
                mutils.sendtobuf(sample.source:t(), buffers.source),
                mutils.sendtobuf(sample.sourcePos:t(), buffers.sourcePos),
            }
        end
    end
}

AvgpoolModel.prepareHidden = argcheck{
    {name='self', type='AvgpoolModel'},
    call = function(self)
        local decoderRNN = mutils.findAnnotatedNode(
            self:network(),
            'decoderRNN'
        )
        assert(decoderRNN ~= nil)

        return function(sample)
            -- The sample contains a _cont entry if this sample is a
            -- continuation of a previous one (for truncated bptt training). In
            -- that case, start from the RNN's previous hidden state.
            if not sample._cont then
                return decoderRNN:initializeHidden(sample.bsz)
            else
                return decoderRNN:getLastHidden()
            end
        end
    end
}

AvgpoolModel.prepareInput = argcheck{
    {name='self', type='AvgpoolModel'},
    call = function(self)
        local buffers = {
            input = {},
        }

        return function(sample)
            -- Copy data to device buffers. Recurrent modules expect a table of
            -- tensors as their input.
            local input = {}
            for i = 1, sample.input:size(1) do
                buffers.input[i] = buffers.input[i]
                    or torch.Tensor():type(self:type())
                input[i] = mutils.sendtobuf(sample.input[i],
                    buffers.input[i])
            end
            return input
        end
    end
}

AvgpoolModel.prepareTarget = argcheck{
    {name='self', type='AvgpoolModel'},
    call = function(self)
        local buffers = {
            target = torch.Tensor():type(self:type()),
        }

        return function(sample)
            local target = mutils.sendtobuf(sample.target, buffers.target)
                :view(sample.target:nElement())
            return target
        end
    end
}

AvgpoolModel.prepareSample = argcheck{
    {name='self', type='AvgpoolModel'},
    call = function(self)
        local prepareSource = self:prepareSource()
        local prepareHidden = self:prepareHidden()
        local prepareInput = self:prepareInput()
        local prepareTarget = self:prepareTarget()

        return function(sample)
            local source = prepareSource(sample)
            local hid = prepareHidden(sample)
            local input = prepareInput(sample)
            local target = prepareTarget(sample)

            sample.target = target
            sample.input = {hid, input, source}
        end
    end
}

AvgpoolModel.generationSetup = argcheck{
    {name='self', type='AvgpoolModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local beam = config.beam
        local bbsz = beam * bsz
        local m = self:network()
        local prepareSource = self:prepareSource()
        local decoderRNN = mutils.findAnnotatedNode(m, 'decoderRNN')
        assert(decoderRNN ~= nil)

        return function(sample)
            m:evaluate()

            local state = {
                sourceIn = prepareSource(sample),
                prevhIn = decoderRNN:initializeHidden(bbsz),
            }
            return state
        end
    end
}

AvgpoolModel.generationEncode = argcheck{
    {name='self', type='AvgpoolModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local m = self:network()
        local encoder = mutils.findAnnotatedNode(m, 'encoder')
        local beam = config.beam
        local bbsz = beam * bsz

        return function(state)
            local encoderOut = encoder:forward(state.sourceIn)

            -- There will be 'beam' hypotheses for each sentence in the batch,
            -- so duplicate the encoder output accordingly.
            local index = torch.range(1, bsz + 1, 1 / beam)
            index = index:narrow(1, 1, bbsz):floor():long()
            for i = 1, #encoderOut do
                encoderOut[i] = encoderOut[i]:index(1, index)
            end
            state.encoderOut = encoderOut
        end
    end
}

AvgpoolModel.generationDecode = argcheck{
    {name='self', type='AvgpoolModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local softmax = nn.SoftMax():type(self:type())
        local m = self:network()
        local decoder = mutils.findAnnotatedNode(m, 'decoder')
        return function(state, targetIn)
            local out = decoder:forward(
                {state.prevhIn, {targetIn}, state.encoderOut})
            return softmax:forward(out)
        end
    end
}

AvgpoolModel.generationAttention = argcheck{
    {name='self', type='AvgpoolModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local m = self:network()
        local decoderRNN = mutils.findAnnotatedNode(m, 'decoderRNN')
        assert(decoderRNN ~= nil)

        -- Make sure the model is unrolled for the single time step that
        -- we'll use later so that we can get a pointer to
        -- the attention model.
        if torch.isTypeOf(decoderRNN, 'nn.SequenceTable') then
            decoderRNN:apply(function(module)
                if torch.isTypeOf(module, 'nn.RecurrentTable') then
                    module:extend(1)
                end
            end)
        end

        local attnscores = mutils.findAnnotatedNode(
            decoderRNN:get(1):get(1), 'attentionScores')
        if attnscores == nil then
            -- XXX Hack for older models with missing annotations.
            -- This assumes that the decoder is a LSTM with a single layer.
            local attnmodule =
                decoderRNN.rnn[1]:get(1):get(2).forwardnodes[15].data.module
            attnscores = attnmodule.forwardnodes[16].data.module
        end

        return function(state)
            return attnscores.output
        end
    end
}

AvgpoolModel.generationUpdate = argcheck{
    {name='self', type='AvgpoolModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local bbsz = config.beam * bsz
        local m = self:network()
        local decoderRNN = mutils.findAnnotatedNode(m, 'decoderRNN')
        assert(decoderRNN ~= nil)

        return function(state, indexH)
            local lastH = decoderRNN:getLastHidden(bbsz)
            for i = 1, #state.prevhIn do
                for j = 1, #state.prevhIn[i] do
                    local dim = lastH[i][j]:dim() - 1
                    state.prevhIn[i][j]:copy(lastH[i][j]:index(dim, indexH))
                end
            end
        end
    end
}

function AvgpoolModel:float(...)
    self.module:replace(function(m)
        if torch.isTypeOf(m, 'nn.WrappedCudnnRnn') then
            return mutils.wrappedCudnnRnnToLSTMs(m)
        elseif torch.typename(m) == 'nn.SequenceTable' then
            -- Use typename() to avoid matching RecurrentTables
            return mutils.replaceCudnnRNNs(m)
        end
        return m
    end)
    return parent.float(self, ...)
end

return AvgpoolModel
