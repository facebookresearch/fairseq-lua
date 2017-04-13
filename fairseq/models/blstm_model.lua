-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- This model uses a bi-directional LSTM encoder. The direction is reversed
-- between layers and two separate columns run in parallel: one on the normal
-- input and one on the reversed input (as described in
-- http://arxiv.org/abs/1606.04199).
--
-- The attention mechanism and the decoder setup are identical to the avgpool
-- model.
--
--]]

require 'nn'
require 'rnnlib'
local usecudnn = pcall(require, 'cudnn')
local argcheck = require 'argcheck'
local mutils = require 'fairseq.models.utils'
local rutils = require 'rnnlib.mutils'

local BLSTMModel = torch.class('BLSTMModel', 'AvgpoolModel')

BLSTMModel.makeEncoderColumn = argcheck{
    {name='self', type='BLSTMModel'},
    {name='config', type='table'},
    {name='inith', type='nngraph.Node'},
    {name='input', type='nngraph.Node'},
    {name='nlayers', type='number'},
    call = function(self, config, inith, input, nlayers)
        local rnnconfig = {
            inputsize = config.nembed,
            hidsize = config.nhid,
            nlayer = 1,
            winitfun = function(network)
                rutils.defwinitfun(network, config.init_range)
            end,
            usecudnn = usecudnn,
        }

        local rnn = nn.LSTM(rnnconfig)
        rnn.saveHidden = false
        local output = nn.SelectTable(-1)(nn.SelectTable(2)(
            rnn({inith, input}):annotate{name = 'encoderRNN'}
        ))
        rnnconfig.inputsize = config.nhid

        for i = 2, nlayers do
            if config.dropout_hid > 0 then
                output = nn.MapTable(nn.Dropout(config.dropout_hid))(output)
            end
            local rnn = nn.LSTM(rnnconfig)
            rnn.saveHidden = false
            output = nn.SelectTable(-1)(nn.SelectTable(2)(
                rnn({
                    inith,
                    nn.ReverseTable()(output),
                })
            ))
        end
        return output
    end
}

BLSTMModel.makeEncoder = argcheck{
    doc=[[
This encoder runs a forward and backward LSTM network and concatenates their
top-most hidden states.
]],
    {name='self', type='BLSTMModel'},
    {name='config', type='table'},
    call = function(self, config)
        local sourceIn = nn.Identity()()
        local inith, tokens = sourceIn:split(2)

        local dict = config.srcdict
        local lut = mutils.makeLookupTable(config, dict:size(),
            dict.pad_index)
        local embed
        if config.dropout_src > 0 then
            embed = nn.MapTable(nn.Sequential()
                :add(lut)
                :add(nn.Dropout(config.dropout_src)))(tokens)
        else
            embed = nn.MapTable(lut)(tokens)
        end

        local col1 = self:makeEncoderColumn{
            config = config,
            inith = inith,
            input = embed,
            nlayers = config.nenclayer,
        }
        local col2 = self:makeEncoderColumn{
            config = config,
            inith = inith,
            input = nn.ReverseTable()(embed),
            nlayers = config.nenclayer,
        }

        -- Each column will switch direction between layers. Before merging,
        -- they should both run in the same direction (here: forward).
        if config.nenclayer % 2 == 0 then
            col1 = nn.ReverseTable()(col1)
        else
            col2 = nn.ReverseTable()(col2)
        end

        local prepare = nn.Sequential()
        -- Concatenate forward and backward states
        prepare:add(nn.JoinTable(2, 2))
        -- Scale down to nhid for further processing
        prepare:add(nn.Linear(config.nhid * 2, config.nembed, false))
        -- Add singleton dimension for subsequent joining
        prepare:add(nn.View(-1, 1, config.nembed))

        local joinedOutput = nn.JoinTable(1, 2)(
            nn.MapTable(prepare)(
                nn.ZipTable()({col1, col2})
            )
        )
        if config.dropout_hid > 0 then
            joinedOutput = nn.Dropout(config.dropout_hid)(joinedOutput)
        end

        -- avgpool_model.makeDecoder() expects two encoder outputs, one for
        -- attention score computation and the other one for applying them.
        -- We'll just use the same output for both.
        return nn.gModule({sourceIn}, {
            joinedOutput, nn.Identity()(joinedOutput)
        })
    end
}

BLSTMModel.prepareSource = argcheck{
    {name='self', type='BLSTMModel'},
    call = function(self)
        -- Device buffers for samples
        local buffers = {
            source = {},
        }

        -- NOTE: It's assumed that all encoders start from the same hidden
        -- state.
        local encoderRNN = mutils.findAnnotatedNode(
            self:network(), 'encoderRNN'
        )
        assert(encoderRNN ~= nil)

        return function(sample)
            -- Encoder input
            local source = {}
            for i = 1, sample.source:size(1) do
                buffers.source[i] = buffers.source[i]
                    or torch.Tensor():type(self:type())
                source[i] = mutils.sendtobuf(sample.source[i],
                    buffers.source[i])
            end

            local initialHidden = encoderRNN:initializeHidden(sample.bsz)
            return {initialHidden, source}
        end
    end
}

return BLSTMModel
