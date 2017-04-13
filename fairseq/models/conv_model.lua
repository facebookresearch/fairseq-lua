-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- A model similar to AvgpoolModel, but with an encoder consisting of
-- 2 parallel stacks of convolutional layers.
--
--]]

require 'nn'
require 'nngraph'
local argcheck = require 'argcheck'
local utils = require 'fairseq.utils'
local mutils = require 'fairseq.models.utils'

local cuda = utils.loadCuda()

local ConvModel, parent = torch.class('ConvModel', 'AvgpoolModel')

ConvModel.__init = argcheck{
    {name='self', type='ConvModel'},
    {name='config', type='table', opt=true},
    call = function(self, config)
        parent.__init(self, config)
    end
}

ConvModel.makeTemporalConvolution = argcheck{
    {name='self', type='ConvModel'},
    {name='config', type='table'},
    {name='ninput', type='number'},
    {name='kwidth', type='number'},
    {name='nhid', type='number'},
    call = function(self, config, ninput, kwidth, nhid)
        local pad = (kwidth - 1) / 2
        local conv
        if config.cudnnconv then
            conv = cuda.cudnn.TemporalConvolution(ninput, nhid, kwidth, 1, pad)
        else
            conv = nn.TemporalConvolutionTBC(ninput, nhid, kwidth, pad)
        end

        -- Initialize weights using the nn implementation
        local nnconv = nn.TemporalConvolution(ninput, nhid,
            kwidth, 1)
        conv.weight:copy(nnconv.weight)
        conv.bias:copy(nnconv.bias)

        -- Scale gradients by sqrt(ninput) to make learning more stable
        conv = nn.GradMultiply(conv, 1 / math.sqrt(ninput))

        return conv
    end
}

ConvModel.makeEncoder = argcheck{
    {name='self', type='ConvModel'},
    {name='config', type='table'},
    call = function(self, config)
        local sourceIn = nn.Identity()()

        -- First, computing embeddings for input tokens and their positions
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
        if not config.cudnnconv then
            embed = nn.Transpose({1, 2})(embed)
        end

        -- This stack is used for computing attention scores
        local cnnA = nn.Sequential()
        if config.nembed ~= config.nhid then
            -- Up-projection for producing nembed-sized output
            cnnA:add(nn.Bottle(
                nn.Linear(config.nembed, config.nhid)
            ))
            -- Bottle requires a continuous gradOutput
            cnnA:add(nn.Contiguous())
        end

        for i = 1, config.nenclayer-1 do
            -- Residual connections
            cnnA:add(nn.ConcatTable()
                :add(self:makeTemporalConvolution(config, config.nhid,
                    config.kwidth, config.nhid))
                :add(nn.Identity()))
            cnnA:add(nn.CAddTable())
            cnnA:add(nn.Tanh())
        end
        cnnA:add(self:makeTemporalConvolution(config, config.nhid,
            config.kwidth, config.nhid))
        cnnA:add(nn.Tanh())

        if config.nembed ~= config.nhid then
            -- Down-projection for producing nembed-sized output
            cnnA:add(nn.Bottle(
                nn.Linear(config.nhid, config.nembed)
            ))
        end
        if not config.cudnnconv then
            cnnA:add(nn.Transpose({1, 2}))
        end

        -- This stack is used for aggregating the context for the decoder (using
        -- the attention scores)
        local cnnC = nn.Sequential()
        local nagglayer = config.nagglayer
        if nagglayer < 0 then
            -- By default, use fewer layers for aggregation than for attention
            nagglayer = math.floor(config.nenclayer / 2)
            nagglayer = math.max(1, math.min(nagglayer, 5))
        end
        for i = 1, nagglayer-1 do
            -- Residual connections
            cnnC:add(nn.ConcatTable()
                :add(self:makeTemporalConvolution(config, config.nembed,
                    config.kwidth, config.nembed))
                :add(nn.Identity()))
            cnnC:add(nn.CAddTable())
            cnnC:add(nn.Tanh())
        end
        cnnC:add(self:makeTemporalConvolution(config, config.nembed,
            config.kwidth, config.nembed))
        cnnC:add(nn.Tanh())
        if not config.cudnnconv then
            cnnC:add(nn.Transpose({1, 2}))
        end

        return nn.gModule({sourceIn}, {cnnA(embed), cnnC(embed)})
    end
}

function ConvModel:float(...)
    self.module:replace(function(m)
        if torch.isTypeOf(m, 'cudnn.TemporalConvolution') then
            return mutils.moveTemporalConvolutionToCPU(m)
        end
        return m
    end)
    return parent.float(self, ...)
end

return ConvModel
