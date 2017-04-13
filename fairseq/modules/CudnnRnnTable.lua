-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Wraps a Cudnn.RNN so that the API is the same as Rnnlib's.
--
--]]

require 'nn'

local argcheck = require 'argcheck'
local _, cutils = pcall(require, 'rnnlib.cudnnutils')

local CudnnRnnTable, parent = torch.class('nn.CudnnRnnTable', 'nn.Sequential')

CudnnRnnTable.__init = argcheck{
    { name = 'self'      , type = 'nn.CudnnRnnTable' },
    { name = 'module'    , type = 'nn.Module'        },
    { name = 'inputsize' , type = 'number'           },
    { name = 'dropoutin' , type = 'number', default = 0},
    call = function(self, module, inputsize, dropoutin)
        parent.__init(self)

        -- This joins the across the table dimension of the input
        -- (which has dimension {bptt} x bsz x emsize)
        -- to create a tensor of dimension bptt x bsz x emsize.
        self
            :add(nn.MapTable(nn.View(1, -1, inputsize)))
            :add(nn.JoinTable(1))
        if dropoutin > 0 then
            self:add(nn.Dropout(dropoutin))
        end
        self
            :add(module)
            :add(nn.SplitTable(1))

        self.rnn        = module
        self.output     = {}
        self.gradInput  = {}
    end
}

CudnnRnnTable.__init = argcheck{
    { name = 'self'       , type = 'nn.CudnnRnnTable' },
    { name = 'model'      , type = 'nn.SequenceTable' },
    { name = 'cellstring' , type = 'string'           },
    { name = 'inputsize'  , type = 'number'           },
    { name = 'hiddensize' , type = 'number'           },
    { name = 'nlayer'     , type = 'number'           },
    { name = 'dropoutin'  , type = 'number', default = 0},
    overload = CudnnRnnTable.__init,
    call = function(self, model, cellstring, inputsize, hiddensize, nlayer,
        dropoutin)
        local oldparams = model:parameters()
        local rnn = cudnn[cellstring](inputsize, hiddensize, nlayer)
        for l = 1, nlayer do
            cutils.copyParams(
                rnn, cutils.offsets[cellstring],
                oldparams[2*l-1], oldparams[2*l],
                hiddensize, l
            )
        end
        return self.__init(self, rnn, inputsize, dropoutin)
    end,
}

CudnnRnnTable.updateOutput = function(self, input)
    local module = self.rnn

    local hidinput = input[1]
    local seqinput = input[2]
    if module.mode:find('LSTM') then
        module.cellInput   = hidinput[1]
        module.hiddenInput = hidinput[2]
    else
        module.hiddenInput = hidinput
    end

    local seqoutput = parent.updateOutput(self, seqinput)
    local hidoutput
    if module.mode:find('LSTM') then
        hidoutput    = self  .hidoutput    or {}
        hidoutput[1] = module.cellOutput
        hidoutput[2] = module.hiddenOutput
    else
        hidoutput    = module.hiddenOutput
    end
    self.hidoutput = hidoutput

    self.output = { hidoutput, seqoutput }
    return self.output
end

CudnnRnnTable.updateGradInput = function(self, input, gradOutput)
    local module = self.rnn

    local seqinput      = input[2]
    local hidgradoutput = gradOutput[1]

    local seqgradoutput = gradOutput[2]
    if module.mode:find('LSTM') then
        module.gradCellOutput   = hidgradoutput[1]
        module.gradHiddenOutput = hidgradoutput[2]
    else
        module.gradHiddenOutput = hidgradoutput
    end

    local seqgradinput = parent.updateGradInput(self, seqinput, seqgradoutput)
    local hidgradinput
    if module.mode:find('LSTM') then
        hidgradinput    = self  .hidgradinput    or {}
        hidgradinput[1] = module.gradCellInput
        hidgradinput[2] = module.gradHiddenInput
    else
        hidgradinput    = module.gradHiddenInput
    end
    self.hidgradinput = hidgradinput

    self.gradInput = { hidgradinput, seqgradinput }
    return self.gradInput
end

CudnnRnnTable.accGradParameters = function(self, input, gradOutput, scale)
    local module = self.rnn

    local seqinput      = input[2]
    local hidgradoutput = gradOutput[1]
    local seqgradoutput = gradOutput[2]
    if module.mode:find('LSTM') then
        module.gradCellOutput   = hidgradoutput[1]
        module.gradHiddenOutput = hidgradoutput[2]
    else
        module.gradHiddenOutput = hidgradoutput
    end

    parent.accGradParameters(self, seqinput, seqgradoutput, scale)
    -- Zero out gradBias to conform with Rnnlib standard which does
    -- not use biases in linear projections.
    cutils.zeroField(module, 'gradBias')
end

-- | The backward must be overloaded because nn.Sequential's backward does not
-- actually call updateGradInput or accGradParameters.
CudnnRnnTable.backward = function(self, input, gradOutput, scale)
    local gradInput = self:updateGradInput(input, gradOutput)
    self:accGradParameters(input, gradOutput, scale)
    return gradInput
end

-- | Get the last hidden state.
CudnnRnnTable.getLastHidden = function(self)
    return self.hidoutput
end

CudnnRnnTable.makeInitializeHidden = function(self)
    local module = self.rnn
    return function(bsz, t, cache)
        local dim = {
            module.numLayers,
            bsz,
            module.hiddenSize,
        }
        if module.mode:find('LSTM') then
            cache = cache
                or {
                    torch.CudaTensor(),
                    torch.CudaTensor(),
                }
            cache[1]:resize(table.unpack(dim)):fill(0)
            cache[2]:resize(table.unpack(dim)):fill(0)
        else
            cache = cache or torch.CudaTensor()
            cache   :resize(table.unpack(dim)):fill(0)
        end
        return cache
    end
end
