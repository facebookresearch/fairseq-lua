-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- This module allows to perform temporal convolution one time step at a time.
-- It maintains an internal state to buffer signal and accept a single frame
-- as input. This module is for forward evaluation **only** and does not support
-- backpropagation
--
--]]

local LinearizedConvolution, parent =
    torch.class('nn.LinearizedConvolution', 'nn.Linear')

function LinearizedConvolution:__init()
   parent.__init(self, 1, 1)
   self.kw = 0
   self.weight = self.output.new()
   self.bias = self.output.new()
   self.inputBuffer = self.output.new()
   self.tmpBuffer = self.output.new()
end

function LinearizedConvolution:updateOutput(input)
    assert(input:dim() == 3, 'only support batched inputs')
    local bsz = input:size(1)
    local buf = input

    if self.kw > 1 then
        buf = self.inputBuffer
        if buf:dim() == 0 then
            buf:resize(bsz, self.kw, input:size(3)):zero()
        end
        buf:select(2, buf:size(2)):copy(input:select(2, input:size(2)))
    end

    parent.updateOutput(self, buf:view(bsz, -1))
    self.output = self.output:view(bsz, 1, -1)
    return self.output
end

function LinearizedConvolution:resetState()
    self.inputBuffer:resize(0)
end

function LinearizedConvolution:clearState()
   return nn.utils.clear(self, 'output', 'gradInput', 'tmp', 'inputBuffer')
end

function LinearizedConvolution:shiftState(reorder)
    if self.kw > 1 then
        local buf = self.inputBuffer
        local dst = buf:narrow(2, 1, self.kw - 1)
        local src = buf:narrow(2, 2, self.kw - 1)
        local tmp = self.tmpBuffer
        tmp:resizeAs(src):copy(src)
        dst:copy(reorder and tmp:index(1, reorder) or tmp)
    end
end

function LinearizedConvolution:setParameters(weight, bias)
    -- weights should be nout x kw x nin
    local nout, kw, nin = weight:size(1), weight:size(2), weight:size(3)
    self.kw = kw
    self.weight:resize(nout, kw * nin):copy(weight)
    self.bias:resize(bias:size()):copy(bias)
end

function LinearizedConvolution:updateGradInput(input, gradOutput)
    error 'Not supported'
end

function LinearizedConvolution:accGradParameters(input, gradOutput, scale)
    error 'Not supported'
end

function LinearizedConvolution:zeroGradParameters()
    error 'Not supported'
end

function LinearizedConvolution:updateParameters(lr)
    error 'Not supported'
end
