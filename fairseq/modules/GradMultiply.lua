-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- A container that simply sets the scaling factor for accGradParameters
--
--]]

local GradMultiply, parent = torch.class('nn.GradMultiply', 'nn.Container')

function GradMultiply:__init(module, factor)
    parent.__init(self)
    self.modules[1] = module
    self.factor = factor
end

function GradMultiply:updateOutput(input)
    return self.modules[1]:updateOutput(input)
end

function GradMultiply:updateGradInput(input, gradOutput)
    return self.modules[1]:updateGradInput(input, gradOutput)
end

function GradMultiply:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    return self.modules[1]:accGradParameters(input, gradOutput,
        scale * self.factor)
end

function GradMultiply:accUpdateGradParameters(input, gradOutput, lr)
    return self.modules[1]:accUpdateGradParameters(input, gradOutput,
        lr * self.factor)
end
