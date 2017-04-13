-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Like MulConstant, but the factor is slen * sqrt(1 / slen). slen is the
-- current sequence length (determined by size(2) of the second input element).
--
--]]

local SeqMultiply, parent = torch.class('nn.SeqMultiply', 'nn.Module')

function SeqMultiply:__init()
    parent.__init(self)
    self.scale = 1
end

SeqMultiply.updateOutput = function(self, input)
    local slen = input[2]:size(2)
    self.scale = slen * math.sqrt(1 / slen)
    self.output:resizeAs(input[1])
    self.output:copy(input[1])
    self.output:mul(self.scale)
    return self.output
end

SeqMultiply.updateGradInput = function(self, input, gradOutput)
    self.zeroGrads = self.zeroGrads or input[2].new()
    self.zeroGrads:resizeAs(input[2]):zero()
    self.grads = self.grads or input[1].new()
    self.grads:resizeAs(gradOutput)
    self.grads:copy(gradOutput)
    self.grads:mul(self.scale)
    self.gradInput = {self.grads, self.zeroGrads}
    return self.gradInput
end
