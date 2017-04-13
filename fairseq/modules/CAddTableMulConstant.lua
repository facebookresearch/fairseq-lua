-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- CAddTable that scales the output by a constant
--
--]]

local CAddTableMulConstant, parent = torch.class('nn.CAddTableMulConstant', 'nn.CAddTable')

function CAddTableMulConstant:__init(constant_scalar)
    parent.__init(self)
    self.constant_scalar = constant_scalar
end

function CAddTableMulConstant:updateOutput(input)
    parent.updateOutput(self, input)
    self.output:mul(self.constant_scalar)
    return self.output
end

function CAddTableMulConstant:updateGradInput(input, gradOutput)
    parent.updateGradInput(self, input, gradOutput)
    for i=1,#self.gradInput do
        self.gradInput[i]:mul(self.constant_scalar)
    end
    return self.gradInput
end
