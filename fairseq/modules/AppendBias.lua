-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- The module takes a tensor and extends the last dimension by 1,
-- filling new elements with 1:
--
-- Input:
--  x_11 x_12 x_13
--  x_21 x_21 x_22
--
-- Output:
--  x_11 x_12 x_13 x_14(=1)
--  x_21 x_22 x_23 x_24(=1)
--
--]]

local AppendBias, _ = torch.class('nn.AppendBias', 'nn.Module')

function AppendBias:updateOutput(input)
    local dim = input:dim()
    local size = input:size()
    size[dim] = size[dim] + 1
    self.output:resize(size)
    -- copy input
    self.output:narrow(dim, 1, size[dim] - 1):copy(input)
    -- fill new elements with 1
    self.output:select(dim, size[dim]):fill(1)
    return self.output
end

function AppendBias:updateGradInput(input, gradOutput)
    local dim = input:dim()
    local size = input:size()
    self.gradInput:resize(size)
    -- don't copy added elements
    self.gradInput:copy(gradOutput:narrow(dim, 1, size[dim]))
    return self.gradInput
end
