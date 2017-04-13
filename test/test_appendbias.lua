-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Tests for the nn.AppendBias module.
--
--]]

require 'fairseq.modules'

local tester
local test = torch.TestSuite()

function test.AppendBias_Forward()
    local m = nn.AppendBias()
    local input = torch.Tensor{{2, 3}, {4, 5}}
    local output = torch.Tensor{{2, 3, 1}, {4, 5, 1}}
    tester:assert(torch.all(torch.eq(m:forward(input), output)))
end

function test.AppendBias_Backward()
    local m = nn.AppendBias()
    local input = torch.Tensor{{2, 3}, {4, 5}}
    local gradOutput = torch.Tensor{{7, 8, 2}, {9, 10, 2}}
    local gradInput = torch.Tensor{{7, 8}, {9, 10}}
    tester:assert(torch.all(torch.eq(m:backward(input, gradOutput), gradInput)))
end

return function(_tester_)
    tester = _tester_
    return test
end
