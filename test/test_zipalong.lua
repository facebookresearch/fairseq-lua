-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Tests for the nn.ZipAlong module.
--
--]]

require 'fairseq.modules'

local tester
local test = torch.TestSuite()

function test.ZipAlong_Simple()
    local m = nn.ZipAlong()
    tester:assertGeneralEq(
        {{1, 4}, {2, 4}, {3, 4}},
        m:forward{{1, 2, 3}, 4}
    )
    tester:assertGeneralEq(
        {{1, {4, 4}}, {2, {4, 4}}, {3, {4, 4}}},
        m:forward{{1, 2, 3}, {4, 4}}
    )
end

local function toTable(t)
    if type(t) == 'table' then
        local tb = {}
        for k,v in pairs(t) do
            tb[k] = toTable(v)
        end
        return tb
    end
    return t:totable()
end

function test.ZipAlong_Tensor()
    local t = torch.Tensor({0, 1, 2})
    local m = nn.ZipAlong()
    tester:assertGeneralEq(
        toTable({{t, t}, {t*2, t}, {t*4, t}}),
        toTable(m:forward{{t, t*2, t*4}, t})
    )
    tester:assertGeneralEq(
        toTable({{t, t*2, t*4}, t*3}),
        toTable(m:backward({{t, t*2, t*4}, t}, {{t, t}, {t*2, t}, {t*4, t}}))
    )

    -- Add table along
    tester:assertGeneralEq(
        toTable({{t, {t, t*2}}, {t*2, {t, t*2}}, {t*4, {t, t*2}}}),
        toTable(m:forward{{t, t*2, t*4}, {t, t*2}})
    )
    tester:assertGeneralEq(
        toTable({{t, t*2, t*4}, {t*3, t*6}}),
        toTable(m:backward({{t, t*2, t*4}, {t, t*2}},
            {{t, {t, t*2}}, {t*2, {t, t*2}}, {t*4, {t, t*2}}}))
    )
end

-- Test in combination with map and add
function test.ZipAlong_UpdateOutputAdd()
    local tensor = torch.Tensor({1, 2, 3, 4})
    local add = torch.Tensor({3, 2, 2, 2})
    local m = nn.Sequential()
    m:add(nn.ZipAlong())
    m:add(nn.MapTable(nn.CAddTable()))

    local input = {{tensor, tensor * 2, tensor * 4}, add}
    local result = {tensor + add, tensor * 2 + add, tensor * 4 + add}
    local output = m:forward(input)
    for i = 1, #input[1] do
        tester:assertGeneralEq(result[i]:totable(), output[i]:totable())
    end
end

-- Test in combination with map and add
function test.ZipAlong_UpdateGradInputAdd()
    local tensor = torch.Tensor({1, 2, 3, 4})
    local add = torch.Tensor({3, 2, 2, 2})
    local m = nn.Sequential()
    m:add(nn.ZipAlong())
    m:add(nn.MapTable(nn.CAddTable()))

    local input = {{tensor, tensor * 2, tensor * 4}, add}
    local gradients = {torch.Tensor({1, 0, 0, 0}), torch.Tensor({0, 1, 0, 0}),
        torch.Tensor({0, 0, 1, 0})}

    m:forward(input)
    local gradInput = m:backward(input, gradients)
    for i = 1, #input[1] do
        tester:assertGeneralEq(
            gradients[i]:totable(),
            gradInput[1][i]:totable()
        )
        tester:assertGeneralEq(
            torch.add(gradients[1], gradients[2]):add(gradients[3]):totable(),
            gradInput[2]:totable()
        )
    end
end

return function(_tester_)
    tester = _tester_
    return test
end
