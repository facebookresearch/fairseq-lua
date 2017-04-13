-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- This module expects a table with two entries, and returns a single table of
-- pairs as follows:
-- Input: {{x_1, x_2, ... x_n}, y}
-- Output: {{x_1, y}. {x_2, y}, ...., {x_n, y}}
--
--]]

local ZipAlong, parent = torch.class('nn.ZipAlong', 'nn.Module')

function ZipAlong:__init()
    parent.__init(self)
    self.output = {}
    self.gradInputBase = {}
end

function ZipAlong:updateOutput(input)
    local base = input[1]
    local dup = input[2]
    self.output = {}
    for i = 1, #base do
        self.output[i] = {base[i], dup}
    end
    return self.output
end

local function zeroTT(dest, src)
    if type(src) == 'table' then
        if not dest or type(dest) ~= 'table' then
            dest = {}
        end
        for k,v in pairs(src) do
            dest[k] = zeroTT(dest[k], v)
        end
    else
        if not dest or not torch.isTypeOf(dest, src) then
            dest = src.new()
        end
        dest:resizeAs(src)
        dest:zero()
    end
    return dest
end

local function addTT(dest, src)
    if type(src) == 'table' then
        for k,v in pairs(src) do
            addTT(dest[k], v)
        end
    else
        dest:add(src)
    end
end

function ZipAlong:updateGradInput(input, gradOutput)
    local basein = input[1]
    local dupin = input[2]

    self.gradInputBase = {}
    self.gradInputDup = zeroTT(self.gradInputDup, dupin)
    for i = 1, #basein do
        self.gradInputBase[i] = gradOutput[i][1]
        addTT(self.gradInputDup, gradOutput[i][2])
    end

    self.gradInput = {self.gradInputBase, self.gradInputDup}
    return self.gradInput
end

function ZipAlong:clearState()
    return nn.utils.clear(self, 'output', 'gradInputBase', 'gradInputDup',
        'gradInput')
end
