-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Performs unknown word replacement on output from generation.lua. Use
-- 'makealigndict' to generate an alignment dictionary.
-- Prints post-processed hypotheses in sorted order.
--
--]]

require 'fairseq'

local stringx = require 'pl.stringx'
local tablex = require 'pl.tablex'

local cmd = torch.CmdLine()
cmd:option('-genout', '-', 'generation output')
cmd:option('-source', '', 'source language input')
cmd:option('-dict', '', 'path to alignment dictionary')
cmd:option('-unk', '<unk>', 'unknown word marker')
cmd:option('-offset', 0, 'apply offset to attention maxima')

local config = cmd:parse(arg)
local dict = torch.load(config.dict)

local function readFile(path)
    local lines = {}
    local fd = io.open(path)
    while true do
        local line = fd:read()
        if line == nil then
            break
        end
        table.insert(lines, line)
    end
    return lines
end
local srcs = readFile(config.source)

local fd
if config.genout == '-' then
    fd = io.stdin
else
    fd = io.open(config.genout)
end

local hypos = {}
local attns = {}
while true do
    local line = fd:read()
    if line == nil then
        break
    end
    local parts = stringx.split(line, '\t')

    local num = parts[1]:match('^H%-(%d+)')
    if num then
        num = tonumber(num)
        hypos[num] = parts[3]
    else
        num = parts[1]:match('^A%-(%d+)')
        if num then
            num = tonumber(num)
            attns[num] = tablex.map(tonumber, stringx.split(parts[2]))
        end
    end
end

assert(#hypos == #attns,
    'Number of hypotheses and attention scores does not match')
assert(#hypos == #srcs,
    'Number of hypotheses and source sentences does not match')

for i = 1, #hypos do
    local htoks = stringx.split(hypos[i])
    local stoks = stringx.split(srcs[i])
    for j = 1, #htoks do
        if htoks[j] == config.unk then
            local attn = attns[i][j] + config.offset
            if attn < 1 or attn > #stoks then
                io.stderr:write(string.format(
                    'Sentence %d: attention index out of bound: %d\n',
                    i, attn))
            else
                local stok = stoks[attn]
                if dict[stok] then
                    htoks[j] = dict[stok]
                else
                    htoks[j] = stok
                end
            end
        end
    end
    print(stringx.join(' ', htoks))
end
