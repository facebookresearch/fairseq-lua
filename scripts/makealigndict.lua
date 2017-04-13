-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Performs unknown word replacement on output from generation.lua.
-- Prints post-processed hypotheses in sorted order
--
--]]

require 'fairseq'

local tds = require 'tds'
local stringx = require 'pl.stringx'
local tablex = require 'pl.tablex'
local tok = require 'fairseq.text.tokenizer'

local cmd = torch.CmdLine()
cmd:option('-source', '', 'source text')
cmd:option('-target', '', 'target text')
cmd:option('-alignment', '', 'alignment file')
cmd:option('-output', 'aligndict.th7', 'destination file')

local config = cmd:parse(arg)
local tokenize = tok.tokenize
local dict = tds.Hash()

-- Count alignment frequencies
local source = io.open(config.source)
local target = io.open(config.target)
local alignment = io.open(config.alignment)
local n = 0
while true do
    local s = source:read()
    if s == nil then
        break
    end
    local t = target:read()
    local a = alignment:read()

    local stoks = tokenize(s)
    local ttoks = tokenize(t)
    local atoks = tokenize(a)
    for _, atok in ipairs(atoks) do
        local apair = tablex.map(tonumber, stringx.split(atok, '-'))
        local stok = stoks[apair[1] + 1]
        local ttok = ttoks[apair[2] + 1]
        if not dict[stok] then
            dict[stok] = tds.Hash()
        end
        if not dict[stok][ttok] then
            dict[stok][ttok] = 1
        else
            dict[stok][ttok] = dict[stok][ttok] + 1
        end
    end

    n = n + 1
    if n % 25000 == 0 then
        print(string.format('Processed %d sentences', n))
    end
end
print(string.format('Processed %d sentences', n))

-- Only keep the most frequently aligned words
local adict = tds.Hash()
for stok, v in pairs(dict) do
    local maxf = -1
    for ttok, f in pairs(v) do
        if f > maxf then
            maxf = f
            adict[stok] = ttok
        end
    end
end

torch.save(config.output, adict)
