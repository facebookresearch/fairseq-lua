-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Creates a dictionary from raw text.
--
--]]

require 'fairseq.text'
local tok = require 'fairseq.text.tokenizer'

local cmd = torch.CmdLine()
cmd:option('-text', 'source text')
cmd:option('-out', 'target path')
cmd:option('-threshold', 0,
    'map words appearing less than threshold times to unknown')
cmd:option('-nwords', -1, 'number of non-control target words to retain')
local config = cmd:parse(arg)

assert(not (config.nwords >= 0 and config.threshold > 0),
    'Specify either a frequency threshold or a word count')

local dict = tok.buildDictionary{
    filename = config.text,
    threshold = config.threshold,
}
if config.nwords >= 0 then
    dict.cutoff = config.nwords + dict.nspecial
end

print(string.format('| Dictionary: %d types', dict:size()))
torch.save(config.out, dict, 'binary', false)
