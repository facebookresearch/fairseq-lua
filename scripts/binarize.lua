-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
--
--]]

require 'fairseq.text'
local tok = require 'fairseq.text.tokenizer'

local dict = torch.load(arg[1])
local text = arg[2]
local dest = arg[3]

local stats = tok.binarize{
    filename = text,
    dict = dict,
    indexfilename = dest .. '.idx',
    datafilename = dest .. '.bin',
}
print(stats)
