-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- A helper script to convert a CUDA model into a CPU variant.
--
--]]

require 'fairseq'
local utils = require 'fairseq.utils'

local cuda = utils.loadCuda()
assert(cuda.cutorch)

local cmd = torch.CmdLine()
cmd:option('-input_model', 'cuda_model.th7',
    'a th7 file that contains a CUDA model')
cmd:option('-output_model', 'float_model.th7',
    'an output file that will contain the CPU verion of the model')
local config = cmd:parse(arg)

local model = torch.load(config.input_model)
model:float()
model.module:getParameters()
torch.save(config.output_model, model)
