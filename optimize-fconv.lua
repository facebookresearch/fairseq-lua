-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Optimize a fconv model for fast generation.
--
--]]

require 'fairseq'

local cmd = torch.CmdLine()
cmd:option('-input_model', 'fconv_model.th7',
    'a th7 file that contains a fconv model')
cmd:option('-output_model', 'fconv_model_opt.th7',
    'an output file that will contain an optimized version')
local config = cmd:parse(arg)

local model = torch.load(config.input_model)
if torch.typename(model) ~= 'FConvModel' then
    error '"FConvModel" expected'
end

-- Enable faster decoding
model:makeDecoderFast()

-- Clear output buffers and zero gradients for better compressability
model.module:clearState()
local _, gparams = model.module:parameters()
for i = 1, #gparams do
    gparams[i]:zero()
end

torch.save(config.output_model, model)
