-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- init file for some extra modules.
--
--]]

require 'fairseq.modules.AppendBias'
require 'fairseq.modules.BeamableMM'
require 'fairseq.modules.CAddTableMulConstant'
require 'fairseq.modules.CLSTM'
require 'fairseq.modules.CudnnRnnTable'
require 'fairseq.modules.GradMultiply'
require 'fairseq.modules.LinearizedConvolution'
require 'fairseq.modules.SeqMultiply'
require 'fairseq.modules.TrainTestLayer'
require 'fairseq.modules.ZipAlong'
