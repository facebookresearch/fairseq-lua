-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Converts a fconv_model with a full softmax to a vocabulary selection model
-- for CPU decoding.
--
--]]

local tnt = require 'torchnet'
local plpath = require 'pl.path'
local utils = require 'fairseq.utils'
local mutils = require 'fairseq.models.utils'

local cmd = torch.CmdLine()
cmd:option('-modelin', '', 'path to model with flat softmax output layer')
cmd:option('-modelout', '', 'path where vocab selection model is to be written')
cmd:option('-model', '', 'model type {avgpool|blstm|conv|fconv}')
cmd:option('-sourcelang', 'en', 'source language')
cmd:option('-targetlang', 'de', 'target language')
cmd:option('-datadir', '/mnt/vol/gfsai-flash-east/ai-group/users/' ..
    'michaelauli/data/neuralmt/wmt14_en2de_luong/bpej40k')
cmd:option('-aligndictpath', '', 'path to an alignment dictionary (optional)')
cmd:option('-nembed', 256, 'dimension of embeddings and attention')
cmd:option('-noutembed', 256, 'dimension of the output embeddings')
cmd:option('-nhid', 256, 'number of hidden units per layer')
cmd:option('-nlayer', 1, 'number of hidden layers in decoder')
cmd:option('-nenclayer', 1, 'number of hidden layers in encoder')
cmd:option('-nagglayer', -1,
    'number of layers for conv encoder aggregation stack (CNN-c)')
cmd:option('-kwidth', 3, 'kernel width for conv encoder')
cmd:option('-klmwidth', 3, 'kernel width for convolutional language models')

cmd:option('-cudnnconv', false, 'use cudnn.TemporalConvolution (slower)')
cmd:option('-attnlayers', '-1', 'decoder layers with attention (-1: all)')
cmd:option('-bfactor', 0, 'factor to divide nhid in bottleneck structure')
cmd:option('-fconv_nhids', '',
    'comma-separated list of hidden units for each encoder layer')
cmd:option('-fconv_nlmhids', '',
    'comma-separated list of hidden units for each decoder layer')
cmd:option('-fconv_kwidths', '',
    'comma-separated list of kernel widths for conv encoder')
cmd:option('-fconv_klmwidths', '',
    'comma-separated list of kernel widths for convolutional language model')

local config = cmd:parse(arg)

assert(config.model == 'fconv', 'only conversion for fconv models supported')

-- parse hidden sizes and kernal widths
-- encoder
config.nhids = utils.parseListOrDefault(
    config.fconv_nhids, config.nenclayer, config.nhid)
config.kwidths = utils.parseListOrDefault(
    config.fconv_kwidths, config.nenclayer, config.kwidth)

-- deconder
config.nlmhids = utils.parseListOrDefault(
    config.fconv_nlmhids, config.nlayer, config.nhid)
config.klmwidths = utils.parseListOrDefault(
    config.fconv_klmwidths, config.nlayer, config.klmwidth)


-------------------------------------------------------------------
-- Load data
-------------------------------------------------------------------
config.dict = torch.load(plpath.join(config.datadir,
    'dict.' .. config.targetlang .. '.th7'))
print(string.format('| [%s] Dictionary: %d types', config.targetlang,
    config.dict:size()))
config.srcdict = torch.load(plpath.join(config.datadir,
    'dict.' .. config.sourcelang .. '.th7'))
print(string.format('| [%s] Dictionary: %d types', config.sourcelang,
    config.srcdict:size()))

-- augment config with alignaligndictpath
config.aligndict = tnt.IndexedDatasetReader{
    indexfilename = config.aligndictpath .. '.idx',
    datafilename = config.aligndictpath .. '.bin',
    mmap = true,
    mmapidx = true,
}

-- load existing model and build vocab selection model
local model = torch.load(config.modelin)
local selmodel = require(
    string.format('fairseq.models.%s_model',
        config.model)).new(config)

-- convert both models to CPU
model:float()
selmodel:float()

model.module:evaluate()
selmodel.module:evaluate()
model.module:training()
selmodel.module:training()
local p, _ = model.module:parameters()
local sp, _ = selmodel.module:parameters()
assert(#p - 2 == #sp, 'Number of parameters do not match')

-- copy parameters which should match
for i = 1, #p - 3 do
    sp[i]:copy(p[i]:typeAs(sp[i]))
end

-- find Linear/WeightNorm and LookupTable in both models
local lutm = mutils.findAnnotatedNode(selmodel.module, 'outmodule')
    :get(2):get(1):get(2)
local linm = mutils.findAnnotatedNode(model.module, 'outmodule'):get(4)


-- Next, copy the weights computed by WeightNorm(Linear) to LookupTable
-- Note: we cannot copy the parameters as WeightNorm:parameters() does
-- not return the weight of the wrapped module but only the direction (v) and
-- length (g). So we find and copy the weight tensor (lutm.weight) instead.

-- copy Linear.weight to LookupTable
lutm.weight:narrow(2, 1, config.noutembed):copy(
    linm.weight:typeAs(lutm.weight))

-- copy Linear.bias to LookupTable
lutm.weight:narrow(2, config.noutembed + 1, 1):copy(
    linm.bias:typeAs(lutm.weight))


-- check that the norms of the old and new output word embeddings match
local lut = sp[#sp]
print('compare norms of weight/bias of output layers in each model + params:')
print(string.format('bias norms: %f, %f, %f',
    linm.bias:norm(), lutm.weight:narrow(2, config.noutembed + 1, 1):norm(),
    lut:narrow(2, config.noutembed + 1, 1):norm()))
print(string.format('weight norms: %f, %f, %f',
    linm.weight:norm(), lutm.weight:narrow(2, 1, config.noutembed):norm(),
    lut:narrow(2, 1, config.noutembed):norm()))

assert(linm.bias:ne(lutm.weight:narrow(2, config.noutembed + 1, 1)):sum() == 0)
assert(linm.weight:ne(lutm.weight:narrow(2, 1, config.noutembed)):sum() == 0)

-- save vocab selection model
print(string.format('saving vocab selection model to %s', config.modelout))
torch.save(config.modelout, selmodel)
