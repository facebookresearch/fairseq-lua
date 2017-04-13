-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Batch hypothesis generation script.
--
--]]

require 'nn'
require 'xlua'
require 'fairseq'

local tnt = require 'torchnet'
local tds = require 'tds'
local plpath = require 'pl.path'
local hooks = require 'fairseq.torchnet.hooks'
local data = require 'fairseq.torchnet.data'
local search = require 'fairseq.search'
local clib = require 'fairseq.clib'
local mutils = require 'fairseq.models.utils'
local utils = require 'fairseq.utils'
local pretty = require 'fairseq.text.pretty'

local cmd = torch.CmdLine()
cmd:option('-path', 'model1.th7,model2.th7', 'path to saved model(s)')
cmd:option('-nobleu', false, 'don\'t produce final bleu score')
cmd:option('-quiet', false, 'don\'t print generated text')
cmd:option('-beam', 1, 'beam width')
cmd:option('-lenpen', 1,
    'length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
cmd:option('-unkpen', 0,
    'unknown word penalty: <0 produces more, >0 produces less unknown words')
cmd:option('-subwordpen', 0,
    'subword penalty: <0 favors longer, >0 favors shorter words')
cmd:option('-covpen', 0,
    'coverage penalty: favor hypotheses that cover all source tokens')
cmd:option('-nbest', 1, 'number of candidate hypotheses')
cmd:option('-batchsize', 16, 'batch size')
cmd:option('-minlen', 1, 'minimum length of generated hypotheses')
cmd:option('-maxlen', 500, 'maximum length of generated hypotheses')
cmd:option('-sourcelang', 'de', 'source language')
cmd:option('-targetlang', 'en', 'target language')
cmd:option('-datadir', 'data-bin')
cmd:option('-dataset', 'test', 'data subset')
cmd:option('-partial', '1/1',
    'decode only part of the dataset, syntax: part_index/num_parts')
cmd:option('-vocab', '', 'restrict output to target vocab')
cmd:option('-seed', 1111, 'random number seed (for dataset)')
cmd:option('-model', '', 'model type for legacy models')
cmd:option('-ndatathreads', 0, 'number of threads for data preparation')
cmd:option('-aligndictpath', '', 'path to an alignment dictionary (optional)')
cmd:option('-nmostcommon', 500,
           'the number of most common words to keep when using alignment')
cmd:option('-topnalign', 100, 'the number of the most common alignments to use')
cmd:option('-freqthreshold', -1,
    'the minimum frequency for an alignment candidate in order' ..
    'to be considered (default no limit)')
cmd:option('-fconvfast', false, 'make fconv model faster')

local cuda = utils.loadCuda()

local config = cmd:parse(arg)
torch.manualSeed(config.seed)
if cuda.cutorch then
    cutorch.manualSeed(config.seed)
end

local function accTime()
    local total = {}
    return function(times)
        for k, v in pairs(times or {}) do
            if not total[k] then
                total[k] = {real = 0, sys = 0, user = 0}
            end
            for l, w in pairs(v) do
                total[k][l] = total[k][l] + w
            end
        end
        return total
    end
end

local function accBleu(beam, dict)
    local scorer = clib.bleu(dict:getPadIndex(), dict:getEosIndex())
    local unkIndex = dict:getUnkIndex()
    local refBuf, hypoBuf = torch.IntTensor(), torch.IntTensor()
    return function(sample, hypos)
        if sample then
            local tgtT = sample.target:t()
            local ref = refBuf:resizeAs(tgtT):copy(tgtT)
                    :apply(function(x)
                        return x == unkIndex and -unkIndex or x
                    end)
            for i = 1, sample.bsz do
                local hypoL = hypos[(i - 1) * beam + 1]
                local hypo = hypoBuf:resize(hypoL:size()):copy(hypoL)
                scorer:add(ref[i], hypo)
            end
        end
        return scorer
    end
end

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

if config.aligndictpath ~= '' then
    config.aligndict = tnt.IndexedDatasetReader{
        indexfilename = config.aligndictpath .. '.idx',
        datafilename = config.aligndictpath .. '.bin',
        mmap = true,
        mmapidx = true,
    }
    config.nmostcommon = math.max(config.nmostcommon, config.dict.nspecial)
    config.nmostcommon = math.min(config.nmostcommon, config.dict:size())
end

local _, test = data.loadCorpus{config = config, testsets = {config.dataset}}
local dataset = test[config.dataset]

local model
if config.model ~= '' then
    model = mutils.loadLegacyModel(config.path, config.model)
else
    model = require(
        'fairseq.models.ensemble_model'
    ).new(config)
    if config.fconvfast then
        local nfconv = 0
        for _, fconv in ipairs(model.models) do
            if torch.typename(fconv) == 'FConvModel' then
                fconv:makeDecoderFast()
                nfconv = nfconv + 1
            end
        end
        assert(nfconv > 0, '-fconvfast requires an fconv model in the ensemble')
    end
end

local vocab = nil
if config.vocab ~= '' then
    vocab = tds.Hash()
    local fd = io.open(config.vocab)
    while true do
        local line = fd:read()
        if line == nil then
            break
        end
        -- Add word on this line together with all prefixes
        for i = 1, line:len() do
            vocab[line:sub(1, i)] = 1
        end
    end
end

local searchf = search.beam{
    ttype = model:type(),
    dict = config.dict,
    srcdict = config.srcdict,
    beam = config.beam,
    lenPenalty = config.lenpen,
    unkPenalty = config.unkpen,
    subwordPenalty = config.subwordpen,
    coveragePenalty = config.covpen,
    vocab = vocab,
}

local dict, srcdict = config.dict, config.srcdict
local display = pretty.displayResults(dict, srcdict, config.nbest, config.beam)
local computeSampleStats = hooks.computeSampleStats(dict)

-- Ensure that the model is fully unrolled for the maximum source sentence
-- length in the test set. Lazy unrolling might otherwise distort the generation
-- time measurements.
local maxlen = 1
for samples in dataset() do
    for _, sample in ipairs(samples) do
        maxlen = math.max(maxlen, sample.source:size(1))
    end
end
model:extend(maxlen)

-- allow to decode only part of the set k/N means decode part k of N
local partidx, nparts = config.partial:match('(%d+)/(%d+)')
partidx, nparts = tonumber(partidx), tonumber(nparts)

-- let's decode
local addBleu = accBleu(config.beam, dict)
local addTime = accTime()
local timer = torch.Timer()
local nsents, ntoks, nbatch = 0, 0, 0
local state = {}
for samples in dataset() do
    if (nbatch % nparts == partidx - 1) then
        assert(#samples == 1, 'can\'t handle multiple samples')
        state.samples = samples
        computeSampleStats(state)
        local sample = state.samples[1]
        local hypos, scores, attns, t = model:generate(config, sample, searchf)
        nsents = nsents + sample.bsz
        ntoks = ntoks + sample.ntokens
        addTime(t)

        -- print results
        if not config.quiet then
            display(sample, hypos, scores, attns)
        end

        -- accumulate bleu
        if (not config.nobleu) then
            addBleu(sample, hypos)
        end
    end
    nbatch = nbatch + 1
end

-- report overall stats
local elapsed = timer:time().real
local statmsg =
    ('| Translated %d sentences (%d tokens) in %.1fs (%.2f tokens/s)')
    :format(nsents, ntoks, elapsed, ntoks / elapsed)
if state.dictstats then
    local avg = state.dictstats.size / state.dictstats.n
    statmsg = ('%s with avg dict of size %.1f'):format(statmsg, avg)
end
print(statmsg)

local timings = '| Timings:'
local totalTime = addTime()
for k, v in pairs(totalTime) do
    local percent = 100 * v.real / elapsed
    timings = ('%s %s %.1fs (%.1f%%),'):format(timings, k, v.real, percent)
end
print(timings:sub(1, -2))

if not config.nobleu then
    local bleu = addBleu()
    print(('| %s'):format(bleu:resultString()))
end
