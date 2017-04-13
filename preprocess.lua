-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Data pre-processing (binarization). Create dictionary and store parallel data
-- as binary, indexed torchnet datasets, one dataset per language and subset
-- (train/valid/test).
--
-- The following naming scheme is assumed for the parallel corpus:
--
--   $trainpref.$sourcelang - train source language file
--   $trainpref.$targetlang - train target language file
--   $validpref.$sourcelang - validation source language file
--   $validpref.$targetlang - validation target language file
--   $testpref.sourcelang - test source language file
--   $testpref.targetlang - test target language file
--
-- For example:
--   -sourcelang de \
--   -targetlang en \
--   -trainpref ./data/iwslt14.tokenized.de-en/train
-- assumes that there are two files present:
--   ./data/iwslt14.tokenized.de-en/train.de
--   ./data/iwslt14.tokenized.de-en/train.en
--
-- If a file with alignments is given (-alignfile) this script also produces a
-- list (of length -ncandidates) of most common words from the target language
-- for each word from the source language.
--
-- The alignemnt file uses "Pharaoh format", where a pair i-j (zero based)
-- indicates that the ith word of the source language is aligned to the jth
-- word of the target language. For example:
--
--  0-0 1-1 2-4 3-2 4-3 5-5 6-6
--  0-0 1-1 2-2 2-3 3-4 4-5
--
--]]

require 'fairseq'
local tok = require 'fairseq.text.tokenizer'
local lmc = require 'fairseq.text.lm_corpus'
local plpath = require 'pl.path'
local pldir = require 'pl.dir'

local cmd = torch.CmdLine()
cmd:option('-sourcelang', 'de', 'source language')
cmd:option('-targetlang', 'en', 'target language')
cmd:option('-trainpref', 'train', 'training file prefix')
cmd:option('-validpref', 'valid', 'validation file prefix')
cmd:option('-testpref', 'test', 'testing file prefix')
cmd:option('-alignfile', '', 'an alignment file (optional)')
cmd:option('-ncandidates', 1000, 'number of candidates per a source word')
cmd:option('-thresholdtgt', 0,
    'map words appearing less than threshold times to unknown')
cmd:option('-thresholdsrc', 0,
    'map words appearing less than threshold times to unknown')
cmd:option('-nwordstgt', -1,
    'number of target words to retain')
cmd:option('-nwordssrc', -1,
    'number of source words to retain')
cmd:option('-destdir', 'data-bin')

local config = cmd:parse(arg)
assert(not (config.nwordstgt >= 0 and config.thresholdtgt > 0),
    'Specify either a frequency threshold or a word count')
assert(not (config.nwordssrc >= 0 and config.thresholdsrc > 0),
    'Specify either a frequency threshold or a word count')

local langcode = config.sourcelang .. '-' .. config.targetlang
local srcext = string.format('.%s.%s', langcode, config.sourcelang)
local tgtext = string.format('.%s.%s', langcode, config.targetlang)
pldir.makepath(config.destdir)

local src = {
    lang = config.sourcelang,
    threshold = config.thresholdsrc,
    nwords = config.nwordssrc,
    dictbin = plpath.join(
        config.destdir, 'dict.' .. config.sourcelang .. '.th7'
    ),
    traintxt = config.trainpref .. '.' .. config.sourcelang,
    validtxt = config.validpref .. '.' .. config.sourcelang,
    testtxt = config.testpref .. '.' .. config.sourcelang,
    trainbin = plpath.join(config.destdir,  'train' .. srcext),
    validbin = plpath.join(config.destdir, 'valid' .. srcext),
    testbin = plpath.join(config.destdir, 'test' .. srcext),
}

local tgt = {
    lang = config.targetlang,
    threshold = config.thresholdtgt,
    nwords = config.nwordstgt,
    dictbin = plpath.join(
        config.destdir, 'dict.' .. config.targetlang .. '.th7'
    ),
    traintxt = config.trainpref .. '.' .. config.targetlang,
    validtxt = config.validpref .. '.' .. config.targetlang,
    testtxt = config.testpref .. '.' .. config.targetlang,
    trainbin = plpath.join(config.destdir, 'train' .. tgtext),
    validbin = plpath.join(config.destdir, 'valid' .. tgtext),
    testbin = plpath.join(config.destdir, 'test' .. tgtext),
}

for _, lang in ipairs({src, tgt}) do
    lang.dict = tok.buildDictionary{
        filename = lang.traintxt,
        threshold = lang.threshold,
    }
    if lang.nwords >= 0 then
        lang.dict.cutoff = lang.nwords + lang.dict.nspecial
    end

    print(string.format('| [%s] Dictionary: %d types',
        lang.lang, lang.dict:size()))
    torch.save(lang.dictbin, lang.dict, 'binary', false)
    collectgarbage()

    local res = lmc.binarizeCorpus{
        files = {
            {dest=lang.trainbin, src=lang.traintxt},
            {dest=lang.validbin, src=lang.validtxt},
            {dest=lang.testbin, src=lang.testtxt},
        },
        dict = lang.dict,
    }

    local files = {lang.traintxt, lang.validtxt, lang.testtxt}
    for i = 1, #files do
        print(string.format(
            '| [%s] %s: %d sents, %d tokens, %.2f%% replaced by %s',
            lang.lang, files[i], res[i].nseq, res[i].ntok,
            100 * res[i].nunk / res[i].ntok, lang.dict.unk))
    end

    print(string.format('| [%s] Wrote preprocessed data to %s',
        lang.lang, config.destdir))
    collectgarbage()
end

if config.alignfile ~= '' then
    -- Process the alignment file
    local alignfreqmap = tok.buildAlignFreqMap{
        alignfile = config.alignfile,
        srcfile = src.traintxt,
        tgtfile = tgt.traintxt,
        srcdict = src.dict,
        tgtdict = tgt.dict,
    }
    local dest = plpath.join(config.destdir, 'alignment.' .. langcode)
    local stats = tok.binarizeAlignFreqMap{
        freqmap = alignfreqmap,
        srcdict = src.dict,
        indexfilename = dest .. '.idx',
        datafilename = dest .. '.bin',
        ncandidates = config.ncandidates,
    }
    print(string.format(
        '| [%s] Alignments: %d valid pairs', langcode, stats.npairs))
    print(string.format('| [%s] Wrote preprocessed data to %s', langcode, dest))
    collectgarbage()
end
