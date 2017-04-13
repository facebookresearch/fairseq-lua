-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Tests for tokenizer.
--
--]]

require 'fairseq'
local tokenizer = require 'fairseq.text.tokenizer'
local tnt = require 'torchnet'
local path = require 'pl.path'
local pltablex = require 'pl.tablex'
local plutils = require 'pl.utils'

local tester
local test = torch.TestSuite()

local testdir = path.abspath(path.dirname(debug.getinfo(1).source:sub(2)))
local testdata = testdir .. '/tst2012.en'
local testdataUrl = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en'
if not path.exists(testdata) then
    require 'os'
    os.execute('curl ' .. testdataUrl .. ' > ' .. testdata)
    if path.getsize(testdata) ~= 140250 then
        error('Failed to download test data from ' .. testdataUrl)
    end
    local head = io.open(testdata):read(15)
    if head ~= 'How can I speak' then
        error('Failed to download test data from ' .. testdataUrl)
    end
end

function test.Tokenizer_BuildDictionary()
    local dict = tokenizer.buildDictionary{
        filename = testdata,
        threshold = 0,
    }
    tester:assertGeneralEq(3730, dict:size())
    tester:assertGeneralEq(dict.unk_index, dict:getIndex('NotInCorpus'))

    local dict2 = tokenizer.buildDictionary{
        filename = testdata,
        threshold = 100,
    }
    tester:assertGeneralEq(38, dict2:size())

    -- Use a custom tokenizer that removes all 'the's
    local dict3 = tokenizer.buildDictionary{
        filename = testdata,
        tokenize = function(line)
            local words  = tokenizer.tokenize(line)
            return pltablex.filter(words, function (w) return w ~= 'the' end)
        end,
        threshold = 0,
    }
    tester:assertGeneralEq(dict3.unk_index, dict3:getIndex('the'))
    tester:assertGeneralEq(3729, dict3:size())
end

function test.Tokenizer_BuildDictionaryMultipleFiles()
    local dict2 = tokenizer.buildDictionary{
        filenames = {testdata, testdata, testdata, testdata},
        threshold = 100 * 4,
    }
    tester:assertGeneralEq(38, dict2:size())
end

function test.Tokenizer_Tensorize()
    local dict = tokenizer.buildDictionary{
        filename = testdata,
        threshold = 0,
    }
    local data, stats = tokenizer.tensorize{
        filename = testdata,
        dict = dict,
    }
    local smap, words = data.smap, data.words
    tester:assertGeneralEq({1553, 2}, smap:size():totable())
    tester:assertGeneralEq({29536}, words:size():totable())
    tester:assertGeneralEq(1553, stats.nseq)
    tester:assertGeneralEq(29536, stats.ntok)
    tester:assertGeneralEq(0, stats.nunk)

    tester:assertGeneralEq('He is my grandfather . </s>', dict:getString(
        words:narrow(1, smap[11][1], smap[11][2])
    ))
end

function test.Tokenizer_TensorizeString()
    local dict = tokenizer.makeDictionary{
        threshold = 0,
    }
    local tokens = plutils.split('aa bb cc', ' ')
    for _, token in ipairs(tokens) do
        dict:addSymbol(token)
    end
    local text = 'aa cc'
    local tensor = tokenizer.tensorizeString{
        text = text,
        dict = dict,
    }
    for i, token in ipairs(plutils.split(text, ' ')) do
        tester:assertGeneralEq(dict:getIndex(token), tensor[i])
    end
end

function test.Tokenizer_TensorizeAlignment()
    local alignmenttext = '0-1 1-2 2-1'
    local tensor = tokenizer.tensorizeAlignment{
        text = alignmenttext,
    }
    tester:assertGeneralEq(tensor:size(1), 3)
    tester:assertGeneralEq(tensor:size(2), 2)
    tester:assertGeneralEq(tensor[1][1], 0 + 1)
    tester:assertGeneralEq(tensor[1][2], 1 + 1)
    tester:assertGeneralEq(tensor[2][1], 1 + 1)
    tester:assertGeneralEq(tensor[2][2], 2 + 1)
    tester:assertGeneralEq(tensor[3][1], 2 + 1)
    tester:assertGeneralEq(tensor[3][2], 1 + 1)
end

function test.Tokenizer_TensorizeThresh()
    local dict = tokenizer.buildDictionary{
        filename = testdata,
        threshold = 50,
    }
    local data, stats = tokenizer.tensorize{
        filename = testdata,
        dict = dict,
    }
    local smap, words = data.smap, data.words
    tester:assertGeneralEq({1553, 2}, smap:size():totable())
    tester:assertGeneralEq({29536}, words:size():totable())
    tester:assertGeneralEq(1553, stats.nseq)
    tester:assertGeneralEq(29536, stats.ntok)
    tester:assertGeneralEq(11485, stats.nunk)

    tester:assertGeneralEq('<unk> is my <unk> . </s>', dict:getString(
        words:narrow(1, smap[11][1], smap[11][2])
    ))
end

function test.Tokenizer_Binarize()
    local dict = tokenizer.buildDictionary{
        filename = testdata,
        threshold = 0,
    }

    -- XXX A temporary directory function would be great
    local dest = os.tmpname()

    local res = tokenizer.binarize{
        filename = testdata,
        dict = dict,
        indexfilename = dest .. '.idx',
        datafilename = dest .. '.bin',
    }
    tester:assertGeneralEq(1553, res.nseq)
    tester:assertGeneralEq(29536, res.ntok)
    tester:assertGeneralEq(0, res.nunk)

    local field = path.basename(dest)
    local ds = tnt.IndexedDataset{
        fields = {field},
        path = paths.dirname(dest),
    }
    tester:assertGeneralEq(1553, ds:size())
    tester:assertGeneralEq('He is my grandfather . </s>', dict:getString(
        ds:get(11)[field]
    ))
end

function test.Tokenizer_BinarizeThresh()
    local dict = tokenizer.buildDictionary{
        filename = testdata,
        threshold = 50,
    }

    -- XXX A temporary directory function would be great
    local dest = os.tmpname()

    local res = tokenizer.binarize{
        filename = testdata,
        dict = dict,
        indexfilename = dest .. '.idx',
        datafilename = dest .. '.bin',
    }
    tester:assertGeneralEq(1553, res.nseq)
    tester:assertGeneralEq(29536, res.ntok)
    tester:assertGeneralEq(11485, res.nunk)

    local field = path.basename(dest)
    local ds = tnt.IndexedDataset{
        fields = {field},
        path = paths.dirname(dest),
    }
    tester:assertGeneralEq(1553, ds:size())
    tester:assertGeneralEq('<unk> is my <unk> . </s>', dict:getString(
        ds:get(11)[field]
    ))
end

function test.Tokenizer_BinarizeAlignment()
    local function makeFile(line)
        local filename = os.tmpname()
        local file = io.open(filename, 'w')
        file:write(line .. '\n')
        file:close(file)
        return filename
    end

    local srcfile = makeFile('a b c a')
    local srcdict = tokenizer.buildDictionary{
        filename = srcfile,
        threshold = 0,
    }

    local tgtfile = makeFile('x y z w x')
    local tgtdict = tokenizer.buildDictionary{
        filename = tgtfile,
        threshold = 0,
    }

    local alignfile = makeFile('0-0 0-1 1-1 2-2 2-4 3-1 3-3')
    local alignfreqmap = tokenizer.buildAlignFreqMap{
        alignfile = alignfile,
        srcfile = srcfile,
        tgtfile = tgtfile,
        srcdict = srcdict,
        tgtdict = tgtdict,
    }

    tester:assertGeneralEq(alignfreqmap[srcdict:getEosIndex()], nil)
    tester:assertGeneralEq(alignfreqmap[srcdict:getPadIndex()], nil)
    tester:assertGeneralEq(alignfreqmap[srcdict:getUnkIndex()], nil)

    tester:assertGeneralEq(
        alignfreqmap[srcdict:getIndex('a')][tgtdict:getIndex('x')], 1)
    tester:assertGeneralEq(
        alignfreqmap[srcdict:getIndex('a')][tgtdict:getIndex('y')], 2)
    tester:assertGeneralEq(
        alignfreqmap[srcdict:getIndex('a')][tgtdict:getIndex('w')], 1)

    tester:assertGeneralEq(
        alignfreqmap[srcdict:getIndex('b')][tgtdict:getIndex('y')], 1)

    tester:assertGeneralEq(
        alignfreqmap[srcdict:getIndex('c')][tgtdict:getIndex('x')], 1)
    tester:assertGeneralEq(
        alignfreqmap[srcdict:getIndex('c')][tgtdict:getIndex('z')], 1)

    local dest = os.tmpname()
    local stats = tokenizer.binarizeAlignFreqMap{
        freqmap = alignfreqmap,
        srcdict = srcdict,
        indexfilename = dest .. '.idx',
        datafilename = dest .. '.bin',
        ncandidates = 2,
    }

    tester:assertGeneralEq(stats.npairs, 5)

    local reader = tnt.IndexedDatasetReader{
        indexfilename = dest .. '.idx',
        datafilename = dest .. '.bin',
    }

    tester:assertGeneralEq(reader:get(srcdict:getEosIndex()):dim(), 0)
    tester:assertGeneralEq(reader:get(srcdict:getPadIndex()):dim(), 0)
    tester:assertGeneralEq(reader:get(srcdict:getUnkIndex()):dim(), 0)
    tester:assert(torch.all(torch.eq(
        reader:get(srcdict:getIndex('a')),
        torch.IntTensor{
            {tgtdict:getIndex('y'), 2},
            {tgtdict:getIndex('x'), 1}})))
    tester:assert(torch.all(torch.eq(
        reader:get(srcdict:getIndex('b')),
        torch.IntTensor{{tgtdict:getIndex('y'), 1}})))
    tester:assert(torch.all(torch.eq(
        reader:get(srcdict:getIndex('c')),
        torch.IntTensor{
            {tgtdict:getIndex('z'), 1},
            {tgtdict:getIndex('x'), 1}})))
end

return function(_tester_)
    tester = _tester_
    return test
end
