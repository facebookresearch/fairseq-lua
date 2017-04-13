-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Tests for the Dictionary class.
--
--]]

require 'fairseq.text'

local tester
local test = torch.TestSuite()

function test.Dictionary_Simple()
    local dict = Dictionary{}
    dict:addSymbol('foo')
    dict:addSymbol('bar')
    dict:addSymbol('baz')
    dict:addSymbol('foo')
    dict:finalize()
    tester:assertGeneralEq(6, dict:size()) -- 3 special tokens for unk/pad/eos
    tester:assertGeneralNe(dict.unk_index, dict:getIndex('foo'))
    tester:assertGeneralEq('foo', dict:getSymbol(dict:getIndex('foo')))
    tester:assertGeneralEq(dict.unk_index, dict:getIndex('???'))
    tester:assertGeneralEq(dict.unk, dict:getSymbol(dict:getIndex('???')))
    tester:assertGeneralEq('foo bar </s>', dict:getString(torch.IntTensor{
        dict:getIndex('foo'),
        dict:getIndex('bar'),
        dict.eos_index,
    }))
    -- Dictionary is sorted by frequency
    tester:assert(dict:getIndex('foo') < dict:getIndex('bar'))
end

function test.Dictionary_NoFinalize()
    local dict = Dictionary{}
    dict:addSymbol('foo')
    dict:addSymbol('bar')
    tester:assertError(function() return dict:size() end)
    dict:finalize()
    tester:assertGeneralEq(5, dict:size())
end

function test.Dictionary_Thresholding()
    local dict = Dictionary{threshold=3}
    dict:addSymbol('baz')
    dict:addSymbol('foo')
    dict:addSymbol('foo')
    dict:addSymbol('foo')
    dict:addSymbol('bar')
    dict:addSymbol('bar')
    dict:finalize()
    tester:assertGeneralEq(dict.unk_index, dict:getIndex(dict.unk))
    tester:assertGeneralEq(dict.pad_index, dict:getIndex(dict.pad))
    tester:assertGeneralEq(dict.eos_index, dict:getIndex(dict.eos))
    tester:assertGeneralEq(4, dict:size())
    tester:assertGeneralEq(4, dict:getIndex('foo'))
    tester:assertGeneralEq(dict.unk_index, dict:getIndex('baz'))
    tester:assertGeneralEq(dict.unk_index, dict:getIndex('bar'))

    local dict2 = Dictionary{threshold=2}
    dict2:addSymbol('foo')
    dict2:addSymbol('bar')
    dict2:addSymbol('baz')
    dict2:finalize()
    tester:assertGeneralEq(3, dict2:size())
end

function test.Dictionary_CustomSpecialSymbols()
    local dict = Dictionary{unk='UNK', pad='PAD', eos='EOS'}
    tester:assertGeneralEq('UNK', dict.unk)
    tester:assertGeneralEq('PAD', dict.pad)
    tester:assertGeneralEq('EOS', dict.eos)
    tester:assertGeneralEq(1, dict:getIndex('UNK'))
    tester:assertGeneralEq(2, dict:getIndex('PAD'))
    tester:assertGeneralEq(3, dict:getIndex('EOS'))
end

return function(_tester_)
    tester = _tester_
    return test
end
