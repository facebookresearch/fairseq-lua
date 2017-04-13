-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Tests for the topk C implementation
--
--]]

local clib = require 'fairseq.clib'

local tester
local test = torch.TestSuite()

local function timing(f, N, showtime)
    local timer = torch.Timer()
    for i = 1, N do f() end
    local elapsed = timer:time().real
    if showtime then
        print(('topk %.4f msec (ran %.4f sec)'):format(elapsed*1000/N, elapsed))
    end
end

local function dotest(N, bsz, beam, n, v, showtime)
    local t = torch.FloatTensor(bsz, beam * v):uniform()

    if showtime then
        print('torch')
    end
    local top, ind = torch.FloatTensor(), torch.LongTensor()
    timing(function() torch.topk(top, ind, t, n, 2, true, true) end, N, showtime)

    if showtime then
        print('cpp')
    end
    local top2, ind2 = torch.FloatTensor(), torch.LongTensor()
    timing(function() clib.topk(top2, ind2, t, n) end, N, showtime)

    if (v <= 100) then
        -- equality happens if too many samples accuracy is only tested with
        -- short lists.
        tester:assert(ind2:clone():add(-1, ind):abs():sum() == 0)
    end
end

function test.TopK_Accuracy()
    dotest(1, 32, 5, 10, 100, false)
end

--[[ Disable speed test since they're time-consuming
function test.TopK_SingleSpeed()
    dotest(10000/32, 32, 5, 10, 40*1000, true)
end

function test.TopK_Batch()
    dotest(10000, 1, 5, 10, 40*1000, true)
end
--]]

return function(_tester_)
    tester = _tester_
    return test
end
