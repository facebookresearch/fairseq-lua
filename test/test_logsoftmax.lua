-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Tests for the log-softmax C implementation
--
--]]

local clib = require 'fairseq.clib'
local nn = require 'nn'

local tester
local test = torch.TestSuite()

local function timing(f, N, showtime)
    local timer = torch.Timer()
    for i = 1, N do f() end
    local elapsed = timer:time().real
    if showtime then
        print(('lsm %.4f msec (ran %.4f sec)'):format(elapsed*1000/N, elapsed))
    end
end

local function dotest(N, bsz, beam, v, showtime)
    local x = torch.FloatTensor(bsz * beam, v):uniform()

    if showtime then
        print('nn')
    end
    local lsm = nn.LogSoftMax():float()
    timing(function() lsm:forward(x) end, N, showtime)

    if showtime then
        print('cpp')
    end
    local lsm2 = clib.logsoftmax()
    local y
    timing(function() y = lsm2(x) end, N, showtime)

    if N == 1 then
        local err = y:clone():add(-1, lsm.output):abs():max()
        tester:assert(err < 1e-6)
    end
end

function test.LogSoftmax_Accuracy()
    dotest(1, 32, 5, 40*1000, false)
end

--[[ Disable speed test since they're time-consuming
function test.LogSoftmax_SingleSpeed()
    dotest(5000, 1, 5, 40*1000, true)
end

function test.LogSoftmax_Batch()
    dotest(5000/32, 32, 5, 40*1000, true)
end
--]]

return function(_tester_)
    tester = _tester_
    return test
end
