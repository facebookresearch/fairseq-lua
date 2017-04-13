-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.

if not package.loaded['fairseq'] then
    __main__ = true
    require 'fairseq'
end
local dir = require 'pl.dir'
local path = require 'pl.path'

local tester = torch.Tester()
-- Collect all the tests
local testdir = path.abspath(path.dirname(debug.getinfo(1).source:sub(2)))
for _, file in pairs(dir.getfiles(testdir, 'test_*.lua')) do
    tester:add(paths.dofile(file)(tester))
end

local function dotest(tests)
    tester:run(tests)
end

if __main__ then
    if #arg > 0 then
        dotest(arg)
    else
        dotest()
    end
else
    return tester
end
