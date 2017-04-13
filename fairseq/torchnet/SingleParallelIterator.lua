-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- A parallel dataset iterator that can wrap another iterator and is thus
-- limited to using a single thread only.
--
--]]

local tnt = require 'torchnet'
local argcheck = require 'argcheck'
local Threads = require 'threads'
local doc = require 'argcheck.doc'

local SingleParallelIterator = torch.class('tnt.SingleParallelIterator', 'tnt.DatasetIterator', tnt)

SingleParallelIterator.__init = argcheck{
    {name='self', type='tnt.SingleParallelIterator'},
    {name='init', type='function', default=function(idx) end},
    {name='closure', type='function'},
    call = function(self, init, closure)
        local function main(idx)
            giterator = closure(idx)
            assert(torch.isTypeOf(giterator, 'tnt.DatasetIterator'),
            'closure should return a DatasetIterator class')
            gloop = nil
         end
         Threads.serialization('threads.sharedserialize')
         local threads = Threads(1, init, main)
         self.__threads = threads
         local sample -- beware: do not put this line in loop()
         local sampleOrigIdx
         function self.run()
            -- make sure we are not in the middle of something
            threads:synchronize()
            local function enqueue()
                threads:addjob(
                    function()
                        if not gloop then
                            gloop = giterator:run()
                        end
                        local sample = gloop()
                        collectgarbage()
                        collectgarbage()
                        if not sample then
                            gloop = nil
                        end
                        return sample
                    end,
                    function(_sample_)
                        sample = _sample_
                    end)
            end

            enqueue()
            local iterFunction = function()
                while threads:hasjob() do
                    threads:dojob()
                    if threads:haserror() then
                        threads:synchronize()
                    end
                    if sample then
                        enqueue()
                    end
                    return sample
                end
            end

            return iterFunction
         end
    end
}

SingleParallelIterator.exec =
    function(self, name, ...)
        assert(not self.__threads:hasjob(), 'cannot exec during loop')
        local args = {...}
        local res
        self.__threads:addjob(
            function()
                return giterator:exec(name, table.unpack(args))
            end,
            function(...)
                res = {...}
            end)
        self.__threads:synchronize()
        return table.unpack(res)
    end
