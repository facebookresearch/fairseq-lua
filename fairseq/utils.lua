-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Helper functions.
--
--]]

local argcheck = require 'argcheck'
local stringx = require 'pl.stringx'

local util = {}

util.isint = function(x)
    return x ~= nil and type(x) == 'number' and x == math.floor(x)
end

util.loadCuda = argcheck{
    call = function()
        local names = {'cutorch', 'cudnn', 'cunn', 'tbc', 'nccl'}
        local modules = {}
        for _, name in ipairs(names) do
            local ok, module = pcall(require, name)
            modules[name] = ok and module or nil
        end
        return modules
    end
}

util.parseListOrDefault = argcheck{
    {name='str', type='string'},
    {name='n', type='number'},
    {name='val', type='number'},
    {name='del', type='string', default=','},
    call = function(str, n, val, del)
        local kv = {}
        if str == '' then
            for i = 1, n do
                kv[i] = val
            end
        else
            kv = stringx.split(str, del)
            for k, v in pairs(kv) do
                kv[k] = tonumber(v)
            end
        end
        return kv
    end
}

util.sendtogpu = function(data, data_gpu)
    data_gpu = data_gpu or torch.CudaTensor()
    assert(data_gpu and torch.type(data_gpu) == 'torch.CudaTensor')
    assert(data and torch.isTensor(data))
    data_gpu:resize(data:size()):copy(data)
    return data_gpu
end

util.retry = function(n, ...)
    for i = 1, n do
        local status, err = pcall(...)
        if status then
            return true
        end
        print(err)
    end
    return false
end

util.RecyclableSet = function(n)
    local buffer = torch.IntTensor(n):zero()
    local t = 1
    return {
        set = function(self, idx)
            buffer[idx] = t
        end,
        isset = function(self, idx)
            return buffer[idx] == t
        end,
        clear = function(self)
            t = t + 1
        end,
    }
end

return util
