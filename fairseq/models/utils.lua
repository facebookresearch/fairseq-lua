-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Shared utility functions used for model construction.
--
--]]

require 'math'
require 'nn'
local argcheck = require 'argcheck'

local mutils = {}

function mutils.makeLookupTable(config, size, pad)
    local lut = nn.LookupTable(size, config.nembed, pad)
    lut.weight:uniform(-config.init_range, config.init_range)
    return lut
end

function mutils.makeTargetMapping(config, size, nhid)
    local nhid = nhid or config.nembed
    local m = nn.Linear(nhid, size)
    m.bias:fill(0)
    m.weight:uniform(-config.init_range, config.init_range)
    return m
end

function mutils.makeTargetMappingWithSelection(config, size, input, vocab, nhid)
    local nhid = nhid or config.nembed
    local inputwithbias = nn.AppendBias()(input)
    local vocabembed = nn.LookupTable(size, nhid + 1)(vocab)
    vocabembed.data.module.weight:uniform(-config.init_range, config.init_range)
    local m = nn.MM(false, true)({inputwithbias, vocabembed})
    return m
end

function mutils.makeDebug(prefix)
    local debug = nn.Identity()
    debug.updateOutput = function(self, input)
        if self.train then
            print(prefix, 'updateOutput', input:norm(), input:var())
        end
        return nn.Identity.updateOutput(self, input)
    end
    debug.updateGradInput = function(self, input, gradOutput)
        if self.train then
            print(prefix, 'updateGradInput',
                gradOutput:norm(), gradOutput:var())
        end
        return nn.Identity.updateGradInput(self, input, gradOutput)
    end
    return debug
end

function mutils.makeBreakpoint(name)
    -- Requires https://github.com/facebook/fblualib
    local debug = nn.Identity()
    debug.updateOutput = function(self, input)
        print(name)
        require('fb.debugger').enter()
        return nn.Identity.updateOutput(self, input)
    end
    return debug
end

-- Calls a function for all modules in a given container
function mutils.forAllModules(module, fun, ...)
    fun(module, ...)
    if torch.isTypeOf(module, 'nn.gModule') then
        for _, node in pairs(module.forwardnodes) do
            if torch.isTypeOf(node.data.module, 'nn.Module') then
                mutils.forAllModules(node.data.module, fun, ...)
            end
        end
    elseif torch.isTypeOf(module, 'nn.Container') then
        for i = 1, #module.modules do
            mutils.forAllModules(module:get(i), fun, ...)
        end
    elseif torch.isTypeOf(module, 'nn.Recurrent') then
        mutils.forAllModules(module.core, fun, ...)
    end
end

function mutils.findAnnotatedNode(module, name)
    local found = nil
    mutils.forAllModules(module, function(m)
        if not torch.isTypeOf(m, 'nn.gModule') then
            return
        end
        for _, node in pairs(m.forwardnodes) do
            if node.data.annotations.name == name then
                found = node.data.module
            end
        end
    end)

    return found
end

function mutils.loadLegacyModel(path, typename)
    -- XXX This makes a couple of assumptions on the model internals
    local M = require(string.format(
        'fairseq.models.%s_model', typename))
    local model = torch.factory(M.__typename)()
    model.module = torch.load(path)
    return model
end

function mutils.sendtobuf(data, buffer)
    assert(data and torch.isTensor(data))
    assert(buffer and torch.isTensor(buffer))
    buffer:resize(data:size()):copy(data)
    return buffer
end

function mutils.profiler(network)
    local timer = torch.Timer()

    local function addtiming(layer)
        local time = 0
        local fwd = layer.updateOutput
        local bwd = layer.updateGradInput

        -- time forward
        function layer:updateOutput(input)
            if cutorch then cutorch.synchronize() end
            local atstart = timer:time().real
            fwd(self, input)
            if cutorch then cutorch.synchronize() end
            local atend = timer:time().real
            time = time + atend - atstart
            return self.output
        end

        -- time backward
        function layer:updateGradInput(input, gradOutput)
            if cutorch then cutorch.synchronize() end
            local atstart = timer:time().real
            bwd(self, input, gradOutput)
            if cutorch then cutorch.synchronize() end
            local atend = timer:time().real
            time = time + atend - atstart
            return self.gradInput
        end

        return function() return time end
    end

    local profile = {}
    network:apply(function(m)
        if (not torch.isTypeOf(m, 'nn.Container')) then
            local name = torch.typename(m)
            local sz = m.weight and table.concat(m.weight:size():totable(), 'x')
            table.insert(profile, {name=name, time=addtiming(m), sz=sz})
        end
    end)

    function profile:start()
        timer:reset()
    end

    function profile:print()
        local elapsed = timer:time().real
        local name, time = {}, {}
        -- seconds to fraction of time
        for k, v in ipairs(profile) do
            time[k] = v.time()
            name[k] = ('%s (%s)'):format(v.name, v.sz or '')
        end
        -- sort by decreasing time, print cummulative
        time = torch.Tensor(time)
        local totalT = time:sum()
        local _, idx = time:div(totalT):sort(1, true)
        local sum = 0
        print('#Prof CUMUL%  INDIV%  LAYER (weight dim)')
        idx:apply(function(i)
            sum = sum + time[i]
            print(('#PROF %.5f %.5f %s'):format(sum, time[i], name[i]))
        end)

        print(
            ('#Profiler ran for %.5f sec (timed operations %.5f sec, %.5f %%)')
            :format(elapsed, totalT, totalT/elapsed))
    end

    return profile
end

function mutils.singleStackTrace()
    torch.getmetatable('nn.Container').rethrowErrors =
    function(self, module, moduleIndex, funcName, ...)
        return module[funcName](module, ...)
    end
end

mutils.moveTemporalConvolutionToCPU = argcheck{
    {name='layer', type='cudnn.TemporalConvolution'},
    call = function(layer)
        local cpu_model = nn.Sequential()
        local indim = layer.inputFrameSize
        local outdim = layer.outputFrameSize
        local kwidth = layer.kH
        local pad = layer.padH
        -- Stock nn.TemporalConvolution doesn't perform padding
        cpu_model:add(nn.View(1, -1, indim):setNumInputDims(2))
        cpu_model:add(nn.SpatialZeroPadding(0, 0, pad, pad))
        cpu_model:add(nn.View(-1, indim):setNumInputDims(3))
        local conv = {}
        torch.setmetatable(conv, 'nn.TemporalConvolution')
        for k,u in pairs(layer) do
            conv[k] = u
        end
        conv.kW = kwidth
        conv.dW = 1
        conv.inputFrameSize = indim
        conv.outputFrameSize = outdim
        cpu_model:add(conv)
        return cpu_model
    end
}

-- | Copy parameters (including biases) from an cudnn model
-- | to a rnnlib one.
local function copyParamsB(rnn, offset, oldin, oldbin, oldhid, oldbhid,
        hidSize, nlayer)
    local cutils = require 'rnnlib.cudnnutils'

    local inps = oldin:split(hidSize, 1)
    local binps = oldbin:split(hidSize, 1)
    local hids = oldhid:split(hidSize, 1)
    local bhids = oldbhid:split(hidSize, 1)
    local ngates = #inps
    for dir, t in pairs(offset) do
        local old = dir == "input" and inps or hids
        local oldb = dir == "input" and binps or bhids
        for gate, id in pairs(t) do
            local params = cutils.getParams(rnn, nlayer-1, id)
            old[id % ngates + 1]:view(-1):copy(params.weight)
            oldb[id % ngates + 1]:view(-1):copy(params.bias)
        end
    end
end

-- An LSTM cell with biases.
-- This is useful when converting cudnn.LSTM models to rnnlib ones as they may
-- have non-zero biase units.
local BiasLSTM = function(nin, nhid)
    local rnnlib = require 'rnnlib'

    local make = function(prevch, input)
        -- prevch : { prevc : node, prevh : node }
        -- input : node
        local split = {prevch:split(2)}
        local prevc = split[1]
        local prevh = split[2]

        -- the four gates are computed simulatenously
        local i2h   = nn.Linear(nin,  4 * nhid)(input):annotate{name="lstm_i2h"}
        local h2h   = nn.Linear(nhid, 4 * nhid)(prevh):annotate{name="lstm_h2h"}
        -- the gates are separated
        local gates = nn.CAddTable()({i2h, h2h})
        -- assumes that input is of dimension nbatch x ngate * nhid
        gates = nn.SplitTable(2)(nn.Reshape(4, nhid)(gates))
        -- apply nonlinearities:
        local igate = nn.Sigmoid()(nn.SelectTable(1)(gates)):annotate{name="lstm_ig"}
        local fgate = nn.Sigmoid()(nn.SelectTable(2)(gates)):annotate{name="lstm_fg"}
        local cgate = nn.Tanh   ()(nn.SelectTable(3)(gates)):annotate{name="lstm_cg"}
        local ogate = nn.Sigmoid()(nn.SelectTable(4)(gates)):annotate{name="lstm_og"}
        -- c_{t+1} = fgate * c_t + igate * f(h_{t+1}, i_{t+1})
        local nextc = nn.CAddTable()({
            nn.CMulTable()({fgate, prevc}),
            nn.CMulTable()({igate, cgate})
        }):annotate{name="nextc"}
        -- h_{t+1} = ogate * c_{t+1}
        local nexth  = nn.CMulTable()({ogate, nn.Tanh()(nextc)}):annotate{name="lstm_nexth"}
        local nextch = nn.Identity ()({nextc, nexth}):annotate{name="lstm_nextch"}
        return nextch, nexth
    end

    local _, init = rnnlib.cell.LSTM(nin, nhid)
    return make, init
end

mutils.wrappedCudnnRnnToLSTMs = argcheck{
    {name='wtable', type='nn.WrappedCudnnRnn'},
    call = function(wtable)
        local rnnlib = require 'rnnlib'
        local cutils = require 'rnnlib.cudnnutils'

        local culstm = wtable:findModules('cudnn.LSTM')[1]
        -- Reset descriptors so that weights can be accessed
        culstm:resetDropoutDescriptor()
        culstm:resetRNNDescriptor()
        culstm:resetInputDescriptor()
        culstm:resetOutputDescriptor()

        local insize = culstm.inputSize
        local hidsize = culstm.hiddenSize
        local hids = {}
        for i = 1, culstm.numLayers do
            table.insert(hids, hidsize)
        end

        -- Use a LSTM cell with bias units to ensure that we're able to
        -- exactly replicate the cuDNN functionality.
        local rnn = rnnlib.makeRecurrent{
            cellfn = BiasLSTM,
            inputsize = insize,
            hids = hids,
            winitfun = function() end, -- don't reinitialize
            savehidden = wtable.savehidden,
        }

        -- Copy parameters
        rnn:float()
        local params = rnn:parameters()
        for i = 1, culstm.numLayers do
            local off = 4 * (i - 1)
            copyParamsB(culstm, cutils.offsets['LSTM'], params[4 * off + 1],
                params[4 * off + 2], params[4 * off + 3], params[4 * off + 4],
                hidsize, i)
        end
        return rnn
    end
}

mutils.cudnnRnnTableToLSTMs = argcheck{
    {name='ctable', type='nn.CudnnRnnTable'},
    call = function(ctable)
        local rnnlib = require 'rnnlib'
        local cutils = require 'rnnlib.cudnnutils'

        local culstm = ctable:findModules('cudnn.LSTM')[1]
        -- Reset descriptors so that weights can be accessed
        culstm:resetDropoutDescriptor()
        culstm:resetRNNDescriptor()
        culstm:resetInputDescriptor()
        culstm:resetOutputDescriptor()

        local insize = culstm.inputSize
        local hidsize = culstm.hiddenSize
        local result = {}
        for i = 1, culstm.numLayers do
            -- Use a LSTM cell with bias units to ensure that we're able to
            -- exactly replicate the cuDNN functionality.
            local c, f = BiasLSTM(insize, hidsize)
            local module = nn.RecurrentTable{
                dim = 2,
                module = rnnlib.cell.gModule(c)
            }
            local initfs = f

            -- Copy parameters
            module:float()
            local params = module:parameters()
            copyParamsB(culstm, cutils.offsets['LSTM'], params[1],
                params[2], params[3], params[4], hidsize, i)
            insize = hidsize

            table.insert(result, {module, initfs})
        end
        return result
    end
}

mutils.replaceCudnnRNNs = argcheck{
    {name='stable', type='nn.SequenceTable'},
    call = function(stable)
        local rnnlib = require 'rnnlib'

        -- We'll reconstruct the SequenceTable since initialization functions
        -- may have Cuda or cuDNN-related upvalues
        local modules = {}
        local initfs = {}

        -- Previous member initialization functions have been saved as upvalues
        -- (see rnnlib.setupRecurrent). It's fine to keep them as long as
        -- they're not used for cuDNN RNNs.
        local prevInits = {}
        local i = 1
        while true do
            local name, uv = debug.getupvalue(stable.initializeHidden, i)
            if name == 'initfs' then
                prevInits = uv
                break
            elseif not name then
                assert(name, 'initfs not found')
            end
            i = i + 1
        end

        for i, m in ipairs(stable.modules) do
            if torch.isTypeOf(m, 'nn.RecurrentTable') then
                -- Contains a nngraph cell, should be fine
                -- Remove unrolled cell copies
                m.modules = {m.modules[1]}
                table.insert(modules, m)
                table.insert(initfs, prevInits[i])
            elseif torch.isTypeOf(m, 'nn.CudnnRnnTable') then
                -- This is used in CLSTM decoders. Replace it by a couple of
                -- nngraph LSTMs with bias units
                local lstms = mutils.cudnnRnnTableToLSTMs(m)
                for j, rt in pairs(lstms) do
                    local mod, init = table.unpack(rt)
                    table.insert(modules, mod)
                    table.insert(initfs, init)
                end
            else
                -- What's this?
                print('! Warning: dogscience conversion for ' ..
                    torch.typename(m))
                table.insert(modules, m)
                table.insert(initfs, prevInits[i])
            end
        end

        local network = nn.SequenceTable{
            dim = 1,
            modules = modules,
        }
        local rnn = rnnlib.setupRecurrent{
            network = network,
            initfs = initfs,
            winitfun = function() end, -- don't reinitialize
            savehidden = true,
        }
        rnn.getLastHidden = function(self)
            local hids = {}
            for i = 1, #self.modules do
                if self.modules[i].getLastHidden then
                    hids[i] = self.modules[i]:getLastHidden()
                else
                    local hout = self.output[1]
                    hids[i] = hout[i][#hout[i]]
                end
            end
            return hids
        end
        return rnn
    end
}

return mutils
