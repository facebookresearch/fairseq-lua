-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- A conditional LSTM network.
--
--]]

require 'nn'

local argcheck = require 'argcheck'
local rmutils  = require 'rnnlib.mutils'
local rnnlib   = require 'rnnlib.env'

rnnlib.cell.CLSTM = function(nin, nhid, attention)
    local makeVanilla, initVanilla = rnnlib.cell.LSTM(nin * 2, nhid)

    local make = function(prevch, input)
        local input, cond = input:split(2)
        local prevc, prevh = prevch:split(2)

        return makeVanilla(
            nn.Identity()({prevc, prevh}),
            nn.JoinTable(1, 1)({
                input,
                attention({input, prevh, cond}),
            })
        )
    end

    return make, initVanilla
end

local function addDropoutToInput(make, init, prob)
    return function(state, input)
        return make(state, nn.Dropout(prob)(input))
    end, init
end

nn.CLSTM = argcheck{
    { name = "inputsize" , type = "number"    ,                               },
    { name = "hidsize"   , type = "number"    ,                               },
    { name = "nlayer"    , type = "number"    ,                               },
    { name = "attention" , type = "nn.Module" ,                               },
    { name = "hinitfun"   , type = "function" , opt     = true                },
    { name = "winitfun"   , type = "function" , default = rmutils.defwinitfun },
    { name = "savehidden" , type = "boolean"  , default = true                },
    { name = "dropout"    , type = "number"   , default = 0                   },
    { name = "usecudnn"   , type = "boolean"  , default = false               },
    call = function(inputsize, hidsize, nlayer, attention, hinitfun, winitfun,
        savehidden, dropout, usecudnn)

        local modules, initfs = {}, {}
        local c, f = rnnlib.cell.CLSTM(inputsize, hidsize, attention)
        modules[1] = nn.RecurrentTable{dim = 2, module = rnnlib.cell.gModule(c)}
        initfs[1] = f

        if usecudnn and nlayer > 1 then
            modules[2] = nn.CudnnRnnTable{
                module = cudnn.LSTM(hidsize, hidsize, nlayer - 1, false,
                    dropout),
                inputsize = hidsize,
                dropoutin = dropout,
            }
            initfs[2] = modules[2]:makeInitializeHidden()
        else
            for i = 2, nlayer do
                local c, f = rnnlib.cell.LSTM(hidsize, hidsize)
                if dropout > 0 then
                    c, f = addDropoutToInput(c, f, dropout)
                end
                modules[i] = nn.RecurrentTable{
                    dim = 2,
                    module = rnnlib.cell.gModule(c)
                }
                initfs[i] = f
            end
        end

        local network = nn.SequenceTable{
            dim = 1,
            modules = modules,
        }

        local rnn = rnnlib.setupRecurrent{
            network = network,
            initfs = initfs,
            hinitfun = hinitfun,
            winitfun = winitfun,
            savehidden = savehidden,
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
