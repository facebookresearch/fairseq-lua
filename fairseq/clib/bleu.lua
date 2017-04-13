-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- BLEU scorer that operates directly over tensors
-- as opposed to bleu.lua which is string based and
-- takes tables as inputs
--
--]]

local ffi = require 'ffi'
local function initBleu(C)
    local bleu = torch.class('Bleu')

    function Bleu:__init(pad, eos)
        self.stat = ffi.new('bleu_stat')
        self.pad = pad or 2
        self.eos = eos or 3
        self:reset()
    end

    function Bleu:reset(oneinit)
        self.nsent = 0
        if oneinit then
            C.bleu_one_init(self.stat)
        else
            C.bleu_zero_init(self.stat)
        end
        return self
    end

    function Bleu:add(ref, pred)
        local nogc = {ref, pred} -- keep pointers to prevent gc

        local reflen, refdata = ref:size(1), ref:data()
        local predlen, preddata = pred:size(1), pred:data()

        C.bleu_add(
            self.stat, reflen, refdata, predlen, preddata, self.pad, self.eos)
        self.nsent = self.nsent + 1

        table.unpack(nogc)
        return self
    end

    function Bleu:precision(n)
        local function ratio(a, b)
            return tonumber(b) > 0 and (tonumber(a) / tonumber(b)) or 0
        end
        local precision = {
            ratio(self.stat.match1, self.stat.count1),
            ratio(self.stat.match2, self.stat.count2),
            ratio(self.stat.match3, self.stat.count3),
            ratio(self.stat.match4, self.stat.count4),
        }
        return n and precision[n] or precision
    end

    function Bleu:brevity()
        local r = tonumber(self.stat.reflen)/tonumber(self.stat.predlen)
        return math.min(1, math.exp(1 - r))
    end

    function Bleu:score()
        local psum = 0
        for _, p in ipairs(self:precision()) do
            psum = psum + math.log(p)
        end

        return self:brevity() * math.exp(psum / 4) * 100
    end

    function Bleu:results()
        return {
            bleu = self:score(),
            precision = self:precision(),
            brevPenalty = self:brevity(),
            totalSys = tonumber(self.stat.predlen),
            totalRef = tonumber(self.stat.reflen),
        }
    end

    function Bleu:resultString()
        local r = self:results()
        local str = string.format('BLEU4 = %.2f, ', r.bleu)
        local precs = {}
        for i = 1, 4 do
            precs[i] = string.format('%.1f', r.precision[i] * 100)
        end
        str = str .. table.concat(precs, '/')
        str = str .. string.format(
            ' (BP=%.3f, ratio=%.3f, sys_len=%d, ref_len=%d)',
            r.brevPenalty, r.totalRef / r.totalSys, r.totalSys, r.totalRef
        )
        return str
    end

    return function(...) return Bleu(...) end
end

return initBleu
