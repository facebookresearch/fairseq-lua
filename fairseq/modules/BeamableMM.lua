-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- This module specialize MM for beam decoding by attention modules.
-- It leverage the fact that the source-side of the input is replicated beam
-- times and that the target-side of the input is of width one. This layer speed
-- up inference by replacing the inputs {(bsz x 1 x nhu), (bsz x sz2 x nhu)}
-- with smaller inputs {(bsz/beam x beam x nhu), (bsz/beam x sz2 x nhu)}
--
--]]

local BeamableMM, parent = torch.class('nn.BeamableMM', 'nn.MM')

function BeamableMM:__init(...)
    parent.__init(self, ...)
    self.beam = 0
end

function BeamableMM:updateOutput(input)
    if not(self.train == false)     -- test mode
        and (self.beam > 0)         -- beam size is set
        and (input[1]:dim() == 3)   -- only support batched inputs
        and (input[1]:size(2) == 1) -- single time step update
    then
        local bsz, beam = input[1]:size(1), self.beam

        -- bsz x 1 x nhu --> bsz/beam x beam x nhu
        local in1 = input[1]:select(2, 1):unfold(1, beam, beam):transpose(3, 2)
        -- bsz x sz2 x nhu --> bsz/beam x sz2 x nhu
        local in2 = input[2]:unfold(1, beam, beam):select(4, 1)
        -- use non batched operation if bsz = beam
        if in1:size(1) == 1 then in1, in2 = in1[1], in2[1] end

        -- forward and restore correct size
        parent.updateOutput(self, {in1, in2})
        self.output = self.output:view(bsz, 1, -1)
        return self.output

    else
        return parent.updateOutput(self, input)
    end
end

function BeamableMM:setBeamSize(beam)
    self.beam = beam or 0
end
