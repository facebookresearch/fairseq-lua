-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- This module contains 2 modules one for train mode, one for evaluate
-- (test) mode
--
--]]

local TrainTestLayer, parent = torch.class('nn.TrainTestLayer', 'nn.Container')

function TrainTestLayer:__init(trainModule, evalModule, onTrain, onEvaluate)
    parent.__init(self)
    self.modules[1] = trainModule
    self.modules[2] = evalModule
    self.onTrain = onTrain
    self.onEvaluate = onEvaluate
    self.train = true
end

function TrainTestLayer:evaluate()
    if self.train then
        parent.evaluate(self)
        self.onEvaluate(self.modules[1], self.modules[2])
    end
end

function TrainTestLayer:training()
    if not self.train then
        parent.training(self)
        self.onTrain(self.modules[1], self.modules[2])
    end
end

function TrainTestLayer:updateOutput(input)
    local i = self.train and 1 or 2
    self.output:set(self.modules[i]:updateOutput(input))
    return self.output
end

function TrainTestLayer:updateGradInput(input, gradOutput)
    assert(self.train, 'updateGradInput only in training mode')
    self.gradInput:set(self.modules[1]:updateGradInput(input, gradOutput))
    return self.gradInput
end

function TrainTestLayer:accGradParameters(input, gradOutput, scale)
    assert(self.train, 'accGradParameters only in training mode')
    self.modules[1]:accGradParameters(input, gradOutput, scale)
end

function TrainTestLayer:zeroGradParameters()
    assert(self.train, 'zeroGradParameters only in training mode')
    self.modules[1]:zeroGradParameters()
end

function TrainTestLayer:updateParameters(lr)
    assert(self.train, 'updateParameters only in training mode')
    self.modules[1]:updateParameters(lr)
end

function TrainTestLayer:parameters()
    assert(self.train, 'TrainTestLayer support parameters in train mode')
    return self.modules[1]:parameters()
end

function TrainTestLayer:__tostring__()
    local fmt = 'nn.TrainTestLayer [ %s ; %s ]'
    return fmt:format(tostring(self.modules[1]), tostring(self.modules[2]))
end
