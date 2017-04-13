-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[ A plain implementation of Nesterov's momentum
Implements Nesterov's momentum using the simplified
formulation of https://arxiv.org/pdf/1212.0901.pdf
ARGS:
- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.momentum`          : momentum
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)
RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update
(Yann Dauphin, 2016)
]]

local function nag(opfunc, x, config, state)

   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local l2 = config.l2 or 0
   local mom = config.momentum or 0
   state.evalCounter = state.evalCounter or 0

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   if not state.dfdx then
      state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
   end

   -- (2) weight decay
   if l2 ~= 0 then
      dfdx:add(l2, x)
   end

   -- (3) apply update
   x:add(mom*mom, state.dfdx):add(-(1 + mom) * lr, dfdx)

   -- (4) apply momentum
   state.dfdx:mul(mom):add(-lr, dfdx)

   -- (5) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}

end

return nag
