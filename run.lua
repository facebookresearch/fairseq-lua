-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.
--
--[[
--
-- A "master" script that can launch a given script.
--
--]]

if #arg > 0 then
    if arg[1] == '--help' or arg[1] == '-h' or arg[2] == '-?' then
        arg[1] = 'help'
    end
    require('fairseq.scripts.' .. table.remove(arg, 1))
else
    print('Usage: fairseq <tool> [options]')
end
