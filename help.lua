-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.
--
--[[
--
-- List available scripts.
--
--]]

local dir = require 'pl.dir'
local path = require 'pl.path'

local scriptdir = path.abspath(path.dirname(debug.getinfo(1).source:sub(2)))
local tools = {}
local maxbase = 0
for _, file in pairs(dir.getfiles(scriptdir, '*.lua')) do
    local base, _ = path.splitext(path.basename(file))

    local f = io.open(file)
    local source = f:read("*all")
    f:close()
    -- First sentence of first multi-line comment block is regarded as a brief
    -- description
    local m = source:gsub('\n', ' '):match('%-%-%[%[.*%-%-%]%]')
    local description = m:match('(%w+[^%.]*)%.')

    table.insert(tools, {base = base, description = description})
    if #base > maxbase then
        maxbase = #base
    end
end

print('Available tools:')
for i, tool in ipairs(tools) do
    io.stdout:write('  ')
    io.stdout:write(tool.base)
    for j = #tool.base, maxbase + 2 do
        io.stdout:write(' ')
    end
    print(tool.description)
end
