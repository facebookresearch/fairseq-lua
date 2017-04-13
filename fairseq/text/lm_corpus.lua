-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- An utility class to ease loading of a language model corpus.
--
--]]

local tnt = require 'torchnet'
local argcheck = require 'argcheck'
local tokenizer = require 'fairseq.text.tokenizer'

local lmc = {}

local function makeDataPipeline(ds, batchsize, bptt)
    -- bptt batching
    return tnt.BatchDataset{
        -- Add targets
        dataset = tnt.TargetNextDataset{
            -- Place tensor in table
            dataset = tnt.TransformDataset{
                -- Batching across sentences
                dataset = tnt.SequenceBatchDataset{
                    dataset = ds,
                    batchsize = batchsize,
                    policy = 'skip-remainder',
                },
                transform = function(sample)
                    return {input = sample}
                end,
            },
        },
        batchsize = bptt,
        merge = function(sample)
            -- Merge target vectors into a tensor, keep input as a table
            local targets = sample.target
            sample.target = targets[1].new()
            torch.cat(sample.target, targets)
            return sample
        end,
    }
end

lmc.iteratorFromIndexed = argcheck{
    {name='indexfilename', type='string'},
    {name='datafilename', type='string'},
    {name='batchsize', type='number'},
    {name='bptt', type='number'},
    call = function(indexfilename, datafilename, batchsize, bptt)
        return tnt.ParallelDatasetIterator{
            nthread = 1,
            init = function()
                require 'torchnet'
            end,
            closure = function()
                local ds = tnt.FlatIndexedDataset{
                    indexfilename = indexfilename,
                    datafilename = datafilename,
                }
                return makeDataPipeline(ds, batchsize, bptt)
            end,
        }
    end
}

lmc.iteratorFromText = argcheck{
    {name='filename', type='string'},
    {name='dict', type='Dictionary'},
    {name='batchsize', type='number'},
    {name='bptt', type='number'},
    call = function(filename, dict, batchsize, bptt)
        -- XXX Won't scale to large datasets
        local data, _ = tokenizer.tensorize(filename, dict)
        return tnt.ParallelDatasetIterator{
            nthread = 1,
            init = function()
                require 'torchnet'
            end,
            closure = function()
                local ds = tnt.TableDataset(data.words:totable())
                collectgarbage()
                return makeDataPipeline(ds, batchsize, bptt)
            end,
        }
    end
}

lmc.loadTextCorpus = argcheck{
    {name='trainfilename', type='string'},
    {name='validfilename', type='string', opt=true},
    {name='testfilename', type='string', opt=true},
    {name='batchsize', type='number'},
    {name='bptt', type='number'},
    {name='dict', type='Dictionary', opt=true},
    call = function(trainfilename, validfilename, testfilename, batchsize,
        bptt, dict)
        if not dict then
            dict = tokenizer.buildDictionary{
                filename = trainfilename,
                threshold = 0,
            }
        end

        local train, valid, test = nil, nil, nil
        if trainfilename then
            train = lmc.iteratorFromText{
                filename = trainfilename,
                dict = dict,
                batchsize = batchsize,
                bptt = bptt,
            }
        end
        if validfilename then
            valid = lmc.iteratorFromText{
                filename = validfilename,
                dict = dict,
                batchsize = batchsize,
                bptt = bptt,
            }
        end
        if testfilename then
            test = lmc.iteratorFromText{
                filename = testfilename,
                dict = dict,
                batchsize = batchsize,
                bptt = bptt,
            }
        end

        return {
            dict = dict,
            train = train,
            valid = valid,
            test = test,
        }
    end
}

lmc.loadBinarizedCorpus = argcheck{
    {name='trainprefix', type='string'},
    {name='validprefix', type='string', opt=true},
    {name='testprefix', type='string', opt=true},
    {name='dictfilename', type='string', opt=true},
    {name='batchsize', type='number'},
    {name='bptt', type='number'},
    call = function(trainprefix, validprefix, testprefix, dictfilename,
        batchsize, bptt)

        local dict = nil
        if dictfilename then
            dict = torch.load(dictfilename)
        end

        local train, valid, test = nil, nil, nil
        if trainprefix then
            train = lmc.iteratorFromIndexed{
                indexfilename = trainprefix .. '.idx',
                datafilename = trainprefix .. '.bin',
                dict = dict,
                batchsize = batchsize,
                bptt = bptt,
            }
        end
        if validprefix then
            valid = lmc.iteratorFromIndexed{
                indexfilename = validprefix .. '.idx',
                datafilename = validprefix .. '.bin',
                dict = dict,
                batchsize = batchsize,
                bptt = bptt,
            }
        end
        if testprefix then
            test = lmc.iteratorFromIndexed{
                indexfilename = testprefix .. '.idx',
                datafilename = testprefix .. '.bin',
                dict = dict,
                batchsize = batchsize,
                bptt = bptt,
            }
        end

        return {
            dict = dict,
            train = train,
            valid = valid,
            test = test,
        }
    end
}

lmc.binarizeCorpus = argcheck{
    {name='files', type='table'},
    {name='dict', type='Dictionary'},
    call = function(files, dict)
        local res = {}
        for _, f in ipairs(files) do
            local r = tokenizer.binarize{
                filename = f.src,
                dict = dict,
                indexfilename = f.dest .. '.idx',
                datafilename = f.dest  .. '.bin',
            }
            table.insert(res, r)
            collectgarbage()
        end
        return res
    end
}

return lmc
