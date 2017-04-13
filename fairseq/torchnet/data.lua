-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- A data provider for the NMT training code. This module provides the following
-- things:
--
--   * `data.makeDataIterators`: This is a pretty generic function for setting
--   up tnt.DatasetIterators for training and testing. Most aspects of the
--   pipeline are configured via function arguments (`init`, `samplesize`,
--   `merge`).  Check out the comment at the function for more detailed
--   documentation about the functions and a full description of the data
--   provider pipeline.
--
--   * `data.loadCorpus`: This function calls data.makeDataIterators with
--   specific arguments for the NMT setup. For other datasets where the NMT data
--   pipeline might be useful, you will likely want to roll your own version of
--   this function.
--
--   * Small utility functions that are useful when dealing with sequential
--   data, e.g. `data.makeInput` or `data.mergeWithPad`.
--
--]]

local tnt = require 'torchnet'
local argcheck = require 'argcheck'
require 'fairseq.torchnet'

local data = {}

function data.makeInput(T, eos)
    -- The training input is the previous token. 'eos' is used as the start
    -- symbol.
    local input = T.new():resize(T:size(1))
    input[1] = eos
    input:narrow(1, 2, T:size(1)-1):copy(T:narrow(1, 1, T:size(1)-1))
    return input
end

function data.makePositions(T, pad)
    return torch.range(pad + 1, pad + T:size(1)):typeAs(T)
end

function data.makeReverse(T)
    return T:index(1, torch.range(T:size(1), 1, -1):long())
end

local function padEnd(res, T, size, pad)
    local oldsize = T:size(1)
    res:resize(size)
    res:narrow(1, 1, oldsize):copy(T)
    res:narrow(1, oldsize + 1, size - oldsize):fill(pad)
    return res
end

local function padBegin(res, T, size, pad)
    local oldsize = T:size(1)
    res:resize(size)
    res:narrow(1, size - oldsize + 1, oldsize):copy(T)
    res:narrow(1, 1, size - oldsize):fill(pad)
    return res
end

local function doMerge(values, pad, padf)
    local maxsize = 0
    for _, item in ipairs(values) do
        maxsize = math.max(maxsize, item:size(1))
    end
    assert(values and #values > 0)
    local res = values[1].new():resize(maxsize, #values)
    for k, item in ipairs(values) do
        local oldsize = item:size(1)
        if oldsize ~= maxsize then
            padf(res:narrow(2, k, 1):squeeze(), item, maxsize, pad)
        else
            res:narrow(2, k, 1):copy(item)
        end
    end
    return res
end

function data.mergeWithPad(values, pad)
    return doMerge(values, pad, padEnd)
end

function data.mergeWithBeginPad(values, pad)
    return doMerge(values, pad, padBegin)
end

-- Returns a function for bucketing a tnt.IndexedDataset with
-- BucketShuffledDataset.
-- This accesses the dataset index directly to determine the size of a sample
-- rather than calling get() to create a new sample (which might be costly)
function data.sizeFromIndex(field_name)
    local dataset
    local fieldNum
    local resamples = {}
    return function(ds, idx)
        -- Find the underlying tnt.IndexedDataset
        if not dataset then
            dataset = ds
            while not torch.isTypeOf(dataset, 'tnt.IndexedDataset') do
                if torch.isTypeOf(dataset, 'tnt.ResampleDataset') then
                    local sampler = dataset.__sampler
                    local source = dataset.__dataset
                    table.insert(resamples, function(idx)
                        return sampler(source, idx)
                    end)
                end
                if dataset.dataset then
                    dataset = dataset.dataset
                elseif dataset.__dataset then
                    dataset = dataset.__dataset
                else
                    error("Cannot find the IndexedDataset.")
                end
            end
        end
        -- Get the field index
        if not fieldNum then
            for i, field in ipairs(dataset.__fields) do
                if field.name == field_name then
                    fieldNum = i
                    break
                end
            end
        end

        for _, resample in ipairs(resamples) do
            idx = resample(idx)
        end
        local index = dataset.__fields[fieldNum].data
        return index.datoffsets[idx+1] - index.datoffsets[idx]
    end
end

-- Returns a function for double-bucketing, assuming that
-- target sentence length / bucketres < 2^12.
-- With a bucket resolution of 1, this effectively sorts the training data by
-- source length and uses the target length as a tie braker. This should be
-- efficient since nearby sentences will have similar target length, which
-- produces less padding. Also, source sentences will usually be of equal length
-- so that encoders shouldn't be only minimally affected.
function data.doubleSizeFromIndex(field_name1, field_name2)
    local getSize1 = data.sizeFromIndex(field_name1)
    local getSize2 = data.sizeFromIndex(field_name2)
    return function(ds, i)
        return 2^12 * getSize1(ds, i) + getSize2(ds,i)
    end
end

data.getTargetVocabFromAlignment = argcheck{
    {name='dictsize', type='number'},
    {name='unk', type='number'},
    {name='aligndict', type='tnt.IndexedDatasetReader'},
    {name='set', type='string'},
    {name='source', type='torch.IntTensor'},
    {name='target', type='torch.IntTensor'},
    {name='nmostcommon', type='number'},
    {name='topnalign', type='number'},
    {name='freqthreshold', type='number'},
    {name='tgtset', type='torch.IntTensor', opt=true},
    {name='tgtsetidx', type='number', default=1},
    call = function(dictsize, unk, aligndict, set, source, target, nmostcommon,
        topnalign, freqthreshold, tgtset, tgtsetidx)
        local tgtvocab = {}
        local tgtvocabmap = {}
        local tgtvocabsize = 0
        tgtset = tgtset or torch.IntTensor(dictsize):zero()
        tgtsetdix = tgtsetidx or 1

        local function addtoken(t)
            if t <= dictsize then
                if tgtset[t] < tgtsetidx then
                    tgtset[t] = tgtsetidx
                    table.insert(tgtvocab, t)
                    tgtvocabsize = tgtvocabsize + 1
                    tgtvocabmap[t] = tgtvocabsize
                end
            end
        end

        local flatSource = source:view(-1)
        local flatTarget = target:view(-1)

        -- Add n most common words
        for tgtidx = 1, nmostcommon do
            addtoken(tgtidx)
        end

        if set == 'train' then
            -- Add words from the target, only for the trainset
            flatTarget:apply(function(x)
                if x > nmostcommon then
                    addtoken(x)
                end
            end)
        end

        -- Add words from the alignment
        local seen = {}
        for i = 1, flatSource:size(1) do
            local srcidx = flatSource[i]
            if not seen[srcidx] then
                local align = aligndict:get(srcidx)
                if align:dim() ~= 0 then
                    local n = math.min(topnalign, align:size(1))
                    for offset = 1, n do
                        if align[offset][2] >= freqthreshold then
                            addtoken(align[offset][1])
                        end
                    end
                end
            end
            seen[srcidx] = true
        end

        local stats = {
            covered = 0,
            unk = 0,
            -- The number words that are unk in the new vocabulary
            fakeunk = 0,
        }
        -- Remap target to use words from the new vocabulary
        flatTarget:apply(
            function(idx)
                if idx <= dictsize then
                    if tgtvocabmap[idx] then
                        if idx == unk then
                            stats.unk = stats.unk + 1
                        else
                            stats.covered = stats.covered + 1
                        end
                        return tgtvocabmap[idx]
                    else
                        stats.fakeunk = stats.fakeunk + 1
                        return tgtvocabmap[unk]
                    end
                else
                    stats.unk = stats.unk + 1
                    return tgtvocabmap[unk]
                end
            end
        )

        return torch.IntTensor(tgtvocab), stats
    end
}

local makeTestDataPipeline = argcheck{
    {name='dataset', type='tnt.Dataset'},
    {name='samplesize', type='function'},
    {name='merge', type='function'},
    {name='batchsize', type='number'},
    call = function(dataset, samplesize, merge, batchsize)
        return tnt.BucketBatchDataset{
            dataset = dataset,
            batchsize = batchsize,
            samplesize = samplesize,
            merge = merge,
            policy = 'include-last',
        }
    end
}

local makeTrainingDataPipeline = argcheck{
    {name='dataset', type='tnt.Dataset'},
    {name='samplesize', type='function'},
    {name='merge', type='function'},
    {name='batchsize', type='number'},
    call = function(dataset, samplesize, merge, batchsize)
        return tnt.ShuffleDataset{
            -- Sentence batching
            dataset = tnt.BatchDataset{
                batchsize = batchsize,
                merge = merge,
                -- Bucketing
                dataset = tnt.BucketSortedDataset{
                    samplesize = samplesize,
                    dataset = dataset,
                    shuffle = true,
                },
            },
        }
    end
}

local makeDataIterator = argcheck{
    {name='set', type='string'},
    {name='init', type='function'},
    {name='samplesize', type='function'},
    {name='merge', type='function'},
    {name='config', type='table'},
    {name='test', type='boolean'},
    {name='transform', type='function', opt=true},
    {name='seed', type='number', opt=true},
    call = function(set, init, samplesize, merge, config,
        test, transform, seed)
        local makePipeline = function()
            -- Note that we create init, samplesize and merge functions here,
            -- becase we want to provide thread locality for their upvalues.
            local params = {
                dataset = init(set)(),
                samplesize = samplesize(set),
                merge = merge(set),
                batchsize = config.batchsize,
            }
            if test then
                return makeTestDataPipeline(params)
            else
                local ds = makeTrainingDataPipeline(params)
                -- Attach a function to set the random seed. This dataset will
                -- live in a seprate thread, and this is a convenient way to
                -- initialize the RNG local to that thread.
                ds.setRandomSeed = function(self, seed)
                    torch.manualSeed(seed)
                end
                return ds
            end
        end

        local makeIterator = function()
            local it
            if config.nthread == 0 then
                it = tnt.DatasetIterator{
                    dataset = makePipeline(),
                }
            else
                it = tnt.ParallelDatasetIterator{
                    nthread = config.nthread,
                    init = function()
                        require 'torchnet'
                        tds = require 'tds'
                        require 'fairseq'
                        if seed then
                            torch.manualSeed(seed)
                        end
                    end,
                    closure = makePipeline,
                    ordered = true,
                }
            end

            -- Apply bptt truncation if needed. TruncatedDatasetIterator
            -- will split up mini-batches that exceed the maximum bptt value.
            if config.bptt > 0 then
                it = tnt.TruncatedDatasetIterator{
                    iterator = it,
                    maxsize = config.bptt,
                    dimension = 1,
                    fields = {'input', 'inputPos', 'target'},
                }
            end

            if config.maxbatch > 0 then
                it = tnt.TruncatedDatasetIterator{
                    iterator = it,
                    maxsizefn = function(sample)
                        local sl = sample.source:size(1)
                        local tl = sample.target:size(1)
                        local bsz = sample.target:size(2)
                        local len = math.max(sl, tl)
                        if len * bsz <= config.maxbatch then
                            return math.huge
                        end
                        repeat
                            bsz = math.floor(bsz / 2)
                        until len * bsz <= config.maxbatch or bsz <= 1
                        return math.max(1, bsz)
                    end,
                    dimension = 2,
                    fields = {'input', 'inputPos', 'target',
                        'source', 'sourcePos'},
                }
            end

            if config.nshards > 0 then
                it = tnt.ShardedDatasetIterator{
                    iterator = it,
                    nshards = config.nshards,
                    dimension = 2,
                    fields = {'input', 'inputPos', 'target',
                        'source', 'sourcePos'},
                }
            end

            if transform then
                it = tnt.DatasetIterator{
                    iterator = it,
                    transform = transform(set),
                }
            end

            return it
        end

        if config.nthread == 0 then
            return makeIterator()
        end

        -- To avoid computation of the target vocabulary in the main thread
        -- (as it can be expensive) offload it to a separate thread.
        return tnt.SingleParallelIterator{
            init = function()
                require 'torchnet'
                tds = require 'tds'
                require 'fairseq'
                if seed then
                    torch.manualSeed(seed)
                end
            end,
            closure = makeIterator,
        }
    end
}

-- This is a generic function that sets up data iterators for training and
-- testing using the *Pipeline functions defined above. The function arguments
-- will be called with the respective set name from `trainsets` or `testsets`
-- and are supposed to do return functions based on `set` that do the following:
--
--   * `init(set)` produces a function that returns a dataset that produces
--   input and target tensors as well as everything else needed for training.
--   Note that this dataset is expected to produce single samples.
--
--   * `samplesize(set)` produces a function that will be used to determine
--   the bucket that a sample belongs to. See the definition of
--   `BucketShuffleDataset` for more information about that.
--
--   * `merge(set)` produces a merge function for `tnt.BatchDataset`. This
--   function converts a mini-batch represented as a table of the format {key1 =
--   {value1_1, value1_2, ..., value1_bsz}, key2 = {value2_1, ...}} to a table
--   of the format {key_1 = batch_value1, key_2 = batch_value2} ...}.
--
--   * `transform(set)` produces an optional function to do additional
--   transformations
--
-- The iterators will be configured wih the following pipeline:
--
--   [     Source Dataset      ]       as returned from `init(set)()`)
--               |
--               V
--   [       Bucketing         ]       according to `samplesize(set)(ds, i)`
--               |
--               V
--   [     Mini-batching       ]       according to `merge(set)(fields)`
--               |
--               V
--   [    Batch splitting*      ]       according to `maxbatch`
--               |
--               V
--   [       Sharding*         ]       according to `nshards`
--               |
--               V
--   [       Transform*        ]       according to `transform(set)`
--               |
--               V
--   [ ParallelDatasetIterator ]       using `nthread`
--
-- For iterators on `testsets`, bucketing and mini-batching is done in a single
-- step (`BucketBatchDataset`) in order to eliminate the effect of padding on
-- validation and testing measurements.
--
data.makeDataIterators = argcheck{
    {name='trainsets', type='table', default={}},
    {name='testsets', type='table', default={}},
    {name='init', type='function'},
    {name='samplesize', type='function'},
    {name='merge', type='function'},
    {name='config', type='table'},
    {name='transform', type='function', opt=true},
    {name='seed', type='number', opt=true},
    call = function(trainsets, testsets, init, samplesize, merge,
        config, transform, seed)
        local trainits = {}
        for _, set in pairs(trainsets) do
            trainits[set] = makeDataIterator{
                set = set,
                init = init,
                samplesize = samplesize,
                merge = merge,
                test = false,
                config = config,
                transform = transform,
                seed = seed,
            }
        end

        local testits = {}
        for _, set in pairs(testsets) do
            testits[set] = makeDataIterator{
                set = set,
                init = init,
                samplesize = samplesize,
                merge = merge,
                test = true,
                config = config,
                transform = transform,
                seed = seed,
            }
        end

        return trainits, testits
    end
}

-- Calls `data.makeDataIterators` with functions specific to the NMT setup and
-- optionally wraps the resulting iterators in bptt-truncating iterators.
data.loadCorpus = argcheck{
    noordered=true,
    {name='config', type='table'},
    {name='trainsets', type='table', default={}},
    {name='testsets', type='table', default={}},
    call = function(config, trainsets, testsets)
        local srcext = string.format('.%s-%s.%s', config.sourcelang,
            config.targetlang, config.sourcelang)
        local tgtext = string.format('.%s-%s.%s', config.sourcelang,
            config.targetlang, config.targetlang)

        -- Make iterators for training and testing datasets. These will be
        -- tnt.ParallelDatasetIterators, running in a separate thread.
        local train, test = data.makeDataIterators{
            trainsets = trainsets,
            testsets = testsets,
            config = {
                batchsize = config.batchsize,
                nthread = config.ndatathreads,
                nshards = config.ngpus or 1,
                bptt = config.bptt or 0,
                maxbatch = config.maxbatch or 0,
            },
            seed = config.seed,

            -- Dataset initialization: load the IndexedDataset produced by
            -- preprocess.lua and wrap it in a tnt.TransformDataset which
            -- performs basic, per-sentence sample setup.
            init = function(set)
                local sourceField = set .. srcext
                local targetField = set .. tgtext
                return function()
                    local dataset = tnt.IndexedDataset{
                        fields = {sourceField, targetField},
                        path = config.datadir,
                    }
                    if set == 'train' and config.maxsourcelen
                        and config.maxsourcelen > 0 then
                        local tds = require 'tds'
                        local resample = tds.Vec()
                        local maxlen = config.maxsourcelen
                        -- XXX this is super slow, we should use the index
                        -- instead
                        for i = 1, dataset:size() do
                            local s = dataset:get(i)
                            if s[sourceField]:nElement() <= maxlen then
                                resample:insert(i)
                            end
                        end
                        print(string.format('| Restricted to maximum ' ..
                            'length %d: %d sentences remain (%.2f%%)',
                            maxlen, #resample,
                            100 * #resample / dataset:size()))
                        dataset = tnt.ResampleDataset{
                            dataset = dataset,
                            sampler = function(ds, i) return resample[i] end,
                            size = #resample,
                        }
                    end

                    if config.samplingpct and config.samplingpct < 1.0
                        and config.samplingpct > 0.0 then
                        dataset = tnt.ShuffleDataset{
                            dataset = dataset,
                            size = torch.ceil(
                                dataset:size() * config.samplingpct),
                            replacement = false,
                        }
                        print(string.format('| Sampled %.1f%% of dataset: ' ..
                            '%d sentences remain',
                            100 * config.samplingpct, dataset:size()))
                    end

                    return tnt.TransformDataset{
                        dataset = dataset,
                        -- Prepare source and target language data, add decoder
                        -- input and source position data
                        transform = function(sample, index)
                            local source = sample[sourceField]
                            local target = sample[targetField]
                            local input = data.makeInput(target,
                                config.dict:getEosIndex())
                            local inputPos = data.makePositions(input,
                                config.dict:getPadIndex())
                            local sourcePos = data.makePositions(source,
                                config.srcdict:getPadIndex())
                            return {
                                input = input,
                                inputPos = inputPos,
                                target = target,
                                source = source,
                                sourcePos = sourcePos,
                                index = index,
                            }
                        end,
                    }
                end
            end,

            -- Determine sample size for bucketing. Bucket sizes can be changed
            -- by returning a different sample size. The bucket resolution will
            -- be set to 1 in data.lua, so the size returned here is effectively
            -- the bucket of the given sample.
            samplesize = function(set)
                local sourceField = set .. srcext
                local targetField = set .. tgtext

                if set == 'train' then
                    return data.doubleSizeFromIndex(sourceField, targetField)
                else
                    return data.sizeFromIndex(sourceField)
                end
            end,

            -- Merge multiple sentences into a mini-batch
            merge = function(set)
                return function(fields)
                    local sample = {
                        -- Pad target language data at the end
                        input = data.mergeWithPad(
                            fields.input, config.dict:getPadIndex()
                        ),
                        inputPos = data.mergeWithPad(
                            fields.inputPos, config.dict:getPadIndex()
                        ),
                        target = data.mergeWithPad(
                            fields.target, config.dict:getPadIndex()
                        ),

                        -- Pad source data at the beginning
                        source = data.mergeWithBeginPad(
                            fields.source, config.srcdict:getPadIndex()
                        ),
                        sourcePos = data.mergeWithBeginPad(
                            fields.sourcePos, config.srcdict:getPadIndex()
                        ),

                        index = torch.IntTensor(fields.index),
                    }
                    return sample
                end
            end,

            transform = function(set)
                if not config.aligndict then
                    return function(sample) return sample end
                end

                return function(samples)
                    local tgtset = torch.IntTensor(config.dict:size()):zero()
                    for i, sample in ipairs(samples) do
                        sample.source = sample.source:contiguous()
                        sample.target = sample.target:contiguous()

                        -- Create a special target dictionary
                        sample.targetVocab, sample.targetVocabStats
                            = data.getTargetVocabFromAlignment(
                                config.dict:size(), config.dict:getUnkIndex(),
                                config.aligndict, set,
                                sample.source, sample.target,
                                config.nmostcommon, config.topnalign,
                                config.freqthreshold,
                                tgtset, i)
                    end
                    collectgarbage()
                    return samples
                end
            end,
        }

        return train, test
    end
}

return data
