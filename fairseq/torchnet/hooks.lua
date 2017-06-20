-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Common torchnet engine hooks. The general format for functions declared
-- here is to return a function that can be used as a torchnet engine hook (i.e.
-- it takes a single 'state' parameter).
--
-- NOTE: If a hook requires persistent data, it should be stored in the state
-- rather than in the hook's context. Otherwise, this data will not persist
-- across training restarts.
--
--]]

require 'math'
local tnt = require 'torchnet'
local argcheck = require 'argcheck'
local json = require 'cjson'
local pltablex = require 'pl.tablex'
local plpath = require 'pl.path'
local utils = require 'fairseq.utils'
local clib = require 'fairseq.clib'

local cuda = utils.loadCuda()

local function printStats(d)
    print("")
    local new_d = {}
    for k, v in pairs(d) do
        if type(v) == "number" and (v ~= v or v == v + 1) then
            if v ~= v then
                new_d[k] = "NaN"
            else
                new_d[k] = "Inf"
            end
            -- Exit on NaNs
            local error_d = {}
            error_d["error"] = true
            error_d[k] = new_d[k]
            print("json_stats: " .. json.encode(error_d))
            os.exit(-1)
        else
            new_d[k] = v
        end
    end
    print("json_stats: " .. json.encode(new_d))
end

local hooks = {}

hooks.call = function(funcs)
    return function(state)
        for _, f in ipairs(funcs) do
            f(state)
        end
    end
end

hooks.computeSampleStats = argcheck{
    {name='dict', type='Dictionary'},
    call = function(dict)
        local padIndex = dict:getPadIndex()

        return function(state)
            -- Gather basic sample statistics. The number of sentences in the
            -- sample (bsz) might vary between batches and some models may need
            -- to be modified if this changs. `sample.ntokens` represents the
            -- number of non-padding tokens in the sample and can be used for
            -- accumulating total average loss, for example.
            for _, sample in ipairs(state.samples) do
                sample.size = sample.target:nElement()
                sample.bptt = sample.target:size(1)
                sample.bsz = sample.target:size(2)
                sample.ntokens = sample.target:ne(padIndex):sum()
                sample.sourcelen = sample.source:size(1)
                if sample.targetVocab then
                    if not state.dictstats then
                        state.dictstats = {
                            size = 0,
                            n = 0,
                            covered = 0,
                            unk = 0,
                            fakeunk = 0,
                        }
                    end
                    state.dictstats.size =
                        state.dictstats.size + sample.targetVocab:size(1)
                    state.dictstats.n = state.dictstats.n + 1
                    local total =
                        sample.targetVocabStats.covered +
                        sample.targetVocabStats.unk +
                        sample.targetVocabStats.fakeunk
                    state.dictstats.covered =
                        state.dictstats.covered +
                        sample.targetVocabStats.covered / total
                    state.dictstats.unk =
                        state.dictstats.unk +
                        sample.targetVocabStats.unk / total
                    state.dictstats.fakeunk =
                        state.dictstats.fakeunk +
                        sample.targetVocabStats.fakeunk / total
                end
            end
        end
    end
}

hooks.shuffleData = argcheck{
    {name='baseSeed', type='number'},
    call = function(baseSeed)
        return function(state)
            local iterator = state.iterator
            iterator:exec('setRandomSeed', baseSeed + 1024 * state.epoch)
            iterator:exec('resampleInBuckets')         -- Bucket level
            iterator:exec('resample')                  -- Mini-batches
        end
    end
}

hooks.saveStateHook = argcheck{
    {name='engine', type='tnt.Engine'},
    {name='path', type='string'},
    call = function(engine, path)
        return function(state)
            -- For epoch states the file name comes as a template
            -- where you can encode the epoch number
            local curpath = string.format(path, state.epoch)
            print(string.format('| epoch %03d | %07d updates | %07d epoch updates | path %s',
                state.epoch, state.t, state.epoch_t, curpath))
            engine:saveState{
                path = curpath,
                state = state
            }
        end
    end
}

hooks.updateMeters = argcheck{
    {name='lossMeter', type='tnt.AverageValueMeter'},
    {name='timeMeter', type='tnt.TimeMeter'},
    call = function(lossMeter, timeMeter)
        return function(state)
            lossMeter:add(state.loss / state.ntokens, state.ntokens)
            timeMeter:incUnit()
        end
    end
}

hooks.runTest = argcheck{
    {name='engine', type='tnt.Engine'},
    call = function(engine)
        return function(state, iterator, meter)
            engine.hooks.onForwardCriterion = function(state)
                meter:add(state.loss / state.ntokens, state.ntokens)
            end
            engine:test(iterator)
        end
    end
}

hooks.runGeneration = argcheck{
    {name='model', type='Model'},
    {name='dict', type='Dictionary'},
    {name='generate', type='function'},
    {name='outfile', type='string', opt=true},
    {name='srcdict', type='Dictionary', opt=true},
    call = function(model, dict, generate, outfile, srcdict)
        local computeSampleStats = hooks.computeSampleStats(dict)
        local eos = dict:getSymbol(dict:getEosIndex())
        local unkidx = dict:getUnkIndex()
        local unk = dict:getSymbol(unkidx)
        -- Select unknown token for reference that can't be produced by the
        -- model so that the hypotheses can be scored correctly.
        local runk = unk
        repeat
            runk = string.format('<%s>', runk)
        until dict:getIndex(runk) == dict:getUnkIndex()

        return function(iterator)
            local scorer = clib.bleu(dict:getPadIndex(), dict:getEosIndex())
            local fp = outfile and io.open(outfile, 'a')
            local targetBuf = torch.IntTensor()
            for samples in iterator() do
                -- We don't shard generation
                computeSampleStats({samples = samples})
                local sample = samples[1]
                local hypos, scores, attns, _ = generate(model, sample)
                local targetTT = sample.target:t()
                local targetT = targetBuf:resizeAs(targetTT):copy(targetTT)
                local beam = #hypos / sample.bsz
                for i = 1, sample.bsz do
                    local hindex = (i - 1) * beam + 1
                    local hypo = hypos[hindex]
                    if fp then
                        local refString = dict:getString(targetT[i])
                        :gsub(eos .. '.*', '') -- reference may contain padding
                        :gsub(unk, runk)

                        local hypoString = dict:getString(hypo):gsub(eos, '')
                        -- Write out source/target/hypo/attention in text form
                        local index = sample.index[i]
                        if srcdict then
                            local seos = srcdict:getSymbol(
                                srcdict:getEosIndex()
                            )
                            local sourceString =
                                srcdict:getString(sample.source:t()[i])
                                :gsub(seos, '')
                            fp:write(string.format(
                                'S-%d\t%s\n', index, sourceString
                            ))
                        end

                        fp:write(string.format('T-%d\t%s\n', index, refString))
                        fp:write(string.format(
                            'H-%d\t%f\t%s\n', index, scores[hindex], hypoString
                        ))
                        local _, maxattns = torch.max(attns[hindex], 2)
                        fp:write(string.format(
                            'A-%d\t%s\n', index,
                            table.concat(maxattns:squeeze(2):totable(), ' ')
                        ))
                    end
                    local ref = targetT[i]
                    ref:apply(function(x)
                        return x == unkidx and -unkidx or x
                    end)
                    scorer:add(ref, hypo:int())
                end
            end

            if fp then
                fp:close()
            end
            return scorer:results(), scorer:resultString()
        end
    end
}

hooks.onCheckpoint = argcheck{
    -- Pass all the arguments!
    {name='engine', type='tnt.Engine'},
    {name='config', type='table'},
    {name='lossMeter', type='tnt.Meter'},
    {name='timeMeter', type='tnt.TimeMeter'},
    {name='runTest', type='function'},
    {name='testsets', type='table'},
    {name='runGeneration', type='function', opt=true},
    {name='gensets', type='table', opt=true},
    {name='annealing', type='boolean', opt=false},
    {name='earlyStopping', type='boolean', opt=false},
    call = function(engine, config, lossMeter, timeMeter,
        runTest, testsets, runGeneration, gensets, annealing, earlyStopping)
        return function(state)
            -- Init hook state
            if not state._onCheckpoint then
                state._onCheckpoint = {
                    isAnnealing = false,
                    prevvalloss = nil,
                    bestvalloss = nil,
                }
            end

            local logPrefix = string.format(
                '| checkpoint %03d | epoch %03d | %07d updates',
                state.checkpoint, state.epoch, state.t
            )
            local stats = {}

            if state.dictstats then
                stats['avg_dict_size'] =
                    state.dictstats.size / state.dictstats.n
            end

            -- Log checkpoint meter stats
            local cptime = timeMeter.n * timeMeter:value()
            local statsstr = string.format(
                '%s | s/checkpnt %7d | words/s %7d | lr %02.6f',
                logPrefix, cptime, lossMeter.n / cptime,
                state.optconfig.learningRate * config.lrscale)
            if state.dictstats then
                statsstr = statsstr .. string.format(
                    ' | avg_dict_size %.2f',
                    state.dictstats.size / state.dictstats.n)
            end
            print(statsstr)
            stats['secspercp'] = cptime
            stats['wordspersec'] = lossMeter.n / cptime
            stats['current_lr'] = state.optconfig.learningRate * config.lrscale

            local loss = lossMeter:value() / math.log(2)
            local ppl = math.pow(2, loss)
            print(string.format(
                '%s | trainloss %8.2f | train ppl %8.2f',
                logPrefix, loss, ppl)
            )
            stats['trainloss'] = loss
            stats['trainppl'] = ppl

            -- Evaluate model on test sets
            local meter = tnt.AverageValueMeter()
            local str2print = logPrefix
            for name, set in pairs(testsets) do
                meter:reset()
                runTest(state, set, meter)
                local loss = meter:value() / math.log(2)
                local ppl = math.pow(2, loss)
                str2print = string.format('%s | %sloss %8.2f | %s ppl %8.2f',
                    str2print, name, loss, name, ppl)
                stats[name .. 'ppl'] = ppl
                stats[name .. 'loss'] = loss
            end
            print(str2print)
            io.stdout:flush()

            -- Run generation and scoring to obtain BLEU scores
            if runGeneration then
                local str2print = logPrefix
                for name, set in pairs(gensets) do
                    local result, _ = runGeneration(set)
                    str2print = string.format(
                        '%s | %sbleu %8.2f | %s BP  %8.2f',
                        str2print, name, result.bleu, name, result.brevPenalty
                    )
                    stats[name .. 'bleu'] = result.bleu
                    stats[name .. 'bp'] = result.brevPenalty
                end
                print(str2print)
                io.stdout:flush()
            end

            if config.log then
                local stateDump = {
                    checkpoint = state.checkpoint,
                    epoch = state.epoch,
                    t = state.t,
                }
                local logDump = pltablex.merge(config, stats, stateDump, true)
                -- Remove dictionaries from JSON log
                logDump.dict = nil
                logDump.srcdict = nil
                logDump.aligndict = nil
                logDump.critWeights = nil
                printStats(logDump)
            end


            local valloss = stats['validloss']

            -- Save model and best model
            if not config.nosave then
                local modelpath = plpath.join(config.savedir,
                    string.format('model_epoch%d.th7', state.epoch))
                if utils.retry(3, engine.saveModel, engine, modelpath) then
                    print(string.format(
                        '%s | saved model to %s', logPrefix, modelpath))
                end
                if valloss
                    and (not state._onCheckpoint.bestvalloss
                        or valloss < state._onCheckpoint.bestvalloss) then
                    local bestmodelpath = plpath.join(config.savedir,
                        'model_best.th7')
                    if utils.retry(3, engine.saveModel, engine, bestmodelpath)
                      then
                        print(string.format(
                            '%s | saved new best model to %s', logPrefix,
                            bestmodelpath))
                    end
                    state._onCheckpoint.bestvalloss = valloss
                end

            end
            io.stdout:flush()

            -- Early stopping
            local stopTraining = false
            if earlyStopping and state._onCheckpoint.prevvalloss
                and state.epoch >= config.minepochtoanneal
                and state._onCheckpoint.prevvalloss < valloss then
                stopTraining = true
            end

            -- Learning rate annealing
            if annealing and state.epoch >= config.minepochtoanneal
                and (state._onCheckpoint.isAnnealing
                or (state._onCheckpoint.prevvalloss
                    and state._onCheckpoint.prevvalloss <= valloss)) then
                state.optconfig.learningRate =
                    state.optconfig.learningRate / config.lrshrink
                if (config.annealing_type == nil or -- default behavior
                    config.annealing_type == 'fast') then
                    state._onCheckpoint.isAnnealing = true
                end
            else
                state._onCheckpoint.prevvalloss = valloss
            end
            if annealing and state.optconfig.learningRate < config.minlr then
                stopTraining = true
            end

            if stopTraining then
                state.epoch = state.maxepoch
            end
        end
    end
}

return hooks
