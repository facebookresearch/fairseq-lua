-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--

local nn = require 'nn'
local ffi = require 'ffi'
local bleu = require 'fairseq.clib.bleu'
local so = package.searchpath('libfairseq_clib', package.cpath)
local cdef =[[
    void ctopk2d(float* top, long* idx, int k, float* values, int len, int n);
    void logsoftmax2d(float* input, float* output, int sz1, int sz2);

    typedef struct
    {
        size_t reflen;
        size_t predlen;
        size_t match1;
        size_t count1;
        size_t match2;
        size_t count2;
        size_t match3;
        size_t count3;
        size_t match4;
        size_t count4;
    } bleu_stat;

    void bleu_zero_init(bleu_stat* self);
    void bleu_one_init(bleu_stat* self);

    void bleu_add(
        bleu_stat* stat, size_t reflen, int* ref, size_t predlen, int* pred,
        int pad, int eos);
]]

ffi.cdef(cdef)
local C = ffi.load(so)

local function topk(top, ind, val, k)
    assert(val:dim() == 2 and k <= val:size(2))
    if not(val:type() == 'torch.FloatTensor' and val:isContiguous()) then
        -- use torch for GPU, non contiguous tensors
        return torch.topk(top, ind, val, k, 2, true, true)
    else
        top, ind = top or torch.FloatTensor(), ind or torch.LongTensor()
        local len, n = val:size(2), val:size(1)
        top:resize(n, k)
        ind:resize(n, k)
        assert(top:isContiguous() and ind:isContiguous())
        C.ctopk2d(top:data(), ind:data(), k, val:data(), len, n)
    end
    return top, ind
end

local function logsoftmax()
    local output = torch.FloatTensor()
    local lsm = nn.LogSoftMax()

    return function(input)
        if input:type()=='torch.FloatTensor'
            and input:dim()==2
            and input:isContiguous()
        then
            output:resizeAs(input)
            C.logsoftmax2d(
                input:data(), output:data(), input:size(1), input:size(2))
            return output
        else
            lsm:type(input:type())
            return lsm:updateOutput(input)
        end
    end
end

return {
    topk = topk,
    logsoftmax = logsoftmax,
    bleu = bleu(C),
}
