package = 'fairseq'
version = 'scm-1'
source = {
  url = 'git://github.com/facebookresearch/fairseq',
  tag = 'master',
}
description = {
  summary = 'Facebook AI Research Sequence-to-Sequence Toolkit',
  homepage = 'https://github.com/facebookresearch/fairseq',
  license = 'BSD 3-clause',
}
dependencies = {
  'argcheck',
  'cudnn',
  'cunn',
  'lua-cjson',
  'nccl',
  'nn',
  'nngraph',
  'penlight',
  'rnnlib',
  'tbc',
  'tds',
  'threads',
  'torch >= 7.0',
  'torchnet',
  'torchnet-sequential',
  'visdom',
}
build = {
  type = "cmake",
  variables = {
    CMAKE_BUILD_TYPE="Release",
    ROCKS_PREFIX="$(PREFIX)",
    ROCKS_LUADIR="$(LUADIR)",
    ROCKS_LIBDIR="$(LIBDIR)",
    ROCKS_BINDIR="$(BINDIR)",
    CMAKE_PREFIX_PATH="$(LUA_BINDIR)/..",
  }
}
