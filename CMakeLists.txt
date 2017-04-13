# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenMP)

SET(CMAKE_CXX_FLAGS "-std=c++11 -Ofast")
IF(OpenMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

# C++ library
IF(APPLE)
  SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
ENDIF(APPLE)
FILE(GLOB CPPSRC fairseq/clib/*.cpp)
ADD_LIBRARY(fairseq_clib SHARED ${CPPSRC})
INSTALL(TARGETS fairseq_clib DESTINATION "${ROCKS_LIBDIR}")

# Lua library
INSTALL(DIRECTORY "fairseq" DESTINATION "${ROCKS_LUADIR}" FILES_MATCHING PATTERN "*.lua")

# Scripts and main executable
FOREACH(SCRIPT preprocess train tofloat generate generate-lines score optimize-fconv help)
  INSTALL(FILES "${SCRIPT}.lua" DESTINATION "${ROCKS_LUADIR}/fairseq/scripts")
ENDFOREACH(SCRIPT)
INSTALL(FILES "run.lua" DESTINATION "${ROCKS_BINDIR}" RENAME "fairseq")
