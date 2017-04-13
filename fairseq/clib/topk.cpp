/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>    // std::partial_sort_copy
#include <vector>       // std::vector

typedef struct {
  int key;
  float value;
} kvp;

bool comparekvp(const kvp& a, const kvp& b) { return (a.value > b.value); }

void topk(std::vector<kvp>& top, const std::vector<kvp>& list) {
  std::partial_sort_copy(
    list.begin(), list.end(), top.begin(), top.end(),
    comparekvp
  );
}

void rangetopk(
  std::vector<kvp>& top, const std::vector<kvp>& list, int start, int len) {
  std::partial_sort_copy(
    list.begin() + start, list.begin() + start + len, top.begin(), top.end(),
    comparekvp
  );
}

void multithreadtopk(
    std::vector<kvp>& top, const std::vector<kvp>& list, int nthr) {
  int k = top.size();
  int len = list.size();

  // does multi-threading worth it?
  if ((nthr < 2) || (len / nthr < 2*k)) {
    topk(top, list);
    return;
  }

  // map
  std::vector<std::vector<kvp>> mtop(nthr);
  #pragma omp parallel for
  for (int i = 0; i < nthr; i++) {
    mtop[i].resize(k);
    int start = i * len / nthr;
    int end = (i + 1) * len / nthr;
    rangetopk(mtop[i], list, start, end - start);
  }

  // reduce
  std::vector<kvp> tinylist;
  for (int i = 0; i < nthr; i++) {
    tinylist.insert(tinylist.end(), mtop[i].begin(), mtop[i].end());
  }
  topk(top, tinylist);
}

extern "C" {

void ctopk1d(float* top, long* ind, int k, float* values, int len, int nthr) {
  std::vector<kvp> list(len);
  std::vector<kvp> vtop(k);
  for (int i = 0; i < len; i++) {
    kvp elt {i, *values++};
    list[i] = elt;
  }
  multithreadtopk(vtop, list, nthr);
  for (const auto& elt : vtop) {
    *ind++ = elt.key + 1;
    *top++ = elt.value;
  }
}

void ctopk2d(float* top, long* ind, int k, float* values, int len, int n) {
  if (n == 1)  {
    // parallel at the sort level
    ctopk1d(top, ind, k, values, len, 20);
  } else {
    // parallel at the batch level
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      ctopk1d(top + i * k, ind + i * k, k, values + i * len, len, 1);
    }
  }
}

}
