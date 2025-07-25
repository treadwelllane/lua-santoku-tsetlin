require("santoku.rvec")
local serialize = require("santoku.serialize") -- luacheck: ignore
local tbl = require("santoku.table")
local pvec = require("santoku.pvec")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
local it = require("santoku.iter")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local num = require("santoku.num")
local err = require("santoku.error")

local M = {}

M.read_snli_pairs = function (fp, max, single_split)
  max = max or num.huge
  local sentences = {}
  local pos = {}
  local neg = {}
  local ns = 0
  for line in it.drop(1, fs.lines(fp)) do
    local chunks = str.gmatch(line, "[^\t]+")
    local label = chunks()
    chunks = it.drop(4, chunks)
    local a = chunks()
    local b = chunks()
    err.assert(label and a and b, "unable to parse line", line)
    label = label == "entailment" and 1 or label == "contradiction" and 0 or nil
    if label then
      local na = sentences[a]
      local nb = sentences[b]
      if not na then
        ns = ns + 1
        na = ns
        sentences[a] = na
        sentences[na] = a
      end
      if not nb then
        ns = ns + 1
        nb = ns
        sentences[b] = nb
        sentences[nb] = b
      end
      if label == 1 then
        arr.push(pos, na, nb)
      else
        arr.push(neg, na, nb)
      end
      if (#pos + #neg) / 2 >= max then
        break
      end
    end
  end
  local ds = {
    pos = pos,
    neg = neg,
    n_pos = #pos / 2,
    n_neg = #neg / 2,
    raw_sentences = sentences,
    n_sentences = ns,
  }
  if single_split then
    return M.split_snli_pairs(ds)
  else
    return ds
  end
end

local function _split_snli_pairs (dataset, split, prop, start, size)
  local pairs_to = {}
  local pairs_from = dataset[prop]
  for i = start, start + size - 1 do
    local ia, ib =
      pairs_from[(i - start) * 2 + 1],
      pairs_from[(i - start) * 2 + 2]
    local sa, sb =
      dataset.raw_sentences[ia],
      dataset.raw_sentences[ib]
    local na, nb =
      split.raw_sentences[sa],
      split.raw_sentences[sb]
    if not na then
      split.n_sentences = split.n_sentences + 1
      na = split.n_sentences
      split.raw_sentences[na] = sa
      split.raw_sentences[sa] = na
    end
    if not nb then
      split.n_sentences = split.n_sentences + 1
      nb = split.n_sentences
      split.raw_sentences[nb] = sb
      split.raw_sentences[sb] = nb
    end
    arr.push(pairs_to, na - 1, nb - 1)
  end
  split[prop] = ivec.create(pairs_to)
end

M.split_snli_pairs = function (dataset, ratio)

  if not ratio then
    local train = tbl.merge({}, dataset)
    train.n_pos = dataset.n_pos
    train.n_neg = dataset.n_neg
    train.n_sentences = 0
    train.raw_sentences = {}
    _split_snli_pairs(dataset, train, "pos", 1, train.n_pos)
    _split_snli_pairs(dataset, train, "neg", 1, train.n_neg)
    return train
  end

  local test, train = {}, {}

  train.n_pos = num.floor(dataset.n_pos * ratio)
  train.n_neg = num.floor(dataset.n_neg * ratio)
  train.n_sentences = 0
  train.raw_sentences = {}

  test.n_pos = dataset.n_pos - train.n_pos
  test.n_neg = dataset.n_neg - train.n_neg
  test.n_sentences = 0
  test.raw_sentences = {}

  _split_snli_pairs(dataset, train, "pos", 1, train.n_pos)
  _split_snli_pairs(dataset, train, "neg", 1, train.n_neg)

  _split_snli_pairs(dataset, test, "pos", train.n_pos + 1, test.n_pos)
  _split_snli_pairs(dataset, test, "neg", train.n_neg + 1, test.n_neg)

  return train, test

end

M.read_binary_mnist = function (fp, n_features, max, class_max)
  local class_cnt = {}
  local offsets = ivec.create()
  local problems = ivec.create()
  local solutions = ivec.create()
  local n = 0
  local tmp = ivec.create()
  for l in fs.lines(fp) do
    if max and n >= max then
      break
    end
    local start = problems:size()
    local feature = 0
    tmp:clear()
    local cls = nil
    for s in str.gmatch(l, "%S+") do
      if feature == n_features then
        cls = s
        break
      elseif s == "1" then
        tmp:push(feature)
      elseif s ~= "0" then
        err.error("unexpected string", s)
      end
      feature = feature + 1
    end
    if not cls then -- luacheck: ignore
      -- malformed row: no label; skip silently (or log)
    else
      class_cnt[cls] = (class_cnt[cls] or 0) + 1
      if not (class_max and class_cnt[cls] > class_max) then
        offsets:push(start)
        problems:copy(tmp, 0, tmp:size(), problems:size())
        solutions:push(tonumber(cls))
        n = n + 1
      end
    end
  end
  return {
    offsets = offsets,
    problems = problems,
    solutions = solutions,
    n_features = n_features,
    n = n,
  }
end

local function _split_binary_mnist (dataset, s, e)
  local ps = ivec.create()
  local ss = ivec.create()
  for i = s, e do
    local pss = dataset.offsets:get(i - 1)
    local pse = i == dataset.n and dataset.problems:size() or dataset.offsets:get(i)
    local m = ps:size()
    ps:copy(dataset.problems, pss, pse, m)
    ps:add((i - s) * dataset.n_features, m, ps:size())
  end
  ss:copy(dataset.solutions, s - 1, e, 0)
  return {
    problems = ps,
    solutions = ss,
    n_features = dataset.n_features,
    n = e - s + 1,
  }
end

M.split_binary_mnist = function (dataset, ratio)
  if ratio >= 1 then
    return
      _split_binary_mnist(dataset, 1, dataset.n)
  else
    local n_train = num.floor(dataset.n * ratio)
    return
      _split_binary_mnist(dataset, 1, n_train),
      _split_binary_mnist(dataset, n_train + 1, dataset.n)
  end
end

M.multiclass_pairs = function (labels, n_anchors_pos, n_anchors_neg)
  -- nil => default; 0 => skip
  if n_anchors_pos == nil then n_anchors_pos = 1 end
  if n_anchors_neg == nil then
    n_anchors_neg = (n_anchors_pos > 0) and n_anchors_pos or 1
  end
  local pos, neg = pvec.create(), pvec.create()
  local class_to_indices, class_to_anchors = {}, {}
  local classes = ivec.create()
  -- Collect indices per class
  for i, y in labels:ieach() do
    if not class_to_indices[y] then
      class_to_indices[y] = ivec.create()
      classes:push(y)
    end
    class_to_indices[y]:push(i)
  end
  -- Helper: Fisher–Yates on a Lua table of 0-based indices
  local function shuffled_order(n)
    local t = {}
    for i = 0, n - 1 do t[i] = i end
    for i = n - 1, 1, -1 do
      local j = num.random(i + 1) - 1
      t[i], t[j] = t[j], t[i]
    end
    return t
  end
  -- Anchors for positives
  for class in classes:each() do
    class_to_anchors[class] = ivec.create()
  end
  if n_anchors_pos > 0 then
    for class in classes:each() do
      local idxs = class_to_indices[class]
      local k = num.min(n_anchors_pos, idxs:size())
      if k > 0 then
        local order = shuffled_order(idxs:size())
        local anchors = class_to_anchors[class]
        for i = 0, k - 1 do
          anchors:push(idxs:get(order[i]))
        end
      end
    end
  end
  -- Positive edges
  if n_anchors_pos > 0 then
    for class in classes:each() do
      local idxs = class_to_indices[class]
      local anchors = class_to_anchors[class]
      for idx in idxs:each() do
        for a in anchors:each() do
          if idx ~= a then pos:push(idx, a) end
        end
      end
    end
  end
  -- Precompute negative pools once per class
  if n_anchors_neg > 0 then
    local neg_pool_for = {}
    local order_for = {}
    for class in classes:each() do
      local pool = ivec.create()
      for other_class in classes:each() do
        if other_class ~= class then
          local src = class_to_anchors[other_class]
          if src:size() == 0 then src = class_to_indices[other_class] end
          for a in src:each() do pool:push(a) end
        end
      end
      neg_pool_for[class] = pool
      if pool:size() > 0 then
        order_for[class] = shuffled_order(pool:size())
      end
    end
    -- Negative edges, reuse shuffled order by sliding window
    for class in classes:each() do
      local pool = neg_pool_for[class]
      local pool_size = pool:size()
      if pool_size > 0 then
        local order = order_for[class]
        local need = n_anchors_neg
        local wrap = (need > pool_size)
        local idxs = class_to_indices[class]
        local offset = 0
        for idx in idxs:each() do
          if not wrap then
            for j = 0, need - 1 do
              local a = pool:get(order[(offset + j) % pool_size])
              neg:push(idx, a)
            end
            offset = (offset + need) % pool_size
          else
            -- with replacement (still cheap)
            for _ = 1, need do
              local a = pool:get(order[num.random(pool_size) - 1])
              neg:push(idx, a)
            end
          end
        end
      end
    end
  end
  return pos, neg
end

-- M.multiclass_pairs = function (labels, n_anchors_pos, n_anchors_neg)
--   n_anchors_pos = n_anchors_pos or 1
--   n_anchors_neg = n_anchors_neg or n_anchors_pos or 1
--   local pos, neg = pvec.create(), pvec.create()
--   local class_to_indices, class_to_anchors = {}, {}
--   local classes = ivec.create()
--   -- Collect indices per class
--   for i, y in labels:ieach() do
--     if not class_to_indices[y] then
--       class_to_indices[y] = ivec.create()
--       classes:push(y)
--     end
--     class_to_indices[y]:push(i)
--   end
--   -- Choose anchors per class
--   for class in classes:each() do
--     local indices = class_to_indices[class]
--     local anchors = ivec.create()
--     local used = {}
--     while anchors:size() < num.min(n_anchors_pos, indices:size()) do
--       local candidate = indices:get(num.random(indices:size()) - 1)
--       if not used[candidate] then
--         anchors:push(candidate)
--         used[candidate] = true
--       end
--     end
--     class_to_anchors[class] = anchors
--   end
--   -- Positive edges: connect each node to its own class anchors
--   for class in classes:each() do
--     local indices = class_to_indices[class]
--     local anchors = class_to_anchors[class]
--     for idx in indices:each() do
--       for anchor in anchors:each() do
--         if idx ~= anchor then
--           pos:push(idx, anchor)
--         end
--       end
--     end
--   end
--   -- Negative edges: connect each node ONLY to other-class anchors (once!)
--   for class in classes:each() do
--     local indices = class_to_indices[class]
--     local negative_anchors = ivec.create()
--     -- Gather all other-class anchors
--     for other_class in classes:each() do
--       if other_class ~= class then
--         local anchors = class_to_anchors[other_class]
--         for anchor in anchors:each() do
--           negative_anchors:push(anchor)
--         end
--       end
--     end
--     -- Connect each node negatively to selected other-class anchors
--     for idx in indices:each() do
--       local used_neg = {}
--       for _ = 1, n_anchors_neg do
--         local anchor_neg
--         repeat
--           anchor_neg = negative_anchors:get(num.random(negative_anchors:size()) - 1)
--         until not used_neg[anchor_neg]
--         used_neg[anchor_neg] = true
--         neg:push(idx, anchor_neg)
--       end
--     end
--   end
--   return pos, neg
-- end

M.read_imdb = function (dir, max)
  local problems = {}
  local solutions = {}
  local pos = it.paste(1, it.take(max or num.huge, fs.files(dir .. "/pos")))
  local neg = it.paste(0, it.take(max or num.huge, fs.files(dir .. "/neg")))
  local samples = it.map(function (label, text)
    return label, fs.readfile(text)
  end, it.chain(pos, neg))
  for label, review in samples do
    solutions[#solutions + 1] = label
    problems[#problems + 1] = review
  end
  arr.shuffle(problems, solutions)
  return {
    n = #problems,
    problems = problems,
    solutions = solutions
  }
end

local function _split_imdb (dataset, s, e)
  local ps = arr.copy({}, dataset.problems, 1, s, e)
  local ss = arr.copy({}, dataset.solutions, 1, s, e)
  ss = ivec.create(ss)
  return {
    n = #ps,
    problems = ps,
    solutions = ss
  }
end

M.split_imdb = function (dataset, ratio)
  local n_train = num.floor(#dataset.problems * ratio)
  return
    _split_imdb(dataset, 1, n_train),
    _split_imdb(dataset, n_train + 1, #dataset.problems)
end

M.read_glove = function (fp, max)
  local n_dims = nil
  local embeddings = dvec.create()
  local words = {}
  local added = 0
  for l in it.take(max or num.huge, fs.lines(fp)) do
    local chunks = str.gmatch(l, "%S+")
    local word = chunks()
    words[#words + 1] = word
    words[word] = #words
    for n in chunks do
      n = err.assert(tonumber(n), "error parsing dimension")
      embeddings:push(n)
      added = added + 1
    end
    if not n_dims then
      n_dims = added
    end
    err.assert(added == n_dims, "dimension mismatch", "expected", n_dims, "found", added)
    added = 0
  end
  embeddings:shrink()
  return {
    n_embeddings = #words,
    embeddings = embeddings,
    n_dims = n_dims,
    words = words
  }
end

return M
