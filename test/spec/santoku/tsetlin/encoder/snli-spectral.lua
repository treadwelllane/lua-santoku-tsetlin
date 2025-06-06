local corex = require("santoku.corex")
local fs = require("santoku.fs")
local it = require("santoku.iter")
local ivec = require("santoku.ivec")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

local tm = require("santoku.tsetlin")
local ds = require("santoku.tsetlin.dataset")
-- local inv = require("santoku.tsetlin.inv")
local booleanizer = require("santoku.tsetlin.booleanizer")
local eval = require("santoku.tsetlin.evaluator")
local graph = require("santoku.tsetlin.graph")
local tch = require("santoku.tsetlin.tch")
local tokenizer = require("santoku.tsetlin.tokenizer")
local spectral = require("santoku.tsetlin.spectral")

local TRAIN_TEST_RATIO = 0.8
local TM_ITERS = 10
local MAX_RECORDS = 1000
local EVALUATE_EVERY = 1
local THREADS = nil

local HIDDEN = 64
local CLAUSES = 512
local TARGET = 0.1
local SPECIFICITY = 60

local TRANS_HOPS = 2
local TRANS_POS = 0
local TRANS_NEG = 2
local COREX_TOP_ITERS = 300
local COREX_TOP_ANCHOR = 1000.0
local TOP_ALGO = "chi2" -- mi, chi2, or corex
local TOP_K = 1024

local TOKENIZER_CONFIG = {
  max_df = 0.95,
  min_df = 0.001,
  max_len = 20,
  min_len = 1,
  max_run = 1,
  ngrams = 3,
  cgrams_min = 3,
  cgrams_max = 4,
  skips = 2,
  negations = 4,
  align = tm.align
}

test("tsetlin", function ()

  print("Creating tokenizer")
  local tokenizer = tokenizer.create(TOKENIZER_CONFIG)

  print("Reading data")
  local dataset = ds.read_snli_pairs("test/res/snli.10k.txt", MAX_RECORDS)
  local train, test = ds.split_snli_pairs(dataset, TRAIN_TEST_RATIO)

  print()
  str.printf("\tAll\tTrain\tTest\n\n")
  str.printf("Pos\t%d\t%d\t%d\n", dataset.n_pos, train.n_pos, test.n_pos)
  str.printf("Neg\t%d\t%d\t%d\n", dataset.n_neg, train.n_neg, test.n_neg)
  str.printf("Sent\t%d\t%d\t%d\n", dataset.n_sentences, train.n_sentences, test.n_sentences)

  print("\nTraining tokenizer\n")
  tokenizer.train({ corpus = train.raw_sentences })
  tokenizer.finalize()
  dataset.n_features = tokenizer.features()
  str.printf("Feat\t\t%d\t\t\n", dataset.n_features)

  print("\nTokenizing train")
  train.sentences = tokenizer.tokenize(train.raw_sentences)
  local stopwatch = utc.stopwatch()

  -- print("\nIndexing train")
  -- train.index = inv.create({ features = dataset.n_features })
  -- train.index:add(train.sentences)

  print("Creating graph")
  train.graph = graph.create({
    pos = train.pos,
    neg = train.neg,
    -- index = train.index,
    nodes = train.sentences,
    n_nodes = train.n_sentences,
    n_features = dataset.n_features,
    trans_hops = TRANS_HOPS,
    trans_pos = TRANS_POS,
    trans_neg = TRANS_NEG,
    threads = THREADS,
    each = function (s, b, n, dt)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f  Graph: %-9s  Components: %-6d  Positives: %3d  Negatives: %3d\n", d, dd, dt, s, b, n) -- luacheck: ignore
    end
  })

  print("\nSpectral hashing\n")
  train.codes0 = spectral.encode({
    graph = train.graph,
    n_hidden = HIDDEN,
    threads = THREADS,
    each = function (s)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f  Spectral Steps: %3d\n", d, dd, s)
    end
  })

  print("\nBooleanizing\n")
  local bzr = booleanizer.create({ n_thresholds = 1 })
  bzr:observe(train.codes0, HIDDEN)
  bzr:finalize()
  train.codes0 = bzr:encode(train.codes0, HIDDEN)
  train.n_hidden = bzr:bits()

  print("\nRefining\n")
  tch.refine({
    codes = train.codes0,
    graph = train.graph,
    n_hidden = train.n_hidden,
    threads = THREADS,
    each = function (s)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
    end
  })

  train.codes0_raw = train.codes0:raw_bitmap(train.n_sentences, train.n_hidden)
  train.similarity0 = eval.optimize_retrieval(train.codes0_raw, train.pos, train.neg, train.n_hidden, THREADS)
  str.printi("AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", -- luacheck: ignore
    train.similarity0)

  train.entropy0 = eval.entropy_stats(train.codes0_raw, train.n_sentences, train.n_hidden, THREADS)
  str.printi("Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)\n",
    train.entropy0)

  local top_v =
    TOP_ALGO == "chi2" and train.sentences:top_chi2(train.codes0_raw, train.n_sentences, dataset.n_features, train.n_hidden, TOP_K, THREADS) or -- luacheck: ignore
    TOP_ALGO == "mi" and train.sentences:top_mi(train.codes0_raw, train.n_sentences, dataset.n_features, train.n_hidden, TOP_K, THREADS) or -- luacheck: ignore
    TOP_ALGO == "corex" and (function ()
      print("Creating Corex")
      train.sel_corpus = ivec.create(train.sentences)
      train.sel_corpus:extend_bits(train.codes0, dataset.n_features, train.n_hidden)
      local cor = corex.create({
        visible = dataset.n_features + train.n_hidden,
        hidden = train.n_hidden,
        anchor = COREX_TOP_ANCHOR,
        threads = THREADS,
      })
      print("Training Corex")
      cor.train({
        corpus = train.sel_corpus,
        samples = train.n_sentences,
        iterations = COREX_TOP_ITERS,
        each = function (epoch, tc, dev)
          local duration, total = stopwatch()
          str.printf("Epoch  %-4d   Time  %6.3f  %6.3f   Convergence  %4.6f  %4.6f\n",
            epoch, duration, total, tc, dev)
        end
      })
      print("Extracting top features")
      return cor.top_visible(TOP_K)
    end)() or (function ()
      -- Fallback to all words
      local t = ivec.create(dataset.n_features)
      t:fill_indices()
      return t
    end)()

  local n_top_v = top_v:size()
  print("After top k filter", n_top_v)

  -- Show top words
  local words = tokenizer.index()
  for id in it.take(32, top_v:each()) do
    print(id, words[id + 1])
  end

  print("Re-encoding train/test with top features")
  tokenizer.restrict(top_v)
  train.sentences = tokenizer.tokenize(train.raw_sentences)
  test.sentences = tokenizer.tokenize(test.raw_sentences)

  print("Prepping for encoder")
  train.sentences:flip_interleave(train.n_sentences, n_top_v)
  test.sentences:flip_interleave(test.n_sentences, n_top_v)
  train.sentences = train.sentences:raw_bitmap(train.n_sentences, n_top_v * 2)
  test.sentences = test.sentences:raw_bitmap(test.n_sentences, n_top_v * 2)

  print()
  print("Input Features", n_top_v * 2)
  print("Encoded Features", train.n_hidden)
  print("Train Sentences", train.n_sentences)
  print("Test Sentences", test.n_sentences)

  print("Creating encoder")
  local t = tm.encoder({
    visible = n_top_v,
    hidden = train.n_hidden,
    clauses = CLAUSES,
    target = TARGET,
    specificity = SPECIFICITY,
    threads = THREADS,
  })

  print("Training encoder")
  stopwatch = utc.stopwatch()
  t.train({
    sentences = train.sentences,
    codes = train.codes0_raw,
    samples = train.n_sentences,
    iterations = TM_ITERS,
    each = function (epoch)
      if epoch == TM_ITERS or epoch % EVALUATE_EVERY == 0 then
        train.codes1 = t.predict(train.sentences, train.n_sentences)
        test.codes1 = t.predict(test.sentences, test.n_sentences)
        train.accuracy0 = eval.encoding_accuracy(train.codes1, train.codes0_raw, train.n_sentences, train.n_hidden, THREADS) -- luacheck: ignore
        train.similarity1 = eval.optimize_retrieval(train.codes1, train.pos, train.neg, train.n_hidden, THREADS) -- luacheck: ignore
        test.similarity1 = eval.optimize_retrieval(test.codes1, test.pos, test.neg, train.n_hidden, THREADS) -- luacheck: ignore
        print()
        str.printf("Epoch %3d  Time %3.2f %3.2f\n",
          epoch, stopwatch())
        print()
        -- print(serialize(train.accuracy0))
        str.printi("  Train (acc) |           | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | F1 Spread: %.2f#(f1_min) %.2f#(f1_max) %.2f#(f1_std)", train.accuracy0) -- luacheck: ignore
        str.printi("  Codes (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity0) -- luacheck: ignore
        str.printi("  Train (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity1) -- luacheck: ignore
        str.printi("  Test (sim)  | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", test.similarity1) -- luacheck: ignore
      else
        str.printf("Epoch %3d  Time %3.2f %3.2f\n",
          epoch, stopwatch())
      end
    end
  })

  if TM_ITERS == 0 then
    return
  end

  print()
  print("Persisting")
  fs.rm("model.bin", true)
  t.persist("model.bin")
  print("Restoring")
  t = tm.load("model.bin")

  do
    train.codes1 = t.predict(train.sentences, train.n_sentences)
    test.codes1 = t.predict(test.sentences, test.n_sentences)
    train.accuracy0 = eval.encoding_accuracy(
      train.codes1, train.codes0_raw, train.n_sentences, train.n_hidden, THREADS)
    train.similarity1 = eval.optimize_retrieval(
      train.codes1, train.pos, train.neg, train.n_hidden, THREADS)
    test.similarity1 = eval.optimize_retrieval(
      test.codes1, test.pos, test.neg, train.n_hidden, THREADS)
    print()
    str.printi("  Train (acc) |           | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | F1 Spread: %.2f#(f1_min) %.2f#(f1_max) %.2f#(f1_std)", train.accuracy0) -- luacheck: ignore
    str.printi("  Codes (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity0) -- luacheck: ignore
    str.printi("  Train (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity1) -- luacheck: ignore
    str.printi("  Test (sim)  | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", test.similarity1) -- luacheck: ignore
  end

end)
