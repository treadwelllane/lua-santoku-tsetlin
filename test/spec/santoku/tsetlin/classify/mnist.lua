local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local fs = require("santoku.fs")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local TTR = 0.9
local MAX = 10000
local EVALUATE_EVERY = 1
local ITERATIONS = 10

local CLASSES = 10
local CLAUSES = 1024
local TARGET = 32
local SPECIFICITY = 10
local NEGATIVE = 0.1

local FEATURES = 784

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", FEATURES, MAX)
  local train, test = ds.split_binary_mnist(dataset, TTR)

  print("Transforming train")
  train.problems:flip_interleave(train.n, dataset.n_features)
  train.problems = train.problems:raw_bitmap(train.n, dataset.n_features * 2)
  train.solutions = train.solutions:raw("u32")

  print("Transforming test")
  test.problems:flip_interleave(test.n, dataset.n_features)
  test.problems = test.problems:raw_bitmap(test.n, dataset.n_features * 2)
  test.solutions = test.solutions:raw("u32")

  print("Creating")
  local t = tm.classifier({
    features = dataset.n_features,
    classes = CLASSES,
    clauses = CLAUSES,
    target = TARGET,
    specificity = SPECIFICITY,
    negative = NEGATIVE,
  })

  print("Training")
  local stopwatch = utc.stopwatch()
  t.train({
    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,
    iterations = ITERATIONS,
    each = function (epoch)
      local train_pred = t.predict(train.problems, train.n)
      local test_pred = t.predict(test.problems, test.n)
      local duration = stopwatch()
      if epoch == ITERATIONS or epoch % EVALUATE_EVERY == 0 then
        local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, CLASSES)
        local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, CLASSES)
        str.printf("Epoch %-4d  Time %4.2f  Test %4.2f  Train %4.2f\n",
          epoch, duration, test_stats.f1, train_stats.f1)
      else
        str.printf("Epoch %-4d  Time %4.2f\n",
          epoch, duration)
      end
    end
  })

  print()
  print("Persisting")
  fs.rm("model.bin", true)
  t.persist("model.bin", true)

  print("Testing restore")
  t = tm.load("model.bin", nil, true)
  local train_pred = t.predict(train.problems, train.n)
  local test_pred = t.predict(test.problems, test.n)
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, CLASSES)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, CLASSES)
  str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_stats.f1, train_stats.f1)

end)
