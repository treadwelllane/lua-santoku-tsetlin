local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local utc = require("santoku.utc")
local tm = require("santoku.tsetlin")
local num = require("santoku.num")
local bm = require("santoku.bitmap")
local mtx = require("santoku.matrix")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")

local TTR = 0.9
local THREADS = nil
local EVALUATE_EVERY = 1
local ITERATIONS = 100

local CLASSES = 10
local CLAUSES = 8192
local STATE = 8
local TARGET = 32
local BOOST = true
local SPEC_LOW = 2
local SPEC_HIGH = 200
local ACTIVE = 0.75

local FEATURES = 784
local WIDTH = 28
local HEIGHT = 28
local PATCH_WIDTH = 10
local PATCH_HEIGHT = 10
local PATCH_STRIDE_X = 1
local PATCH_STRIDE_Y = 1

local function read_data (fp, skip, max)
  local problems = {}
  local solutions = {}
  local bits = {}
  local skip = skip or 0
  for l in fs.lines(fp) do
    if skip > 0 then
      skip = skip - 1
    else
      local n = 0
      for bit in str.gmatch(l, "%S+") do
        n = n + 1
        if n == FEATURES + 1 then
          local s = tonumber(bit)
          solutions[#solutions + 1] = s
          break
        end
        bit = bit == "1"
        if bit then
          bits[n] = true
        else
          bits[n] = nil
        end
      end
      if n ~= FEATURES + 1 then
        error("bitmap length mismatch")
      else
        problems[#problems + 1] = bm.create(bits, FEATURES)
      end
      if max and #problems >= max then
        break
      end
    end
  end
  return {
    problems = problems,
    solutions = solutions
  }
end

local function split_dataset (dataset, s, e)
  local ps, ss = {}, {}
  for i = s, e do
    arr.push(ps, dataset.problems[i])
    arr.push(ss, dataset.solutions[i])
  end
  local b = bm.matrix(ps, FEATURES)
  local m = mtx.create(1, #ss)
  mtx.set(m, 1, ss)
  return b, mtx.raw(m, 1, 1, "u32")
end

test("tsetlin", function ()

  local SKIP = 0
  local MAX = 6000

  print("Reading data")
  local dataset = read_data("test/res/santoku/tsetlin/BinarizedMNISTData/MNISTTraining.txt", SKIP, MAX)

  print("Splitting & packing")
  local n_train = num.floor(#dataset.problems * TTR)
  local n_test = #dataset.problems - n_train
  local train_problems, train_solutions = split_dataset(dataset, 1, n_train)
  local test_problems, test_solutions = split_dataset(dataset, n_train + 1, n_train + n_test)

  print("Patching")
  local num_patches, patch_size
  train_problems, num_patches, patch_size = bm.convolve_2d(train_problems, n_train, WIDTH, HEIGHT, PATCH_WIDTH, PATCH_HEIGHT, PATCH_STRIDE_X, PATCH_STRIDE_Y, 128)
  test_problems = bm.convolve_2d(test_problems, n_test, WIDTH, HEIGHT, PATCH_WIDTH, PATCH_HEIGHT, PATCH_STRIDE_X, PATCH_STRIDE_Y, 128)
  print("Patch size", patch_size)
  print("Number of patches", num_patches)
  print("Effective features", num_patches * patch_size)
  -- convolve_1d(x, patch, stride, round)
  -- convolve_sinusoidal(x, wave, buckets, dims, round)

  str.printf("Train %d  Test %d\n", n_train, n_test)

  print("Flip/interleave")
  train_problems = bm.raw(bm.flip_interleave(train_problems, n_train * num_patches, patch_size), n_train * num_patches * patch_size * 2)
  test_problems = bm.raw(bm.flip_interleave(test_problems, n_test * num_patches, patch_size), n_test * num_patches * patch_size * 2)

  print("Train", n_train)
  print("Test", n_test)

  print("Creating")
  local t = tm.classifier({
    features = patch_size,
    convolution = num_patches,
    classes = CLASSES,
    clauses = CLAUSES,
    state = STATE,
    target = TARGET,
    boost = BOOST,
    specificity_low = SPEC_LOW,
    specificity_high = SPEC_HIGH,
    threads = THREADS,
  })

  print("Training")
  local stopwatch = utc.stopwatch()
  t.train({
    samples = n_train,
    problems = train_problems,
    solutions = train_solutions,
    iterations = ITERATIONS,
    active = ACTIVE,
    each = function (epoch)
      local duration = stopwatch()
      if epoch == ITERATIONS or epoch % EVALUATE_EVERY == 0 then
        local test_score =
          t.evaluate({
            problems = test_problems,
            solutions = test_solutions,
            samples = n_test,
          })
        local train_score --[[, confusion, observed, predicted]] =
          t.evaluate({
            problems = train_problems,
            solutions = train_solutions,
            samples = n_train
          })
        str.printf("Epoch %-4d  Time %4.2f  Test %4.2f  Train %4.2f\n",
          epoch, duration, test_score, train_score)
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
  local test_score  =
    t.evaluate({
      problems = test_problems,
      solutions = test_solutions,
      samples = n_test,
    })
  local train_score--[[, confusion, predictions]] =
    t.evaluate({
      problems = train_problems,
      solutions = train_solutions,
      samples = n_train,
      -- stats = true,
    })
  -- print()
  -- print("Confusion:")
  -- print(require("santoku.serialize")(confusion))
  -- print()
  -- print("Predictions:")
  -- print(require("santoku.serialize")(predictions))
  -- print()
  str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_score, train_score)

end)
