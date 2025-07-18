local arr = require("santoku.array")
local fs = require("santoku.fs")
local base = fs.runfile("make.common.lua")
base.env.cflags = arr.extend({ "-O3", "-funroll-loops", "-ftree-vectorize", "-fno-math-errno", "-fassociative-math", "-freciprocal-math", "-fno-signed-zeros" }, base.env.cflags)
base.env.ldflags = arr.extend({ "-O3" }, base.env.ldflags)
return base
