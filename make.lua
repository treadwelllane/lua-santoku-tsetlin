local env = {

  name = "santoku-tsetlin",
  version = "0.0.56-1",
  variable_prefix = "TK_TSETLIN",
  public = true,

  cflags = { "-march=native", "-O3", "-ffast-math", "-Wall", "-Wextra", "-Wsign-compare", "-Wsign-conversion", "-Wstrict-overflow", "-Wpointer-sign", "-Wno-unused-parameter" },
  ldflags = { "-march=native", "-O3", "-lm", "-lpthread", },

  dependencies = {
    "lua >= 5.1",
    "santoku >= 0.0.245-1",
  },

  test = {
    cflags = { "-fopt-info-vec=optimize.txt", "-fopt-info-vec-missed=optimize.txt", "-g3" },
    ldflags = { "-fopt-info-vec=optimize.txt", "-fopt-info-vec-missed=optimize.txt", "-g3" },
    dependencies = {
      "luacov >= 0.15.0-1",
      "santoku-bitmap >= 0.0.23-1",
      "santoku-matrix >= 0.0.13-1",
      "santoku-fs >= 0.0.33-1",
    }
  },

}

env.homepage = "https://github.com/treadwelllane/lua-" .. env.name
env.tarball = env.name .. "-" .. env.version .. ".tar.gz"
env.download = env.homepage .. "/releases/download/" .. env.version .. "/" .. env.tarball

return {
  type = "lib",
  env = env,
}
