local env = {

  name = "santoku-tsetlin",
  version = "0.0.11-1",
  variable_prefix = "TK_TSETLIN",
  public = true,

  cflags = "-O3 -march=native -ffast-math -Wall -Wextra",

  dependencies = {
    "lua == 5.1",
    "santoku == 0.0.201-1",
  },

  test = {
    dependencies = {
      "luacov == 0.15.0-1",
      "santoku-bitmap == 0.0.3-1",
      "santoku-matrix == 0.0.4-1",
      "santoku-fs == 0.0.29-1",
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
