# JAGS:
poor_model_path <- "../src/models/poor_model.jags"
advanced_model_path <- "../src/models/advanced_model.jags"
advanced_model_cheat_path <- "../src/models/advanced_model_cheat.jags"
ar_model_path <- "../src/models/ar_model.jags"

# Models:
model_files <- list.files("models", full.names = TRUE, pattern = "\\.R$")
sapply(model_files, source)

# Functions
function_files <- list.files("functions", full.names = TRUE, pattern = "\\.R$")
sapply(function_files, source)

source("../src/functions/data_generation_local.R")
source("../src/functions/data_generation_global.R")
source("../src/functions/simulate_model.R")
source("../src/functions/cumulative_regret.R")
source("../src/functions/baselines.R")
source("../src/models/adv_ts.R")
source("../src/models/adv_ts_nimble.R")
source("../src/models/adv_ts_batching.R")
source("../src/models/adv_ucb.R")
source("../src/models/poor_ts.R")
source("../src/models/poor_ucb.R")
source("../src/models/ar_ts.R")
source("../src/models/ar_ucb.R")
