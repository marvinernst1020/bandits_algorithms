simulate_model_on_run <- function(run_id,                 # unchanged
                                  N  = NULL,              # now optional
                                  K  = NULL,
                                  algorithm  = c("ts","ucb"),
                                  complexity = c("advanced","poor",
                                                 "baseline","ar","kfas"),
                                  dynamics   = c("common","independent"),
                                  data_path  = c("data_global/A","data_local/A"),
                                  setting    = c("global","local"))
{
  algorithm  <- match.arg(algorithm)
  complexity <- match.arg(complexity)
  dynamics   <- match.arg(dynamics)
  setting    <- match.arg(setting)

  # -------------------------------------------------------------------
  file_prefix <- if (setting == "global") "global_truth_" else "local_truth_"
  file_path   <- file.path(data_path, paste0(file_prefix, run_id, ".rds"))
  data        <- readRDS(file_path)

  ## Always recompute K & N from the file -----------------------------
  if (is.null(K)) K <- nrow(data$y)           # works for both global/local
  if (is.null(N)) N <- ncol(data$y)

  message(glue::glue(
    "[{Sys.time()}] Run {run_id}: {toupper(setting)} | {algorithm} |",
    " {complexity} | {dynamics} | K={K}, N={N}"
  ))

  # -------------------------------------------------------------------
  if (algorithm == "ts") {
    if (complexity == "advanced") {
      res   <- thompson_advanced(K,N,data$mu,data$y,data$z,
                                 100, 1000, 200, dynamics)
      model <- "M2 TS"

    } else if (complexity == "poor") {
      res   <- thompson_poor(K,N,data$mu,data$y,data$z,
                             100, 1000, 200, dynamics)
      model <- "M1 TS"

    } else if (complexity == "baseline") {
      res   <- bandit_baselines("ts",K,N,data$y,data$z,data$mu,
                                dynamics, batch_size = 1)
      model <- "M0 TS"

    } else if (complexity == "ar") {
      res   <- thompson_ar(K,N,data$mu,data$y,data$z,
                           100, 1000, 200, dynamics)
      model <- "AR TS"

    } else if (complexity == "kfas") {
      res   <- thompson_ar_kfas(K,N,data$mu,data$y,data$z,
                                100, 1000, 200, dynamics)
      model <- "KFAS TS"
    }

  } else if (algorithm == "ucb") {
    if (complexity == "advanced") {
      res   <- ucb_advanced(K,N,data$mu,data$y,data$z,
                            100, 1000, 200, dynamics)
      model <- "M2 UCB"

    } else if (complexity == "poor") {
      res   <- ucb_poor(K,N,data$mu,data$y,data$z,
                        100, 1000, 200, dynamics)
      model <- "M1 UCB"

    } else if (complexity == "baseline") {
      res   <- bandit_baselines("ucb-tuned",K,N,data$y,data$z,data$mu,
                                dynamics, batch_size = 1)
      model <- "M0 UCB"

    } else if (complexity == "ar") {
      res   <- ucb_ar(K,N,data$mu,data$y,data$z,
                      100, 500, 100, dynamics)
      model <- "AR UCB"

    } else if (complexity == "kfas") {
      res   <- ucb_ar_kfas(K,N,data$mu,data$y,data$z,
                           100, 500, 100, dynamics)
      model <- "KFAS UCB"
    }
  }

  tibble(
    time              = 1:N,
    cumulative_reward = res$cumulative_reward,
    cumulative_regret = res$cumulative_regret,
    run               = run_id,
    model             = model
  )
}
