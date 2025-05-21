#' Simulate a bandit model run for a given dataset
#'
#' @param run_id Numeric ID for the dataset file (e.g., 1 loads "..._1.rds")
#' @param N Number of time steps to simulate
#' @param K Number of arms
#' @param algorithm Algorithm to use: "ts" or "ucb"
#' @param complexity "advanced", "poor", "baseline" or "ar"
#' @param dynamics "common" or "independent"
#' @param data_path Folder where the dataset is stored
#' @param model_paths Named list with model file paths
#' @param setting "global" or "local" (controls filename prefix)
#'
#' @return A tibble with results from the model run

simulate_model_on_run <- function(run_id, N, K,
                                  algorithm = c("ts", "ucb"),
                                  complexity = c("advanced", "poor", "baseline","ar"),
                                  dynamics = c("common", "independent"),
                                  data_path = "data_global/A",
                                  setting = c("global", "local")) {
  
  algorithm <- match.arg(algorithm)
  complexity <- match.arg(complexity)
  dynamics <- match.arg(dynamics)
  setting <- match.arg(setting)
  
  message(glue::glue("[{Sys.time()}] Starting run {run_id}: {toupper(setting)} / {algorithm} / {complexity} / {dynamics}"))
  
  # Automatically set file prefix based on setting
  file_prefix <- if (setting == "global") "global_truth_" else "local_truth_"
  file_path <- file.path(data_path, paste0(file_prefix, run_id, ".rds"))
  data <- readRDS(file_path)
  
  # Run specified model
  if (algorithm == "ts") {
    if (complexity == "advanced") {
      res <- thompson_advanced(K, N, data$mu, data$y, data$z,
                               batch_size = 100, burn = 500, n_iter = 100,
                               dynamics = dynamics)
      model <- "M2 TS"
    } else if (complexity == "advanced_batching") {
      res <- thompson_advanced(K, N, data$mu, data$y, data$z,
                           batch_size = 100, burn = 500, n_iter = 100,
                           dynamics = dynamics)
      model <- "M1 TS"
    } else if (complexity == "poor") {
      res <- thompson_poor(K, N, data$mu, data$y, data$z,
                           batch_size = 100, burn = 500, n_iter = 100,
                           dynamics = dynamics)
      model <- "M1 TS"
    } else if (complexity == "baseline") {
      res <- bandit_baselines("ts", K, N, data$y, data$z, data$mu,
                              dynamics = dynamics, batch_size = 100)
      model <- "M0 TS"
    } else if (complexity == "ar") {
      res <- thompson_ar(K, N, data$mu, data$y, data$z,
                         batch_size = 100, burn = 500, n_iter = 100,
                         dynamics = dynamics)
      model <- "AR TS"
    }
  } else if (algorithm == "ucb") {
    if (complexity == "advanced") {
      res <- ucb_advanced(K, N, data$mu, data$y, data$z,
                          batch_size = 100, burn = 500, n_iter = 100,
                          dynamics = dynamics)
      model <- "M2 UCB"
    } else if (complexity == "poor") {
      res <- ucb_poor(K, N, data$mu, data$y, data$z,
                      batch_size = 100, burn = 500, n_iter = 100,
                      dynamics = dynamics)
      model <- "M1 UCB"
    } else if (complexity == "baseline") {
      res <- bandit_baselines("ucb-tuned", K, N, data$y, data$z, data$mu,
                              dynamics = dynamics, batch_size = 100)
      model <- "M0 UCB"
    } # else if (complexity == "ar") to be done
  } else {
    stop("Unsupported algorithm")
  }
  
  #model_id <- paste(model, dynamics)
  model_id <- model
  
  tibble(
    time = 1:N,
    cumulative_reward = res$cumulative_reward,
    cumulative_regret = res$cumulative_regret,
    run = run_id,
    model = model,
    model_id = model_id
  )
}
