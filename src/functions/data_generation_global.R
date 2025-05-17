# DATA GENERATION - GLOBAL LATENT STATE

#' Generate datasets where all arms share a common latent state
#'
#' @param K Number of arms
#' @param N Number of time steps
#' @param mu K x 2 matrix of reward probabilities (rows = arms, cols = latent states)
#' @param pi_global 2x2 transition matrix for the global latent state
#' @param n_runs Number of datasets to generate
#' @param save_path Folder path to save the generated datasets

generate_global_datasets <- function(K = 2, N = 5000, mu, pi_global,
                                     n_runs = 25,
                                     scenario_name = NULL,
                                     root_path = "data_global") {
  
  # Auto-generate name if not provided
  if (is.null(scenario_name)) {
    pi_flat <- paste(round(pi_global, 3), collapse = "_")
    scenario_name <- paste0("pi_", pi_flat)
  }
  
  save_path <- file.path(root_path, scenario_name)
  dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
  
  for (run in 1:n_runs) {
    z_global <- numeric(N)
    z_global[1] <- rbinom(1, 1, 0.5)
    
    for (t in 2:N) {
      z_global[t] <- rbinom(1, 1, pi_global[z_global[t - 1] + 1, 2])
    }
    
    y_global <- matrix(0, nrow = K, ncol = N)
    for (i in 1:K) {
      for (t in 1:N) {
        y_global[i, t] <- rbinom(1, 1, mu[i, z_global[t] + 1])
      }
    }
    
    saveRDS(list(y = y_global, z = z_global, mu = mu, pi = pi_global),
            file = file.path(save_path, paste0("global_truth_", run, ".rds")))
  }
  # Save a description of the scenario
  readme_text <- paste0(
    "Scenario: ", scenario_name, "\n",
    "K: ", K, "\n",
    "N: ", N, "\n",
    "mu:\n", paste(capture.output(print(mu)), collapse = "\n"), "\n\n",
    "pi_global:\n", paste(capture.output(print(pi_global)), collapse = "\n")
  )
  writeLines(readme_text, con = file.path(save_path, "README.txt"))
}
