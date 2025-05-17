# DATA GENERATION - LOCAL LATENT STATE

#' Generate datasets where each arm has its own latent state dynamics
#'
#' @param K Number of arms
#' @param N Number of time steps
#' @param theta K x 2 matrix of reward probabilities (rows = arms, cols = latent states)
#' @param pi_list List of K transition matrices (2x2 each) for each arm's latent state
#' @param n_runs Number of datasets to generate
#' @param scenario_name Optional name to use for the saved folder
#' @param root_path Root folder where scenario folders will be stored

generate_local_datasets <- function(K = 3, N = 1000, theta, pi_list,
                                    n_runs = 25,
                                    scenario_name = NULL,
                                    root_path = "data_local") {
  
  # Auto-generate name if not provided
  if (is.null(scenario_name)) {
    pi_summary <- lapply(pi_list, function(pi) paste(round(pi, 2), collapse = "_"))
    scenario_name <- paste0("piList_", paste(pi_summary, collapse = "__"))
  }
  
  save_path <- file.path(root_path, scenario_name)
  dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
  
  for (run in 1:n_runs) {
    z_local <- matrix(0, nrow = K, ncol = N)
    y_local <- matrix(0, nrow = K, ncol = N)
    
    for (i in 1:K) {
      z_local[i, 1] <- rbinom(1, 1, 0.5)
      y_local[i, 1] <- rbinom(1, 1, theta[i, z_local[i, 1] + 1])
      
      for (t in 2:N) {
        z_local[i, t] <- rbinom(1, 1, pi_list[[i]][z_local[i, t - 1] + 1, 2])
        y_local[i, t] <- rbinom(1, 1, theta[i, z_local[i, t] + 1])
      }
    }
    
    saveRDS(list(y = y_local, z = z_local, theta = theta, pi = pi_list),
            file = file.path(save_path, paste0("local_truth_", run, ".rds")))
  }
  
  # Save a description of the scenario
  pi_text <- paste0("Arm ", seq_along(pi_list), ":\n",
                    sapply(pi_list, function(pi) paste(capture.output(print(pi)), collapse = "\n")),
                    collapse = "\n\n")
  
  readme_text <- paste0(
    "Scenario: ", scenario_name, "\n",
    "K: ", K, "\n",
    "N: ", N, "\n\n",
    "theta:\n", paste(capture.output(print(theta)), collapse = "\n"), "\n\n",
    "pi_list:\n", pi_text
  )
  
  writeLines(readme_text, con = file.path(save_path, "README.txt"))
}
