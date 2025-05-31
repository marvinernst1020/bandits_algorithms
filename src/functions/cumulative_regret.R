#' Plot average cumulative regret over time
#'
#' @param df A summary data.frame with columns: time, avg_regret, model_id
#' @param title Plot title
#' @param palette Optional named color palette
#'
#' @return ggplot object
#' 
#' 

plot_cumulative_regret <- function(df, title = NULL, palette = NULL) {

  default_palette <- c(
    "M0 TS" = "#2CA02C",
    "M1 TS" = "#1F77B4",
    "M2 TS" = "#D62728",
    "AR TS" = "#002db3",
    
    "M0 UCB" = "#2CA02C",
    "M1 UCB" = "#1F77B4",
    "M2 UCB" = "#D62728",
<<<<<<< HEAD
    "AR UCB" = "#002db3"
  )
  fallback_colors <- c("#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#FF9896", "#AEC7E8")

  pal <- palette %||% default_palette
  missing_ids <- setdiff(unique(df$model_id), names(pal))
  if (length(missing_ids) > 0) {
    pal[missing_ids] <- fallback_colors[seq_len(length(missing_ids))]
  }
  
  # Determine cumulative‐regret column
  if      ("avg_regret"       %in% names(df)) cumul_var <- "avg_regret"
  else if ("cumulative_regret" %in% names(df)) cumul_var <- "cumulative_regret"
  else stop("Data frame must contain 'avg_regret' or 'cumulative_regret'.")
  
  # Determine instantaneous‐regret column (if any)
  if      ("inst_regret"         %in% names(df)) inst_var <- "inst_regret"
  else if ("instantaneous_regret" %in% names(df)) inst_var <- "instantaneous_regret"
  else inst_var <- NULL
  
  # Check if single model_id (single run)
  single_run <- length(unique(df$model_id)) == 1
  
  # --- Build cumulative plot ---
  p_cum <- ggplot(df, aes_string(x = "time", y = cumul_var, color = "model_id")) +
    geom_line(size = 0.9) +
    labs(title = title, x = "Time Step", y = "Cumulative Regret") +
    scale_color_manual(values = pal) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = if (single_run) "none" else "bottom",
      legend.title = element_blank()
    )
  
  # If single run, return only the cumulative plot
  if (single_run) {
    return(p_cum)
  }
  
  # --- Build instantaneous plot, if available ---
  if (!is.null(inst_var)) {
    p_inst <- ggplot(df, aes_string(x = "time", y = inst_var, color = "model_id")) +
      geom_line(size = 0.9) +
      labs(title = title, x = "Time Step", y = "Instantaneous Regret") +
      scale_color_manual(values = pal) +
      theme_classic(base_size = 12) +
      theme(
        legend.position = "bottom",
        legend.title = element_blank()
      )
    return(list(cumulative = p_cum, instantaneous = p_inst))
  }
  
  # If no instantaneous column, return cumulative only
  p_cum
}

=======
    "AR UCB" = "#002db3",
  )
  
  p1<-ggplot(df, aes(x = time, y = avg_regret, color = model_id)) +
    geom_line(size = 0.9) +
    labs(
      title = title,
      x = "Time Step",
      y = "Average Cumulative Regret"
    ) +
    scale_color_manual(values = palette %||% default_palette) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = "bottom",
      legend.title = element_blank()
    )
  
  p2<-ggplot(df, aes(x = time, y = inst_regret, color = model_id)) +
    geom_line(size = 0.9) +
    labs(
      title = title,
      x = "Time Step",
      y = "Average Instantaneous Regret"
    ) +
    scale_color_manual(values = palette %||% default_palette) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = "bottom",
      legend.title = element_blank()
    )
  
  return(list(cumulative = p1, instantaneous = p2))
}
>>>>>>> c4b2de2277ba0cfea544805f97bbb8f03ebf5044
