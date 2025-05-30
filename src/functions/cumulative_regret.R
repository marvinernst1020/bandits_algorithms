#' Plot average cumulative regret over time
#'
#' @param df A summary data.frame with columns: time, avg_regret, model_id
#' @param title Plot title
#' @param palette Optional named color palette
#'
#' @return ggplot object
plot_cumulative_regret <- function(df, title = NULL, palette = NULL) {
  
  default_palette <- c(
    "M0 TS" = "#2CA02C",
    "M1 TS" = "#1F77B4",
    "M2 TS" = "#D62728",
    "AR TS" = "#002db3",
    
    "M0 UCB" = "#98DF8A",  
    "M1 UCB" = "#AEC7E8",  
    "M2 UCB" = "#FF9896",
    "AR UCB" = "#9999ff",
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


plot_single_cumulative_regret <- function(df, title = NULL, palette = NULL) {
  
  # Default named color palette for known models
  default_palette <- c(
    "M0 TS" = "#2CA02C",
    "M1 TS" = "#1F77B4",
    "M2 TS" = "#D62728",
    "AR TS" = "#002db3",
    
    "M0 UCB" = "#98DF8A",  
    "M1 UCB" = "#AEC7E8",  
    "M2 UCB" = "#FF9896",
    "AR UCB" = "#9999ff",
  )
  
  # Fallback colors for unknown models
  fallback_colors <- c(
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#FF9896", "#AEC7E8"
  )
  
  # Use user-supplied palette if provided
  palette_final <- palette %||% default_palette
  
  # Add fallback colors for any model_id in df not in the palette
  missing_models <- setdiff(unique(df$model_id), names(palette_final))
  if (length(missing_models) > 0) {
    extra_colors <- setNames(fallback_colors[seq_along(missing_models)], missing_models)
    palette_final <- c(palette_final, extra_colors)
  }
  
  ggplot(df, aes(x = time, y = cumulative_regret, color = model_id)) +
    geom_line(size = 0.9) +
    labs(
      title = title,
      x = "Time Step",
      y = "Cumulative Regret"
    ) +
    scale_color_manual(values = palette_final) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = "bottom",
      legend.title = element_blank()
    )
}
