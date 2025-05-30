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
    
    "M0 UCB" = "#2CA02C",
    "M1 UCB" = "#1F77B4",
    "M2 UCB" = "#D62728",
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
