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
    "AR UCB" = "#9999ff"
  )
  
  ggplot(df, aes(x = time, y = avg_regret, color = model_id)) +
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
}
