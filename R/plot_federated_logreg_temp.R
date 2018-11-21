

plot_temp = function (obj, plot_title = "Federated Learning with Logistic Regression", show_averaging = FALSE, X_clean = NULL)
{
  X              =  obj[["data"]]
  data_plot_all  =  obj[["trace"]]
  coef_glm       =  obj[["glm_model"]]
  glm_single     =  obj[["glm_single"]]
  max_iters      =  obj[["max_iters"]]
  true_beta      =  obj[["true_beta"]]

  data_iters = data_plot_all %>% 
    group_by(ItersAtOnce, LearningRate) %>% 
    filter(row_number() == n()) %>% 
    select(Iteration) %>% 
    mutate(Iteration = paste0("Trained Iterations: ", Iteration)) %>%
    group_by(ItersAtOnce, LearningRate)

  # Get final points per dataset:
  final_points = data_plot_all %>% 
    group_by(Dataset, ItersAtOnce, LearningRate) %>%
    filter(row_number() == n()) %>%
    group_by(ItersAtOnce, LearningRate) %>%
    summarize(Feature1 = mean(Feature1), Feature2 = mean(Feature2))

  data_all = data_plot_all %>% 
    group_by(Dataset, ItersAtOnce, LearningRate) 

  if (! show_averaging) {
    data_all = data_all %>%
      filter(Step == "Update" | row_number() == 1) # %>%
      # filter(row_number() %in% seq(1, n(), by = 1))
  }
  data_all_final = data_plot_all %>% 
    group_by(Dataset, ItersAtOnce, LearningRate) %>%
    filter(Step == "Update") %>%
    filter(row_number() == n())

  data_global_model = data_plot_all %>%
    filter(Step == "Averaging" | row_number() == 1) %>%
    filter(Dataset == "Dataset1")
    # group_by(ItersAtOnce, LearningRate) %>%
    # filter(row_number() %in% seq(1, n(), by = 1))

  plot_range1 = c(min(c(data_all$Feature1, glm_single$Feature1, true_beta[1])), max(c(data_all$Feature1, glm_single$Feature1, true_beta[1])))
  margin1 = abs(diff(plot_range1)) * 0.1
  plot_range2 = c(min(c(data_all$Feature2, glm_single$Feature2, true_beta[2])), max(c(data_all$Feature2, glm_single$Feature2, true_beta[2])))
  margin2 = abs(diff(plot_range2)) * 0.1

  if (! is.null(X_clean[1])) {
    X = as.matrix(X_clean)
  }

  # Calculate data frame with the surface:
  data_surface = expand.grid(
    Feature1 = seq(plot_range1[1] - margin1, plot_range1[2] + margin1, length.out = 100), 
    Feature2 = seq(plot_range2[1] - margin2, plot_range2[2] + margin2, length.out = 100)
    )
  data_surface$risk = apply(data_surface, 1, function (params) { 
    empiricalRisk(
      binomialLossFun(X = X, y = y, beta = params, activation = sigmoid)
    )
  })

  gg = ggplot() + 
    geom_raster(data = data_surface, aes(x = Feature1, y = Feature2, fill = risk), show.legend = FALSE) +
    geom_contour(data = data_surface, aes(x = Feature1, y = Feature2, z = risk), colour = "white", alpha = 0.5, size = 0.5) +
    geom_path(data = data_all, aes(x = Feature1, y = Feature2, colour = Dataset), alpha = 0.5) + 
    geom_point(data = data_all_final, aes(x = Feature1, y = Feature2, colour = Dataset), shape = 4, show.legend = FALSE) + 
    geom_path(data = data_global_model, aes(x = Feature1, y = Feature2), colour = "white") + 
    geom_point(data = final_points, aes(x = Feature1, y = Feature2), colour = "white", shape = 4, size = 2) +
    geom_text(data = final_points, aes(x = Feature1, y = Feature2, label = "FedAvg"), colour = "white", vjust = -1) +
    annotate("point", x = coef_glm[1], y = coef_glm[2], colour = rgb(205, 150, 205, 255, maxColorValue = 255), shape = 4, size = 2) + 
    annotate("text", x = coef_glm[1], y = coef_glm[2], label = "GLM", vjust = -1, colour = rgb(205, 150, 205, 255, maxColorValue = 255)) + 
    geom_point(data = glm_single, aes(x = Feature1, y = Feature2, colour = dataset), show.legend = FALSE) +
    geom_text(data = data_iters, aes(x = -Inf, y = Inf, label = Iteration), colour = "black", hjust = 0, vjust = 1) + #, hjust = 0.5, vjust = 1.2) +
    theme_tufte() +
    scale_color_brewer(palette = "Spectral") +
    # facet_grid(rows = vars(ItersAtOnce), cols = vars(LearningRate)) +
    labs(title = plot_title) +
    xlab("Parameter 1") +
    ylab("Parameter 2") +
    xlim(plot_range1[1] - margin1, plot_range1[2] + margin1) +
    ylim(plot_range2[1] - margin2, plot_range2[2] + margin2 * 1.8) +
    annotate("point", x = data_plot_all[1,1], y = data_plot_all[1,2], colour = "white") +
    guides(colour = guide_legend(override.aes = list(fill = NA))) +
    theme(panel.background = element_blank())

  if (! is.null(true_beta[1])) {
    gg = gg + 
      annotate("point", x = true_beta[1], y = true_beta[2], colour = rgb(164, 211, 238, 255, maxColorValue = 255), shape = 4, size = 2) # + 
      # annotate("text", x = true_beta[1], y = true_beta[2], label = "True Parameter", vjust = -1, colour = rgb(164, 211, 238, 255, maxColorValue = 255))
  }
  return (gg)
}