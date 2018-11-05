# ================================================================================= #
#                                                                                   #
#                               Plot Federated Model                                #
#                                                                                   #
# ================================================================================= #

getFittingTrace = function (X, y, index_list, max_iters, vec_iters_at_once, vec_learning_rates, true_beta = NULL)
{
  X = as.matrix(X)

  checkmate::expect_matrix(X, ncols = 2, any.missing = FALSE)
  checkmate::expect_list(index_list)
  checkmate::expect_numeric(y, len = nrow(X), any.missing = FALSE)
  checkmate::expect_numeric(vec_iters_at_once, min.len = 1, any.missing = FALSE)
  checkmate::expect_numeric(vec_learning_rates, min.len = 1, any.missing = FALSE)
  checkmate::expect_numeric(true_beta, null.ok = TRUE, len = 2, any.missing = FALSE)

  # Calculate glm on global dataset to get best parameters which we want to learn: 
  coef_glm = glm(y ~ 0 + X, family = binomial)$coef

  pb = txtProgressBar(min = min(vec_iters_at_once), max = max(vec_iters_at_once), style = 3)

  datasets       =  lapply(index_list, function (idx) { X[idx, ] })
  responses      =  lapply(index_list, function (idx) { y[idx] })
  data_plot_all  =  data.frame()

  message("\n>> Setting up ", length(datasets), " datasets.")
  message(">> Calculating ", length(vec_iters_at_once) * length(vec_learning_rates), " federated learning models.\n")

  for (iters_at_once in vec_iters_at_once) {
    for (learning_rate in vec_learning_rates) {

      setTxtProgressBar(pb, iters_at_once)

      coef_gd       =  rep(0, ncol(X))
      global_model  =  coef_gd
      list_coef     =  lapply(index_list, function (dummy) { coef_gd })
      iterations    =  seq_len(trunc(max_iters / iters_at_once) + 1)
      real_iters    =  numeric(2 * length(iterations) + 1)
      trace_coef    =  lapply(index_list, function (dummy) { matrix(NA, nrow = 2 * length(iterations) + 1, ncol = length(coef_gd)) })

      insert_index = 1

      for (i in iterations) {

        for (k in seq_along(datasets)) {
          trace_coef[[k]][insert_index, ] = global_model
        }
        insert_index = insert_index + 1
        real_iters[insert_index] = i * iters_at_once

        # Conduct gradient descent steps on each dataset:
        for (k in seq_along(datasets)) {
          list_coef[[k]] = gradDescent(X = datasets[[k]], y = responses[[k]], beta_start = global_model, 
            learning_rate = learning_rate, epochs = iters_at_once, loss_grad = binomialLossGrad, activation = sigmoid)
          trace_coef[[k]][insert_index, ] = list_coef[[k]]
        }
        insert_index = insert_index + 1
        real_iters[insert_index] = i * iters_at_once

        # Apply Federated Averaging:
        global_model = unlist(apply(X = do.call(rbind, list_coef), MARGIN = 2, function (x) {
          weighted.mean(x = x, w = unlist(lapply(responses, length)))
        }))

        if (i == last(iterations)) {
          for (k in seq_along(datasets)) {
            trace_coef[[k]][insert_index, ] = global_model
          }
        }
      }

      data_traces = lapply(trace_coef,  as.data.frame)
      for (k in seq_along(datasets)) {
        data_traces[[k]]$set   =  paste0("Dataset", k)
        data_traces[[k]]$step  =  c(rep(c("Averaging", "Update"), times = length(iterations)), "Averaging")
        data_traces[[k]]$iter  =  real_iters
      }
      data_traces = do.call(rbind, data_traces)
      data_traces$learning_rate = learning_rate
      data_traces$iters_at_once = iters_at_once

      names(data_traces) = c("Intercept", "Slope", "Dataset", "Step", "Iteration", "LearningRate", "ItersAtOnce")

      data_plot_all = rbind(data_plot_all, data_traces)
    }
  }
  close(pb)

  # Calculate single glm models:
  glm_single = list()
  for (i in seq_along(datasets)) {
    glm_single[[i]] = coef(glm(responses[[i]] ~ 0 + datasets[[i]], family = binomial))
    names(glm_single[[i]]) = c("Feature1", "Feature2") 
  }
  glm_single = as.data.frame(do.call(rbind, glm_single))
  glm_single$dataset = paste0("Dataset", seq_along(index_list))

  out = list(data = X, trace = data_plot_all, glm_model = coef_glm, glm_single = glm_single, 
    max_iters = max_iters, true_beta = true_beta)
  class(out) = "fedLearn"

  return(out)
}




plot.fedLearn = function (obj, ...)
{
  X              =  obj[["data"]]
  data_plot_all  =  obj[["trace"]]
  coef_glm       =  obj[["glm_model"]]
  glm_single     =  obj[["glm_single"]]
  max_iters      =  obj[["max_iters"]]
  true_beta      =  obj[["true_beta"]]

  # Get final points per dataset:
  final_points = data_plot_all %>% 
    group_by(Dataset, ItersAtOnce, LearningRate) %>%
    filter(row_number() == n()) %>%
    group_by(ItersAtOnce, LearningRate) %>%
    summarize(Intercept = mean(Intercept), Slope = mean(Slope))

  data_all = data_plot_all %>% 
    group_by(Dataset, ItersAtOnce, LearningRate) %>%
    filter(Step == "Update" | row_number() == 1) # %>%
    # filter(row_number() %in% seq(1, n(), by = 1))

  data_all_final = data_plot_all %>% 
    group_by(Dataset, ItersAtOnce, LearningRate) %>%
    filter(Step == "Update") %>%
    filter(row_number() == n())

  data_global_model = data_plot_all %>%
    filter(Step == "Averaging" | row_number() == 1) %>%
    filter(Dataset == "Dataset1")
    # group_by(ItersAtOnce, LearningRate) %>%
    # filter(row_number() %in% seq(1, n(), by = 1))

  plot_range1 = c(min(c(data_all$Intercept, glm_single$Feature1)), max(c(data_all$Intercept, glm_single$Feature1)))
  margin1 = abs(diff(plot_range1)) * 0.1
  plot_range2 = c(min(c(data_all$Slope, glm_single$Feature2)), max(c(data_all$Slope, glm_single$Feature2)))
  margin2 = abs(diff(plot_range2)) * 0.1

  # Calculate data frame with the surface:
  data_surface = expand.grid(
    Intercept = seq(plot_range1[1] - margin1, plot_range1[2] + margin1, length.out = 100), 
    Slope = seq(plot_range2[1] - margin2, plot_range2[2] + margin2, length.out = 100)
    )
  data_surface$risk = apply(data_surface, 1, function (params) { 
    empiricalRisk(
      binomialLossFun(X = X, y = y, beta = params, activation = sigmoid)
    )
  })

  gg = ggplot() + 
    geom_raster(data = data_surface, aes(Intercept, Slope, fill = risk), show.legend = TRUE) +
    geom_contour(data = data_surface, aes(Intercept, Slope, z = risk), colour = "white", alpha = 0.5, size = 0.5) +
    geom_path(data = data_all, aes(x = Intercept, y = Slope, colour = Dataset), alpha = 0.5) + 
    geom_point(data = data_all_final, aes(x = Intercept, y = Slope, colour = Dataset), shape = 4, show.legend = FALSE) + 
    geom_path(data = data_global_model, aes(x = Intercept, y = Slope), colour = "white") + 
    geom_point(data = final_points, aes(x = Intercept, y = Slope), colour = "white", shape = 4, size = 2) +
    geom_text(data = final_points, aes(x = Intercept, y = Slope, label = "FedAvg"), colour = "white", vjust = -1) +
    annotate("point", x = coef_glm[1], y = coef_glm[2], colour = rgb(205, 150, 205, 255, maxColorValue = 255), shape = 4, size = 2) + 
    annotate("text", x = coef_glm[1], y = coef_glm[2], label = "GLM", vjust = -1, colour = rgb(205, 150, 205, 255, maxColorValue = 255)) + 
    geom_point(data = glm_single, aes(x = Feature1, y = Feature2, colour = dataset), show.legend = FALSE) +
    theme_tufte() +
    scale_color_brewer(palette = "Spectral") +
    facet_grid(rows = vars(ItersAtOnce), cols = vars(LearningRate)) +
    labs(title = "Centralized vs. Decentralized Learning", 
     subtitle = paste0("Example using Logistic Regression and ", max_iters, " iterations")) +
    xlab("Feature 1") +
    ylab("Feature 2") +
    xlim(plot_range1[1] - margin1, plot_range1[2] + margin1) +
    ylim(plot_range2[1] - margin2, plot_range2[2] + margin2) +
    annotate("point", x = 0, y = 0, colour = "white")

  if (is.null(true_beta[1])) {
    gg = gg + 
      annotate("point", x = true_beta[1], y = true_beta[2], colour = rgb(164, 211, 238, 255, maxColorValue = 255), shape = 4, size = 2) + 
      annotate("text", x = true_beta[1], y = true_beta[2], label = "GLM", vjust = -1, colour = rgb(164, 211, 238, 255, maxColorValue = 255))
  }

  return (gg)
}