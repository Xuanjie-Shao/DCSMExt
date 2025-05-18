library(RColorBrewer)
library(ggplot2)

myplot = function(x, loc, title = "", xlab = "longitude", ylab = "latitude", 
                  legend_title = "", lims = NULL,
                  colors = RColorBrewer::brewer.pal(n=9, name="Blues")[9:1]) {
  if (is.null(lims)) {
    lims <- c(min(x), max(x))
  }
  
  dataToPlot = data.frame(X = loc[, 1], Y = loc[, 2], Value = x)
  mapPoints <- ggplot() +
    geom_point(
      data = dataToPlot,
      aes(x = X, y = Y, color = Value),
      alpha = 0.9,
      # size = 0.05
      size = 2,
      shape = 15
    ) +
    scale_color_gradientn(colors = colors, 
                          limits = lims,
                          name = legend_title) +
    labs(title = title, fill = "", x = xlab, y = ylab) +
    # guides(color = guide_colorbar(barwidth = 1)) +
    theme(plot.title = element_text(hjust = 0.5), legend.title = element_blank()) +
    theme_bw()
  # print(mapPoints)
  # rev(rainbow(7))
  
  return(mapPoints)
}