rm(list = ls())

.libPaths("/home/shaox0a/R/x86_64-pc-linux-gnu-library/4.3")
Sys.setenv(RETICULATE_PYTHON = "/usr/local/bin/python3.7")

setwd(this.path::here())

# library("deepspat")
library(tensorflow)
library(keras)
library(tfprobability)
library(dplyr)
library(fields)
library(maps)



# adapt to tf2
source("Functions/Aux_functions.R")
source("Functions/Emp_functions.R")
source("Functions/Pre_functions.R")
source("Functions/Other_functions.R")
source("Functions/Plot_functions.R")
source("Functions/NMLL_ext.R")
source("Functions/deepspat_ext_main.R")
source("Functions/predict.deepspat_ext.R")
source("Functions/save_and_load.R")

################################################################################
# load data
# --------------------------------
app_data = "UKpr"
work_dir = "AppData/"
if (!dir.exists(work_dir)) {dir.create(work_dir)}


# Download the data from https://zenodo.org/records/15459157, and place it in AppData/.
# WARNING:The whole script requires approximately 15GB memory.
# See https://github.com/shaox0a/DCSMExt for details
load("AppData/UKpr_waleswindow.rds")
# --------------------------------

lon_range=range(S[,1])
lat_range=range(S[,2])

################################################################################
# Preliminary setting
df <- cbind(S, data) %>% as.data.frame() # 
names(df) = c("s1", "s2", paste0("z", 1:(ncol(df)-2)))
df_loc = dplyr::select(df, s1, s2)
df_data = df[,3:ncol(df)]

# ---------

# standardization to Pareto(1)
trans = T
if (trans) {
  mles.matrix = fit_GPD(data, q = 0.95)
  normalized_data = stand_margins(data, mles.matrix, q = 0.95)
  
  
  # myplot(mles.matrix[,2], S, title = "Scale", colors = rev(rainbow(7)))
  # myplot(mles.matrix[,3], S, title = "Shape", colors = rev(rainbow(7)))
  
  df <- cbind(S, normalized_data) %>% as.data.frame() # 
  names(df) = c("s1", "s2", paste0("z", 1:(ncol(df)-2)))
}

df_loc = dplyr::select(df, s1, s2)
df_data = df[,3:ncol(df)]

model = "r-Pareto" 
sel.pairs = obs_extdep.emp = risk = weight_fun = dWeight_fun = NULL

risk_type = "max"           # "max", "sum2", or "site"
if (model %in% c("r-Pareto", "AI")) {
  site.index = NULL; xi0 = NULL
  if (risk_type == "max") {
    # risk_fun = risk_type #function(rep) { max(rep) }
    risk_fun = function(rep) { sum(rep^20)^{1/20} }
  } else if (risk_type == "sum") {
    risk_fun = risk_type #function(rep) { sum(rep) }
  } else if (risk_type == "site") {
    Cardiff = c(-3.179090, 51.481583)
    site_dist = rdist(S, matrix(Cardiff, ncol=2))
    site.index = which.min(site_dist)
    risk_fun = function(rep) { rep[1] }
  } else if (risk_type == "sum2") {
    xi0 = mean(mles.matrix[,3])
    risk_fun = function(rep) { sum(rep^xi0)^{1/xi0} }
  }
}
# ---------
# ## POT stability checking: r(Zt) - Zt/r(Zt)
# data_to_check = df_data
# risks = apply(data_to_check, 2, risk_fun)
# spectrals = sapply(1:ncol(data_to_check), function(t) data_to_check[,t]/risks[t])
# hist(1-1/risks)
# ---------

################################################################################
# Model fitting

# RNGkind(sample.kind = "Rounding")
seedn1 = 1
set.seed(seedn1)
D_obs = 2000

if (risk_type == "site") {
  # In this case, we include the site of interest in the observation set 
  # In this way, we no longer need to include the site index for deepspat_ext
  sam1 <- c(site.index, sample((1:nrow(df))[-site.index], D_obs-1))
} else {
  sam1 <- sample(1:nrow(df), D_obs)
}

df.obs <- df[sam1,]
obs_loc = df.obs[,c("s1", "s2")]
obs_data = df.obs[,3:ncol(df)]

q <- 0.9; q1 <- 0.95

# # functional exceedances
rexceed_obj = rexceed(obs_data, risk_fun)
obs_exc = rexceed_obj$rep_exc
threshold = rexceed_obj$threshold
# func_x = apply(obs_data, 2, risk_fun)
# threshold <- quantile(as.numeric(func_x), q)
# obs_exc <- as.matrix(obs_data[,which(func_x > threshold)])

obs_exc_all <- cbind(obs_loc, obs_exc) %>% as.data.frame() # 
names(obs_exc_all) = c("s1", "s2", paste0("z", 1:(ncol(obs_exc_all)-2)))


method = "EC"
family = "nonsta"
dtype = "float64"


obs_extdep_odist.emp = emp_extdep_est(obs_data, obs_loc, model, risk_fun, q, q1)
if (method == "EC") {
  obs_extdep.emp = 2 - obs_extdep_odist.emp[,1]
  stplen = c(0.02, 0.01, 0.01)
} else if (method == "GS") {
  weight_fun = WEIGHTS(risk_type, xi0)$weight_fun
  dWeight_fun = WEIGHTS(risk_type, xi0)$dWeight_fun
  stplen = c(0.1, 0.1, 0.01)
}



## Set up warping layers
layer_structure = "_layer1"
r1 <- 50L
if (layer_structure == "_layer1") {
  layers <- c(AWU(r = r1, dim = 1L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
              AWU(r = r1, dim = 2L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
              RBF_block(1L, dtype = dtype),
              LFT(dtype = dtype))
} else if (layer_structure == "_layer2") {
  layers <- c(AWU(r = r1, dim = 1L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
              AWU(r = r1, dim = 2L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
              RBF_block(1L, dtype = dtype),
              RBF_block(2L, dtype = dtype),
              LFT(dtype = dtype))
} else if (layer_structure == "_layer3") {
  layers <- c(AWU(r = r1, dim = 1L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
              AWU(r = r1, dim = 2L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
              RBF_block(1L, dtype = dtype))
} else if (layer_structure == "_layer4") {
  layers <- c(AWU(r = r1, dim = 1L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
              AWU(r = r1, dim = 2L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
              RBF_block(1L, dtype = dtype),
              RBF_block(2L, dtype = dtype))
}  


d1 <- deepspat_ext(f = as.formula(paste(paste(paste0("z", 1:(ncol(obs_exc_all)-2)), collapse= "+"), "~ s1 + s2 -1")),
                   data = obs_exc_all,
                   layers = layers,
                   method = method,
                   family = family,
                   dtype = dtype,
                   nsteps = 150L,
                   nsteps_pre = 100L,
                   par_init = initvars(),
                   learn_rates = init_learn_rates(eta_mean = stplen[1], eta_mean2 = stplen[2], vario = stplen[3]),
                   sel.pairs = sel.pairs,
                   extdep.emp = obs_extdep.emp,
                   risk = risk_fun,
                   weight_fun = weight_fun,
                   dWeight_fun = dWeight_fun,
                   thre = threshold
)

d1_save = to_save(d1)
saveRDS(list(d1 = d1_save,
             df = df,
             df.obs = df.obs,
             risk_fun = risk_fun,
             xi0 = xi0,
             q = q, q1 = q1), file = paste0("AppData/FittedModel_", app_data, "_", model,
                                   "_", method, "_", family, "_", risk_type, "_", D_obs, ".rds"))

################################################################################
# Summary


pred = predict.deepspat_ext(d1, df_loc, family, dtype = dtype)
S.rescaled = pred$srescaled
S.warped = pred$swarped
range_fitted = pred$fitted.phi
dof_fitted = pred$fitted.kappa
D.warped = rdist(S.warped)

# plot(as.matrix(d1$swarped))
# plot(S.warped)

# boundary
# --------------------------------------------
v1 = lon_range[1]; v2 = lon_range[2]; h1 = lat_range[1]; h2 = lat_range[2]
poles = matrix(c(v1, h1, v2, h1, v2, h2, v1, h2), ncol = 2, byrow = T)
bottom = seq(poles[1,1], poles[2,1], 0.1)
bottom = cbind(bottom, rep(poles[1,2], length(bottom)))
right = seq(poles[2,2], poles[3,2], 0.1)
right = cbind(rep(poles[2,1], length(right)), right)
top = seq(poles[3,1], poles[4,1], -0.1)
top = cbind(top, rep(poles[3,2], length(top)))
left = seq(poles[4,2], poles[1,2], -0.1)
left = cbind(rep(poles[4,1], length(left)), left)
edges = rbind(bottom, right, top, left)
boundary = rbind(edges, edges[1,])
plot(boundary, type="l")
# --------------------------------------------

# cities
# --------------------------------------------
cities = matrix(c(-3.943646,51.621441, 
                  -2.375587,52.768555,
                  -2.994781717,53.04793759,
                  -4.1293,53.2274,
                  -3.442000,53.2274,
                  -5.269000,51.882000,
                  -2.983333,53.400002,
                  -1.898575,52.489471,
                  -2.587910,51.454514,
                  -2.244644,53.483959,
                  -4.0829, 52.4153,
                  -3.179090, 51.481583), ncol=2, byrow=T)
rownames(cities) = c('Swansea',
                     'Newport',
                     'Wrexham', 
                     'Bangor',
                     'St. Asaph',
                     'St. Davids', 
                     'Liverpool', 
                     'Birmingham', 
                     'Bristol', 
                     'Manchester',
                     'Aberystwyth',
                     'Cardiff')
df_cities = data.frame(cities)[c(1,4,7,8,9,10,11,12),]
names(df_cities) = c('s1', 's2')
interest = c('Birmingham', 'Liverpool', 'Aberystwyth', 'Cardiff')

pred_cities = predict.deepspat_ext(d1, df_cities, family, dtype = dtype)
df_cities.rescaled = data.frame(pred_cities$srescaled)
df_cities.warped = data.frame(pred_cities$swarped)
names(df_cities.rescaled) = names(df_cities.warped) = c('s1', 's2')
rownames(df_cities) = rownames(df_cities.rescaled) = rownames(df_cities.warped) =
  df_cities$names = df_cities.rescaled$names = df_cities.warped$names = rownames(df_cities)

df_cities$id = df_cities.rescaled$id = df_cities.warped$id = 
  sapply(1:nrow(df_cities), function(i) {
    city = df_cities[i,1:2]
    city_dist = rdist(S, matrix(city, ncol=2))
    which.min(city_dist)
  })
# --------------------------------------------

# extract elevation 
extract_elev = T
if (extract_elev) {
  elev_extract = elevatr::get_elev_point(data.frame(x=S[,1], y=S[,2]), prj = 4326, src = "aws")
  elev = elev_extract$elevation
  elev[is.na(elev)] = 0
  # save(elev, file = paste0(app_data, "_elev.rds"))
}


# elevation contour lines
# -------------------------------
# we cannot get the contour lines over the warped space directly
# instead we extract the contour lines numerically, and warp them
df_elev = data.frame(s1 = S[,1], s2 = S[,2], elev = elev)
df_contour = contoureR::getContourLines(df_elev, nlevels = 5)
df_boundary = data.frame(s1 = boundary[,1], s2 = boundary[,2])

df_contour.warped = predict.deepspat_ext(d1, data.frame(s1 = df_contour$x, s2 = df_contour$y), 
                                         family, dtype = dtype)$swarped
df_contour$xw = df_contour.warped[,1]; df_contour$yw = df_contour.warped[,2]

df_boundary.warped = predict.deepspat_ext(d1, df_boundary,family, dtype = dtype)$swarped
df_boundary$s1w = df_boundary.warped[,1]; df_boundary$s2w = df_boundary.warped[,2]
# -------------------------------

width1 = 14
unit.w = unit(width1, "cm")
height1 = 12
unit.h = unit(height1, "cm")
ref_shap = 24
ref_shap1 = 8
axis.title.size = 18
axis.text.size = 18
legend.text.size = 18
legend.title.size = 20
text.size = 5

library(ggplot2)
library(ggpubr)
library(ggnewscale)
library(gird)
library(girdExtra)
pic_path0 = "AppData/Pic/"
if (!dir.exists(pic_path0)) {dir.create(pic_path0)}
pic_path = paste0(pic_path0, risk_type, "/")
if (!dir.exists(pic_path)) {dir.create(pic_path)}

# warped space
# ------------------------------------------------
uni.lat = seq(lat_range[1], lat_range[2], length.out = 20)
uni.lon = seq(lon_range[1], lon_range[2], length.out = 20)
verti = lapply(1:length(uni.lon), function(i) data.frame(s1 = rep(uni.lon[i], 100), s2 = seq(lat_range[1], lat_range[2], length.out = 100)))
horiz = lapply(1:length(uni.lat), function(i) data.frame(s1 = seq(lon_range[1], lon_range[2], length.out = 100), s2 = rep(uni.lat[i], 100)))
df_verti = data.frame(do.call("rbind", lapply(1:length(verti), function(i) rbind(predict.deepspat_ext(d1, verti[[i]], family, dtype = dtype)$swarped, c(NA, NA))) ))
df_horiz = data.frame(do.call("rbind", lapply(1:length(horiz), function(i) rbind(predict.deepspat_ext(d1, horiz[[i]], family, dtype = dtype)$swarped, c(NA, NA))) ))
names(df_verti) = names(df_horiz) = c("s1", "s2")

grid2 = ggplot(df_verti, aes(x=s1,y=s2)) + geom_path(colour = "gray80", linewidth = 0.4) +
  geom_path(data = df_horiz, mapping = aes(x=s1,y=s2), colour = "gray80", 
            inherit.aes = FALSE, linewidth = 0.4) +
  geom_path(data = df_contour, aes(xw,yw,group=Group, colour=z), inherit.aes = FALSE) +
  scale_color_viridis("Elevation (m)", discrete = F,
                      breaks = c(0, 200, 400, 600), labels = c(0, 200, 400, 600)) +
  geom_path(df_boundary, mapping = aes(s1w, s2w), inherit.aes = FALSE,
            color = "black", linewidth = 0.5, linetype = "dashed") +
  geom_point(aes(x = df_cities.warped['Cardiff',1], y = df_cities.warped['Cardiff',2]),
             size = 2, shape = ref_shap, fill="red", color="black")+
  geom_text(aes(x = df_cities.warped['Cardiff',1], y = df_cities.warped['Cardiff',2], label="Cardiff"), 
            size = text.size, hjust=1, vjust=-0.35) +
  geom_point(data = df_cities.warped[-which(df_cities.warped$names=='Cardiff'),], aes(s1, s2),
             size = 2, shape = 21, fill="red", color="black")+
  geom_text(data = df_cities.warped[-which(df_cities.warped$names=='Cardiff'),], aes(label=names), 
            size = text.size, hjust=1, vjust=-0.35) +
  xlab(expression(f[n1])) + ylab(expression(f[n2])) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size), 
        legend.key.size = unit(0.25, "in"), 
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size), 
        legend.position = "none", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical", 
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))
grid2
ggsave(paste0(pic_path, "warped_space_", method, "_", family, layer_structure, ".pdf"),
       plot = grid2, width = width1, height = height1, units = "cm")

# ------------------------------------------------


# pairwise CEPs
# ------------------------------------------------
# empirical extremal dependnece measure estimates
# This is time consuming!!!
emp_extdep_filename = paste0("AppData/", app_data, "_", model, "_", risk_type, "_empextdep.rds")
if (!file.exists(emp_extdep_filename)) {
  df_extdep_odist.emp = emp_extdep_est(df_data, df_loc, model, risk_fun, q, q1)
  saveRDS(df_extdep_odist.emp, emp_extdep_filename)
}
df_extdep_odist.emp = readRDS(emp_extdep_filename)


cec_dist.emp = df_extdep_odist.emp[,1]
empextdep_upmat = matrix(0, nrow(df), nrow(df))
empextdep_upmat[lower.tri(empextdep_upmat, diag=FALSE)] <- cec_dist.emp
empextdep_upmat <- t(empextdep_upmat)
empextdep_mat = empextdep_upmat + t(empextdep_upmat)
diag(empextdep_mat) = 1

rm(empextdep_upmat)

colors = RColorBrewer::brewer.pal(n=5, name="RdYlBu")[5:1]

pp.emp = pp.fit = list()
for (k in 1:4) {
  vis_city = interest[k]
  vis_id = df_cities$id[which(df_cities$name == vis_city)]
  
  # emp EC
  p.emp = eval(substitute(myplot(empextdep_mat[, vis_id], 
                                 S, legend_title = "CEP", lims = c(0,1),
                                 colors = colors) + 
                            scale_color_gradientn(colors = colors,
                                                  name = "CEP", limits = c(0,1),
                                                  breaks = c(0.00, 0.5, 1.00), 
                                                  labels = c("0.00", "0.50", "1.00"),
                                                  guide  = guide_colorbar(order = 1)) +
                            new_scale_color() + 
                            geom_path(data = df_contour, mapping = aes(x,y,group=Group, colour=z),
                                      inherit.aes = FALSE) +
                            scale_color_viridis("Elevation (m)", discrete = F,
                                                breaks = c(0, 200, 400, 600), 
                                                labels = c(0, 200, 400, 600),
                                                guide  = guide_colorbar(order = 2)) +  #c(min(df_elev[,3]), max(df_elev[,3]))
                            geom_point(aes(x = df_cities[vis_city,1], y = df_cities[vis_city,2]),
                                       size = 2, shape = 21, fill="red", color = "black") +
                            xlab("Longitude") + ylab("Latitude") +
                            theme_bw() +
                            theme(plot.title = element_text(hjust = 0.5, size=25),
                                  axis.title=element_text(size=axis.title.size),
                                  axis.text = element_text(size=axis.text.size), 
                                  legend.key.size = unit(0.25, "in"), 
                                  legend.text = element_text(size=legend.text.size),
                                  legend.title = element_text(size=legend.title.size), 
                                  legend.position = "right", #c(0.18, 0.8),
                                  legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
                                  legend.direction = "vertical", 
                                  legend.box = "vertical",
                                  legend.spacing.y = unit(0.4, "lines"),
                                  legend.margin = margin(5, 5, 5, 5)), list(vis_city = vis_city)))
  pp.emp[[k]] = p.emp
  
  # fitted EC
  EC.fit = sapply(1:nrow(D.warped), function(i) 2-EC_fun(c(range_fitted, dof_fitted), 
                                                         D.warped[df_cities$id[which(df_cities$name == vis_city)], i]))
  p.fit = eval(substitute(myplot(EC.fit, S, legend_title = "CEP", lims = c(0,1),
                                 colors = colors) + 
                            scale_color_gradientn(colors = colors,
                                                  name = "CEP", limits = c(0,1),
                                                  breaks = c(0.00, 0.5, 1.00), 
                                                  labels = c("0.00", "0.50", "1.00"),
                                                  guide  = guide_colorbar(order = 1)) +
                            new_scale_color() +
                            geom_path(data = df_contour, mapping = aes(x,y,group=Group, colour=z),
                                      inherit.aes = FALSE) +
                            scale_color_viridis("Elevation (m)", discrete = F, 
                                                breaks = c(0, 200, 400, 600), 
                                                labels = c(0, 200, 400, 600),
                                                guide  = guide_colorbar(order = 2)) +  #c(min(df_elev[,3]), max(df_elev[,3]))
                            geom_point(aes(x = df_cities[vis_city,1], y = df_cities[vis_city,2]),
                                       size = 2, shape = 21, fill="red", color = "black")+
                            xlab("Longitude") + ylab("Latitude") +
                            theme_bw() +
                            theme(plot.title = element_text(hjust = 0.5, size=25),
                                  axis.title=element_text(size=axis.title.size),
                                  axis.text = element_text(size=axis.text.size), 
                                  legend.key.size = unit(0.25, "in"), 
                                  legend.text = element_text(size=legend.text.size),
                                  legend.title = element_text(size=legend.title.size), 
                                  legend.position = "right", #c(0.18, 0.8),
                                  legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
                                  legend.direction = "vertical", 
                                  legend.box = "vertical",
                                  legend.spacing.y = unit(0.4, "lines"),
                                  legend.margin = margin(5, 5, 5, 5)) , list(vis_city = vis_city)))
  pp.fit[[k]] = p.fit
}

legend_grob <- get_legend(pp.emp[[1]])
p.extdep.emp = grid.arrange(pp.emp[[1]]+theme(legend.position = "none"),
                            pp.emp[[2]]+theme(legend.position = "none"),
                            pp.emp[[3]]+theme(legend.position = "none"),
                            pp.emp[[4]]+theme(legend.position = "none"),
                            legend_grob, ncol=5,
                            widths = unit.c(unit.w, unit.w, unit.w, unit.w, 
                                            unit(width1/3, "cm")),
                            heights = unit.c(unit.h))
ggsave(paste0(pic_path, method, "_", family, "_empextdep.pdf"),
       plot = p.extdep.emp, width = 4*width1+width1/3, height = height1, units = "cm")

legend_grob <- get_legend(pp.fit[[1]])
p.extdep.fit = grid.arrange(pp.fit[[1]]+theme(legend.position = "none"),
                            pp.fit[[2]]+theme(legend.position = "none"),
                            pp.fit[[3]]+theme(legend.position = "none"),
                            pp.fit[[4]]+theme(legend.position = "none"),
                            legend_grob, ncol=5,
                            widths = unit.c(unit.w, unit.w, unit.w, unit.w, 
                                            unit(width1/3, "cm")),
                            heights = unit.c(unit.h))
ggsave(paste0(pic_path, method, "_", family, layer_structure, "_fitextdep.pdf"),
       plot = p.extdep.fit, width = 4*width1+width1/3, height = height1, units = "cm")
# ------------------------------------------------


# pairwise CEPs against distance
# ------------------------------------------------
obs_loc.rescaled = predict.deepspat_ext(d1, obs_loc, family, dtype = dtype)$srescaled
obs_loc.warped = as.matrix(d1$swarped)

dist_obs = rdist(obs_loc.rescaled); dist_warped = rdist(obs_loc.warped)
D0 = nrow(obs_loc.rescaled)
dist_obs.pairs = dist_warped.pairs = numeric(length = (D0-1)*D0/2)
k=1
for (i in 1:(D0-1)) { for (j in (i+1):D0) {
  dist_obs.pairs[k] = dist_obs[i,j]; dist_warped.pairs[k] = dist_warped[i,j]
  k=k+1
}}

df_cloud = data.frame(CEP=obs_extdep_odist.emp[,1], 
                      distance_o=dist_obs.pairs,
                      distance_w=dist_warped.pairs)
df.line.warped = data.frame(x = seq(0,1.4,0.01),
                            y = sapply(seq(0,1.4,0.01), function(i) 2-EC_fun(c(range_fitted, dof_fitted), i)))

df_cloud_extend = df_cloud
df_cloud_extend$line_o = sapply(df_cloud_extend$distance_o, 
                                function(i) 2-EC_fun(c(range_fitted0, dof_fitted0), i))
df_cloud_extend$line_w = sapply(df_cloud_extend$distance_w, 
                                function(i) 2-EC_fun(c(range_fitted, dof_fitted), i))

mean(abs(df_cloud_extend$CEP - df_cloud_extend$line_o))
mean(abs(df_cloud_extend$CEP - df_cloud_extend$line_w))

allow <- 0.05
possible.dists = seq(0.05,1.35,0.1)


list.ec2 <- lapply(possible.dists, function(dd){
  xx <- df_cloud$CEP[(df_cloud$distance_w > (dd - allow)) & (df_cloud$distance_w < (dd + allow))]
  xx[!is.na(xx)]})

df.box2 <- data.frame(emp = Reduce(c, sapply( 1:length(possible.dists), function(i){
  rep(possible.dists[i], length(list.ec2[[i]]))}) ),
  fitted = Reduce(c, list.ec2))
line_col = ifelse(method == "EC", "red", "deepskyblue")
p.cld.warped_box = ggplot(data = df.box2, aes(x = emp, y = fitted, group = emp)) +
  stat_boxplot(geom = "errorbar") + geom_boxplot(outlier.shape = NA) +
  ylab("Theoretical Extremal Coefficient") + xlab("Empirical Extremal Coefficient") +
  geom_line(df.line.warped, mapping = aes(x=x, y=y), color = line_col, linewidth=0.5, inherit.aes = FALSE) +
  theme_bw() + xlab("Distance") + ylab("Conditional Exceedance Probability")+
  labs(colour = "Category", shape = "Category") +
  ylim(c(0,1)) + xlim(c(0,1.4)) +
  theme(plot.title = element_text(hjust = 0.5, size=25),,
        legend.key.size = unit(0.25, "in"), 
        axis.title=element_text(size=axis.title.size),
        axis.text=element_text(size=axis.text.size), 
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.text.size), 
        legend.position = c(0.75, 0.85))
p.cld.warped_box
ggsave(paste0(pic_path, method, "_", family, layer_structure, "_cloud_warped_box.pdf"),
       plot = p.cld.warped_box, width = 20, height = 12, units = "cm")
# ------------------------------------------------