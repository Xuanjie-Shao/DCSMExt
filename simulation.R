rm(list = ls())

.libPaths("/home/shaox0a/R/x86_64-pc-linux-gnu-library/4.3")
Sys.setenv(RETICULATE_PYTHON = "/usr/local/bin/python3.7")

setwd(this.path::here())

library(tensorflow)
library(tfprobability)
library(dplyr)
library(fields)

# adapt to tf2
source("Functions/Sim_functions.R")
source("Functions/Aux_functions.R")
source("Functions/Emp_functions.R")
source("Functions/Pre_functions.R")
source("Functions/Other_functions.R")
source("Functions/Plot_functions.R")
source("Functions/NMLL_ext.R")
source("Functions/deepspat_ext_main.R")
source("Functions/predict.deepspat_ext.R")

# WARNING:The whole script requires approximately 6GB memory.
################################################################################
type = "AWU_RBF_2D" # data generating architecture
model = "r-Pareto"

sel.pairs = obs_extdep.emp = risk = weight_fun = dWeight_fun = NULL
################################################################################
# Simulate data
phi = 0.2; kappa = 1
trep = 5000L; ds = 0.01
n = (1/ds+1)^2 #10201

risk_type = "site"           # "max", "sum", or "site"
if (model %in% c("r-Pareto", "AI")) {
  site.index = NULL
  if (risk_type == "max") {
    # WARNING: data generation using max functional is emteremely slow for high dimension
    risk_fun = risk_type
    # risk_fun = function(rep) { sum(rep^20)^{1/20} }
  } else if (risk_type == "sum") {
    risk_fun = risk_type
  } else if (risk_type == "site") {
    site.index = 5000 #round(n/2)
    risk_fun = function(rep) { rep[1] }
  }
}

# RNGkind(sample.kind = "Rounding")
space_seed = 7
filename = paste0("SimData/", model, "_", risk_type, "+", type, "_", space_seed, "+", trep, "_", n, "_range(", phi, ").rds")
if (!file.exists(filename)) {
  set.seed(34543)
  sim <- sim_data(type = type, model = model, ds = ds, n_obs = 100, trep = trep, phi = phi, kappa = kappa,
                  risk = risk_type, siteindex = site.index, space_seed = space_seed, nCores = 1, cl = NULL)
  saveRDS(sim, filename)
}
sim = readRDS(filename)

plot(sim$swarped)

S = sim$s; data = sim$f_true

df <- cbind(S, data) %>% as.data.frame() # 
names(df) = c("s1", "s2", paste0("z", 1:(ncol(df)-2)))

df_loc = dplyr::select(df, s1, s2)
df_data = df[,3:ncol(df)]

################################################################################
# Model fitting

# RNGkind(sample.kind = "Rounding")
seedn1 = 1
set.seed(seedn1)
D_obs = 500

if (risk_type == "site") {
  # In this case, we include the site of interest in the observation set 
  # In this way, we no longer need to include the site index for deepspat_ext
  sam1 <- c(site.index, sample((1:nrow(df))[-site.index], D_obs-1))
} else {
  sam1 <- sample(1:nrow(df), D_obs)
}


obs_all <- df[sam1,]
obs_loc = obs_all[,c("s1", "s2")]
obs_data = obs_all[,3:ncol(df)]

q <- 0.95; q1 <- 0.95


# functional exceedances
rexceed_obj = rexceed(obs_data, risk_fun)
obs_exc = rexceed_obj$rep_exc
threshold = rexceed_obj$threshold
# obs_func_risk = apply(obs_data, 2, risk_fun)
# threshold <- quantile(obs_func_risk, q)
# id_exc = which(obs_func_risk > threshold)
# obs_exc <- as.matrix(obs_data[,id_exc])


obs_exc_all <- cbind(obs_loc, obs_exc) %>% as.data.frame() # 
names(obs_exc_all) = c("s1", "s2", paste0("z", 1:(ncol(obs_exc_all)-2)))


method = "GS"
family = "nonsta"
dtype = "float64"



if (method == "EC") {
  obs_extdep_odist.emp = emp_extdep_est(obs_data, obs_loc, model, risk_fun, q, q1)
  obs_extdep.emp = 2 - obs_extdep_odist.emp[,1]
  if (sum(is.na(obs_extdep.emp)) != 0) {
    stop("WARNING: NaN in empirical extremal dependence measure.")
  }
} else if (method == "GS") {
  weight_fun = WEIGHTS(risk_type, xi0, weight_type =1)$weight_fun
  dWeight_fun = WEIGHTS(risk_type, xi0, weight_type =1)$dWeight_fun
}
stplen = c(0.1, 0.05, 0.01)

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
                   nsteps = 50L, nsteps_pre = 50L,
                   par_init = initvars(),
                   learn_rates = init_learn_rates(eta_mean = stplen[1], eta_mean2 = stplen[2], vario = stplen[3]),
                   sel.pairs = sel.pairs,
                   extdep.emp = obs_extdep.emp,
                   risk = risk_fun,
                   weight_fun = weight_fun,
                   dWeight_fun = dWeight_fun,
                   thre = threshold
)


################################################################################
# Summary

pred = predict.deepspat_ext(d1, df_loc, family, dtype = dtype)
S = df_loc
S.rescaled = pred$srescaled
S.warped = pred$swarped
range_fitted = pred$fitted.phi
dof_fitted = pred$fitted.kappa

S.warped_tru = sim$swarped
D.true = rdist(S.warped_tru)
D.warped = rdist(S.warped)

plot(as.matrix(d1$swarped))
plot(S.warped)


# site.index = 5000
ref_ids = c(2000, 5000, 7000) # they can also be sampled
df_refpts = data.frame(S[ref_ids,])
names(df_refpts) = c('s1', 's2')
df_refpts.warped = data.frame(S.warped[ref_ids,])
names(df_refpts.warped) = c('s1', 's2')
df_refpts$id = df_refpts.warped$id = ref_ids

width1 = 14
width2 = 12
width3 = 11.5
height1 = 12
ref_shap = 24
ref_shap1 = 8
axis.title.size = 18
axis.text.size = 18
legend.text.size = 18
legend.title.size = 20

library(ggplot2)
library(ggpubr)
pic_path0 = "SimData/Pic/"
if (!dir.exists(pic_path0)) {dir.create(pic_path0)}
pic_path = paste0(pic_path0, risk_type, "/")
if (!dir.exists(pic_path)) {dir.create(pic_path)}
# ---------
# True warped space
df0 <- data.frame(s1 = S[, 1],
                  s2 = S[, 2],
                  h1 = S.warped_tru[, 1],
                  h2 = S.warped_tru[, 2]) %>%
  mutate(s1c = as.integer(round(s1*100)),
         s2c = as.integer(round(s2*100)))
checkers0 <- polygons_from_points(df0, every = 3)
count <- length(unique(checkers0$id))
grid0.chess <- (ggplot(checkers0) + geom_polygon(aes(x, y, group = id,
                                                     fill = as.logical((id + floor((id-0.5)/sqrt(count))) %% 2)), colour="black") +
                  scale_fill_grey(start = 0.1, end = 0.9) + theme_bw())  +
  guides(fill="none", alpha = FALSE) +
  xlab(expression(f[n1])) + ylab(expression(f[n2]))  + coord_fixed(ratio = 1)  +
  theme(text = element_text(size=20),
        axis.title=element_blank()) +
  geom_point(data = data.frame(s1 = S.warped_tru[ref_ids,1], 
                               s2 = S.warped_tru[ref_ids,2]), aes(s1, s2),
             size = 3, shape = 21, fill="red", color="white") +
  geom_point(aes(x = S.warped_tru[site.index,1],
                 y = S.warped_tru[site.index,2]), 
             size = 3, shape = ref_shap, fill="red", color="white")
grid0.chess
ggsave(paste0(pic_path, "true_space_chess.pdf"),
       plot = grid0.chess, width = width1, height = height1, units = "cm")

# estimated warped space
df2 <- data.frame(s1 = S[, 1],
                  s2 = S[, 2],
                  h1 = S.warped[, 1],
                  h2 = S.warped[, 2]) %>%
  mutate(s1c = as.integer(round(s1*100)),
         s2c = as.integer(round(s2*100)))

checkers2 <- polygons_from_points(df2, every = 3)
count <- length(unique(checkers2$id))
grid2.chess <- (ggplot(checkers2) + geom_polygon(aes(x, y, group = id,
                                                     fill = as.logical((id + floor((id-0.5)/sqrt(count))) %% 2)), colour="black") +
                  scale_fill_grey(start = 0.1, end = 0.9) + theme_bw())  +
  guides(fill="none", alpha = FALSE) +
  xlab(expression(f[n1])) + ylab(expression(f[n2]))  + coord_fixed(ratio = 1)  +
  theme(text = element_text(size=20), axis.title=element_blank()) +
  geom_point(data = data.frame(s1 = df_refpts.warped[,1], 
                               s2 = df_refpts.warped[,2]), aes(s1, s2),
             size = 3, shape = 21, fill="red", color="white") +
  geom_point(aes(x = df_refpts.warped[which(ref_ids == site.index),1],
                 y = df_refpts.warped[which(ref_ids == site.index),2]),
             size = 3, shape = ref_shap, fill="red", color="white")
grid2.chess

ggsave(paste0(pic_path, method, "_", family, layer_structure, "_warped_space_chess.pdf"),
       plot = grid2.chess, width = width1, height = height1, units = "cm")

# ---------
# pairwise CEPs

colors = RColorBrewer::brewer.pal(n=5, name="RdYlBu")[5:1]
pp.fit = pp.tru = list()
K = length(ref_ids)
for (k in 1:K) {
  message(k)
  vis_id = ref_ids[k]
  
  # fitted EC
  EC.fit = sapply(1:nrow(D.warped), function(i) 2-EC_fun(c(range_fitted, dof_fitted), 
                                                         D.warped[vis_id, i]))
  df_ec2 = data.frame(s1 = S[, 1], s2 = S[, 2], ec = EC.fit)
  p.fit = eval(substitute(myplot(EC.fit, S, legend_title = "CEP", lims = c(0,1),
                                 colors = colors) + 
                            scale_color_gradientn(colors = colors,
                                                  name = "CEP", limits = c(0,1),
                                                  breaks = c(0.00, 0.5, 1.00), labels = c("0.00", "0.50", "1.00")) +
                            geom_point(aes(x = df_refpts[k,1], y = df_refpts[k,2]),
                                       colour = "black", size = 3, shape = ref_shap1)+
                            xlab(expression(s[1])) + ylab(expression(s[2])) +
                            theme_bw() +
                            theme(plot.title = element_text(hjust = 0.5, size=25),
                                  # axis.title=element_text(size=axis.title.size), 
                                  axis.title = element_blank(),
                                  axis.text = element_text(size=axis.text.size), 
                                  legend.key.size = unit(0.3, "in"), 
                                  legend.text = element_text(size=legend.text.size),
                                  legend.title = element_text(size=legend.title.size), 
                                  legend.position = "none", #c(0.18, 0.7),
                                  legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
                                  legend.direction = "vertical", 
                                  legend.box = "vertical",
                                  legend.spacing.y = unit(0.4, "lines"),
                                  legend.margin = margin(1, 1, 1, 1)), list(k = k)))
  pp.fit[[k]] = p.fit
  
  
  EC.tru = sapply(1:nrow(D.true), function(i) 2-EC_fun(c(0.2, 1), 
                                                       D.true[vis_id, i]))
  df_ec2 = data.frame(s1 = S[, 1], s2 = S[, 2], ec = EC.tru)
  p.tru = eval(substitute(myplot(EC.tru, S, legend_title = "CEP", lims = c(0,1),
                                 colors = colors) + 
                            scale_color_gradientn(colors = colors,
                                                  name = "CEP", limits = c(0,1),
                                                  breaks = c(0.00, 0.5, 1.00), labels = c("0.00", "0.50", "1.00")) +
                            geom_point(aes(x = df_refpts[k,1], y = df_refpts[k,2]),
                                       colour = "black", size = 3, shape = ref_shap1)+
                            xlab("Longitude") + ylab("Latitude") +
                            theme_bw() +
                            theme(plot.title = element_text(hjust = 0.5, size=25),
                                  # axis.title=element_text(size=axis.title.size), 
                                  axis.title = element_blank(),
                                  axis.text = element_text(size=axis.text.size), 
                                  legend.key.size = unit(0.3, "in"), 
                                  legend.text = element_text(size=legend.text.size),
                                  legend.title = element_text(size=legend.title.size), 
                                  legend.position = "none", #c(0.18, 0.7),
                                  legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
                                  legend.direction = "vertical", 
                                  legend.box = "vertical",
                                  legend.spacing.y = unit(0.4, "lines"),
                                  legend.margin = margin(1, 1, 1, 1)), list(k = k)))
  
  pp.tru[[k]] = p.tru
}

p.extdep.fit = ggarrange(pp.fit[[1]], pp.fit[[2]], pp.fit[[3]], 
                         nrow = 1, ncol = K, common.legend = TRUE,
                         legend = "right")
ggsave(paste0(pic_path, method, "_", family, layer_structure, "_fitextdep.pdf"),
       plot = p.extdep.fit, width = K*width2+width2/6, height = height1, units = "cm")


p.extdep.tru = ggarrange(pp.tru[[1]], pp.tru[[2]], pp.tru[[3]], 
                         nrow = 1, ncol = K, common.legend = TRUE,
                         legend = "right")
ggsave(paste0(pic_path, method, "_", family, layer_structure, "_truextdep.pdf"),
       plot = p.extdep.tru, width = K*width2+width2/6, height = height1, units = "cm")

# ---------
# pairwise CEPs against distance
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

obs_extdep_odist.emp = emp_extdep_est(obs_data, obs_loc, model, risk_fun, q, q1)
df_cloud = data.frame(CEP=obs_extdep_odist.emp[,1], 
                      distance_o=dist_obs.pairs,
                      distance_w=dist_warped.pairs)
str(df_cloud)             

cloud_samp = sample(1:nrow(df_cloud), 10000)
df_cloud_plot =df_cloud[cloud_samp,]
df.line.tru = data.frame(x = seq(0,1.4,0.01),
                         y = sapply(seq(0,1.4,0.01), function(i) 2-EC_fun(c(0.2, 1), i)))
df.line.warped = data.frame(x = seq(0,1.4,0.01),
                            y = sapply(seq(0,1.4,0.01), function(i) 2-EC_fun(c(range_fitted, dof_fitted), i)))



df_cloud_plot1 = data.frame(distance = c(df_cloud_plot$distance_o, df_cloud_plot$distance_w),
                            CEP = c(df_cloud_plot$CEP, df_cloud_plot$CEP), 
                            type = c(rep("o", nrow(df_cloud_plot)), rep("w", nrow(df_cloud_plot))),
                            alpha = c(rep(1, nrow(df_cloud_plot)), rep(0.5, nrow(df_cloud_plot))))
line_col = ifelse(method == "EC", "red", "deepskyblue")
p.cld.warped = ggplot(df_cloud_plot1, aes(x=distance, y=CEP, 
                                          colour=type, shape=type)) + 
  geom_point(aes(alpha = alpha))+
  scale_color_manual(values = c("o" = "gray60", "w" = "black"),
                     labels = c("Rescaled original space", "Warped space")) +
  scale_shape_manual(values = c("o" = 1, "w" = 2),
                     labels = c("Rescaled original space", "Warped space"))+
  scale_alpha_continuous(range = c(0.5,1)) +
  guides(alpha = "none") +
  geom_line(df.line.warped, mapping = aes(x=x, y=y), color = line_col, linewidth=0.5, inherit.aes = FALSE) +
  geom_line(df.line.tru, mapping = aes(x=x, y=y), color = "orange", linewidth=0.5, inherit.aes = FALSE) +
  theme_bw() + xlab("Distance") + ylab("Conditional Exceedance Probability")+
  labs(colour = "Category", shape = "Category") +
  xlim(0, sqrt(2)) +
  theme(plot.title = element_text(hjust = 0.5, size=25),,
        legend.key.size = unit(0.3, "in"), 
        axis.title=element_text(size=axis.title.size),
        axis.text=element_text(size=axis.text.size), 
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.text.size), 
        legend.position = c(0.75, 0.85))

p.cld.warped

ggsave(paste0(pic_path, method, "_", family, layer_structure, "cloud_warped.pdf"),
       plot = p.cld.warped, width = 20, height = 12, units = "cm")
