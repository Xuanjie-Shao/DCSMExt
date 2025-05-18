to_save = function(d1_save) {
  d1_save$Cost = as.numeric(d1_save$Cost)
  
  if (!is.null(d1_save$transeta_tf)) {
    d1_save$transeta_tf = lapply(1:d1_save$nlayers, function(i) as.matrix(d1_save$transeta_tf[[i]]))
  }
  
  if (!is.null(d1_save$eta_tf)) {
    d1_save$eta_tf = lapply(1:d1_save$nlayers, function(i) as.matrix(d1_save$eta_tf[[i]]))
  }
  
  if (!is.null(d1_save$a_tf)) {
    d1_save$a_tf = sapply(1:8, function(i) as.numeric(d1_save$a_tf[[i]]))
  }
  
  if (!is.null(d1_save$scalings)) {
    d1_save$scalings = lapply(1:length(d1_save$scalings), function(i) list(min = as.matrix(d1_save$scalings[[i]]$min),
                                                                           max = as.matrix(d1_save$scalings[[i]]$max)))
  }
  
  d1_save$logphi_tf = as.numeric(d1_save$logphi_tf)
  d1_save$logitkappa_tf = as.numeric(d1_save$logitkappa_tf)
  d1_save$s_tf = as.matrix(d1_save$s_tf)
  d1_save$x_tf = as.matrix(d1_save$x_tf)
  
  if (!is.null(d1_save$u_tf)) {
    d1_save$u_tf = as.numeric(d1_save$u_tf)
  }
  
  if (!is.null(d1_save$loc.pairs_t_tf)) {
    d1_save$loc.pairs_t_tf = t(do.call("cbind", sapply(0:(nrow(d1_save$data)-2), function(k1){
      sapply((k1+1):(nrow(d1_save$data)-1), function(k2){ c(k1,k2) } ) } )))
  }
  
  d1_save$swarped_tf = lapply(1:length(d1_save$swarped_tf), function(i) as.matrix(d1_save$swarped_tf[[i]]))
  if (!is.null(d1_save$grad_loss)) { d1_save$grad_loss = as.numeric(d1_save$grad_loss) }
  if (!is.null(d1_save$hess_loss)) { d1_save$hess_loss = as.matrix(d1_save$hess_loss) }
  # d1_save$time
  
  d1_save
}


################################################################################
################################################################################
################################################################################
# model_load = readRDS(paste0("AppData/fitted_models/", app_data, "_", model, "_", method, "_", family, "_", risk_type, ".rds"))

to_load = function(d1_load, weight_fun = NULL, dWeight_fun = NULL, layers = NULL) {
  # d1_load = model_load$d1
  
  d1_load$Cost = tf$constant(d1_load$Cost, dtype = dtype)
  
  if (!is.null(d1_load$transeta_tf)) {
    d1_load$transeta_tf = lapply(1:d1_load$nlayers, function(i) tf$Variable(d1_load$transeta_tf[[i]], dtype = dtype))
  }
  
  if (!is.null(d1_load$eta_tf)) {
    d1_load$eta_tf = lapply(1:d1_load$nlayers, function(i) tf$constant(d1_load$eta_tf[[i]], dtype = dtype))
  }
  
  if (!is.null(d1_load$a_tf)) {
    d1_load$a_tf = lapply(1:8, function(i) tf$Variable(d1_load$a_tf[i], dtype = dtype))
  }
  
  if (!is.null(d1_load$scalings)) {
    d1_load$scalings = lapply(1:length(d1_load$scalings), function(i) list(min = tf$constant(d1_load$scalings[[i]]$min, dtype = dtype),
                                                                           max = tf$constant(d1_load$scalings[[i]]$max, dtype = dtype)))
  }
  
  d1_load$logphi_tf = tf$Variable(d1_load$logphi_tf, dtype = dtype)
  d1_load$logitkappa_tf = tf$Variable(d1_load$logitkappa_tf, dtype = dtype)
  d1_load$s_tf = tf$constant(d1_load$s_tf, dtype = dtype)
  d1_load$x_tf = tf$constant(d1_load$x_tf, dtype = dtype)
  
  if (!is.null(d1_load$u_tf)) {
    d1_load$u_tf = tf$constant(d1_load$u_tf, dtype = dtype)
  }
  
  if (!is.null(d1_load$loc.pairs_t_tf)) {
    loc.pairs_tf = tf$reshape(tf$constant(d1_load$loc.pairs_t_tf, dtype = tf$int32), 
                              c(nrow(d1_load$loc.pairs_t_tf), 2L, 1L))
    d1_load$loc.pairs_t_tf = tf$reshape(tf$transpose(loc.pairs_tf), c(2L, nrow(loc.pairs_tf), 1L))
  }
  
  d1_load$swarped_tf = lapply(1:length(d1_load$swarped_tf), function(i) tf$constant(d1_load$swarped_tf[[i]], dtype = dtype))
  if (!is.null(d1_load$grad_loss)) { tf$constant(d1_load$grad_loss, dtype = dtype) }
  if (!is.null(d1_load$hess_loss)) { tf$constant(d1_load$hess_loss, dtype = dtype) }
  
  if (!is.null(weight_fun)) { 
    d1_load$weight_fun = weight_fun
  }
  
  if (!is.null(dWeight_fun)) { 
    d1_load$dWeight_fun = dWeight_fun
  }
  
  if (!is.null(layers)) { 
    d1_load$layers = layers
  }
  
  d1_load 
  # layers and the weight funtion may need further processing
}

