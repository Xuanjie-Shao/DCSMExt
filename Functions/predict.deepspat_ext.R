predict.deepspat_ext <- function(object, newloc, family, dtype = "float32", ...) {

  d <- object
  mmat <- model.matrix(update(d$f, NULL ~ .), newloc)
  s_tf <- tf$constant(mmat, dtype = dtype, name = "s")
  s_in <- scale_0_5_tf(s_tf, d$scalings[[1]]$min, d$scalings[[1]]$max, dtype)
  
  # warped space
  if (family == "sta") {
    s_out = s_in
  } else if (family == "nonsta") {
    h_tf <- list(s_in)
    # ---
    if(d$nlayers > 1) for(i in 1:d$nlayers) {
      if (d$layers[[i]]$name == "LFT") {
        h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$layers[[i]]$inum(d$a_tf))
      } else { 
        h_tf[[i + 1]] <- d$layers[[i]]$f(h_tf[[i]], d$eta_tf[[i]]) 
      }
      h_tf[[i + 1]] <- scale_0_5_tf(h_tf[[i + 1]], 
                                    d$scalings[[i + 1]]$min, 
                                    d$scalings[[i + 1]]$max, 
                                    dtype = dtype)
    }

    s_out = h_tf[[d$nlayers + 1]]
  }
  
  fitted.phi = as.numeric(exp(d$logphi_tf))
  fitted.kappa = as.numeric(2*tf$sigmoid(d$logitkappa_tf))
  
  list(srescaled = as.matrix(s_in),
       swarped = as.matrix(s_out),
       fitted.phi = fitted.phi, 
       fitted.kappa = fitted.kappa)
}