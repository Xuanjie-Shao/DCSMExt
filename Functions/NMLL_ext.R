lplike <- function(logphi_tf, logitkappa_tf, transeta_tf, a_tf, scalings, 
                   s_tf, x_tf, ndata, method, family = "nonsta",
                   extdep.emp_tf = NULL, sel.pairs_tf = NULL) {
  ## 
  
  nrepli = ncol(x_tf)
  
  # transform sites
  s_in = s_tf
  if (family == "nonsta") {
    nlayers = length(layers)
    eta_tf <- swarped_tf <- list()
    swarped_tf[[1]] <- s_tf
    if(nlayers > 1) for(i in 1:nlayers) {
      # need to adapt this for LFT layer
      if (layers[[i]]$name == "LFT") {
        a_inum_tf = layers[[i]]$inum(a_tf)
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], a_inum_tf)
      } else { 
        eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]]) 
      }
      scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
      swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max, dtype = dtype)
    }
    
    s_in = swarped_tf[[nlayers+1]]
  }
  
  
  # ----------------------------------------------------------------------------
  
  phi_tf = tf$exp(logphi_tf)
  kappa_tf = 2*tf$sigmoid(logitkappa_tf)
  
  sel.pairs_t_tf = tf$reshape(tf$transpose(sel.pairs_tf), c(2L, nrow(sel.pairs_tf), 1L))
  k1_tf = sel.pairs_t_tf[[0]]; k2_tf = sel.pairs_t_tf[[1]]
  
  s1.pairs_tf = tf$gather_nd(s_in, indices = k1_tf)
  s2.pairs_tf = tf$gather_nd(s_in, indices = k2_tf)
  diff.pairs_tf = tf$subtract(s1.pairs_tf, s2.pairs_tf)
  D.pairs_tf = tf$norm(diff.pairs_tf, ord='euclidean', axis = 1L)
  
  a.pairs_tf = tf$sqrt( 2*tf$pow(D.pairs_tf/phi_tf, kappa_tf) )
  a.pairs.rep_tf = tf$transpose(tf$tile(tf$expand_dims(a.pairs_tf, axis=1L), c(1L, nrepli)))
  
  # -------------------------------------
  z1_tf = tf$transpose(tf$gather_nd(x_tf, k1_tf))
  z2_tf = tf$transpose(tf$gather_nd(x_tf, k2_tf))
  # z1Sq_tf = tf$pow(z1_tf, 2); z2Sq_tf = tf$pow(z2_tf, 2)
  # az1z2_tf = a.pairs_tf*z1_tf*z2_tf
  log1_tf = tf$math$log(z1_tf) - tf$math$log(z2_tf); log2_tf = -log1_tf
  u1_tf = a.pairs_tf/2-log1_tf/a.pairs_tf; u2_tf = a.pairs_tf/2-log2_tf/a.pairs_tf
  
  ################# ///////////////////////////
  ########## 12.8
  trunc = tf$constant(12.83) # 12.74
  logic1 = u1_tf > trunc & u2_tf < -trunc
  logic2 = u1_tf < -trunc & u2_tf > trunc
  logic3 = u1_tf > trunc & u2_tf > trunc
  logic4 = tf$math$logical_not(logic1 | logic2 | logic3)
  type1 = tf$where(logic1) # [num of replication, index of obs pairs]
  type2 = tf$where(logic2)
  type3 = tf$where(logic3)
  type4 = tf$where(logic4)
  
  z1_tf1 = tf$gather_nd(z1_tf, type1)
  iz1_tf1 = 1/z1_tf1
  Cost1 = -tf$reduce_sum(2*tf$math$log(iz1_tf1) - iz1_tf1)
  
  z2_tf2 = tf$gather_nd(z2_tf, type2)
  iz2_tf2 = 1/z2_tf2
  Cost2 = -tf$reduce_sum(2*tf$math$log(iz2_tf2) - iz2_tf2)
  
  z1_tf3 = tf$gather_nd(z1_tf, type3); z2_tf3 = tf$gather_nd(z2_tf, type3)
  iz1_tf3 = 1/z1_tf3; iz2_tf3 = 1/z2_tf3
  Cost3 = -tf$reduce_sum(2*tf$math$log(iz1_tf3*iz2_tf3) - iz1_tf3 - iz2_tf3)
  
  # 4
  u1_tf4 = tf$gather_nd(u1_tf, type4); u2_tf4 = tf$gather_nd(u2_tf, type4)
  z1_tf4 = tf$gather_nd(z1_tf, type4); z2_tf4 = tf$gather_nd(z2_tf, type4)
  a.pairs_tf4 = tf$gather_nd(a.pairs.rep_tf, type4)
  z1Sq_tf4 = tf$pow(z1_tf4, 2); z2Sq_tf4 = tf$pow(z2_tf4, 2)
  az1z2_tf4 = a.pairs_tf4*z1_tf4*z2_tf4
  
  norm_tf = tfp$distributions$Normal(loc = 0., scale =1.)
  ##################
  Phi1_tf = norm_tf$cdf(u1_tf4); Phi2_tf = norm_tf$cdf(u2_tf4)
  phi1_tf = norm_tf$prob(u1_tf4); phi2_tf = norm_tf$prob(u2_tf4)
  V_tf = Phi1_tf/z1_tf4 + Phi2_tf/z2_tf4
  V1_tf = -Phi1_tf/z1Sq_tf4 - phi1_tf/(a.pairs_tf4*z1Sq_tf4) + phi2_tf/az1z2_tf4
  V2_tf = -Phi2_tf/z2Sq_tf4 - phi2_tf/(a.pairs_tf4*z2Sq_tf4) + phi1_tf/az1z2_tf4
  V12_tf = -(z2_tf4*u2_tf4*phi1_tf + z1_tf4*u1_tf4*phi2_tf)/tf$pow(az1z2_tf4,2)
  
  Cost4 = -tf$reduce_sum(tf$math$log(V1_tf * V2_tf - V12_tf) - V_tf)
  Cost = Cost1 + Cost2 + Cost3 + Cost4
  
  list(Cost = Cost)
}



# does not work
ECMSE = function(logphi_tf, logitkappa_tf, transeta_tf, a_tf, scalings, 
                 s_tf, ndata, method, 
                 family = "nonsta", 
                 weight_type = c("constant", "distance", "dependence"),
                 extdep.emp_tf = NULL, sel.pairs_tf = NULL) {
  
  s_in = s_tf
  if (family == "nonsta") {
    nlayers = length(layers)
    eta_tf <- swarped_tf <- list()
    swarped_tf[[1]] <- s_tf
    if(nlayers > 1) for(i in 1:nlayers) {
      # need to adapt this for LFT layer
      if (layers[[i]]$name == "LFT") {
        a_inum_tf = layers[[i]]$inum(a_tf)
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], a_inum_tf)
      } else { 
        eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]]) 
      }
      scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
      swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max, dtype = dtype)
    }
    
    s_in = swarped_tf[[nlayers+1]]
  }
  
  phi_tf = tf$exp(logphi_tf)
  kappa_tf = 2*tf$sigmoid(logitkappa_tf)
  
  sel.pairs_t_tf = tf$reshape(tf$transpose(sel.pairs_tf), c(2L, nrow(sel.pairs_tf), 1L))
  k1_tf = sel.pairs_t_tf[[0]]; k2_tf = sel.pairs_t_tf[[1]]
  
  s1.pairs_tf = tf$gather_nd(s_in, indices = k1_tf)
  s2.pairs_tf = tf$gather_nd(s_in, indices = k2_tf)
  diff.pairs_tf = tf$subtract(s1.pairs_tf, s2.pairs_tf)
  D.pairs_tf = tf$norm(diff.pairs_tf, ord='euclidean', axis = 1L)
  
  vario.pairs_tf = 2*tf$pow(D.pairs_tf/phi_tf, kappa_tf)
  
  norm_tf = tfp$distributions$Normal(loc = 0., scale =1.)
  ec_tf = 2*norm_tf$cdf( tf$cast(tf$sqrt(vario.pairs_tf)/2, tf$float32) )
  ec_tf = tf$cast(ec_tf, dtype)
  
  if (weight_type == "constant") {
    weights_tf = 1
  } else if (weight_type == "distance") {
    weights_tf = 1/D.pairs_tf
  } else if (weight_type == "dependence") {
    weights_tf = 1/extdep.emp_tf # stronger dependence indicates larger weights
  }
  
  Cost = tf$reduce_sum(weights_tf * tf$pow(tf$subtract(extdep.emp_tf, ec_tf), 2))
  
  list(Cost = Cost)
}

################################################################################

GradScore = function(logphi_tf, logitkappa_tf, transeta_tf, a_tf, scalings, 
                     s_tf, x_tf, u_tf, loc.pairs_t_tf, ndata, method,
                     risk, weight_fun, dWeight_fun, family = "nonsta") {
  
  nloc = nrow(x_tf)
  nexc = ncol(x_tf)
  z_tf = x_tf/u_tf
  
  s_in = s_tf
  if (family == "nonsta") {
    nlayers = length(layers)
    eta_tf <- swarped_tf <- list()
    swarped_tf[[1]] <- s_tf
    if(nlayers > 1) for(i in 1:nlayers) {
      # need to adapt this for LFT layer
      if (layers[[i]]$name == "LFT") {
        a_inum_tf = layers[[i]]$inum(a_tf)
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], a_inum_tf)
      } else { 
        eta_tf[[i]] <- layers[[i]]$trans(transeta_tf[[i]]) # ensure positivity for some variables
        swarped_tf[[i + 1]] <- layers[[i]]$f(swarped_tf[[i]], eta_tf[[i]]) 
      }
      scalings[[i + 1]] <- scale_lims_tf(swarped_tf[[i + 1]])
      swarped_tf[[i + 1]] <- scale_0_5_tf(swarped_tf[[i + 1]], scalings[[i + 1]]$min, scalings[[i + 1]]$max, dtype = dtype)
    }
    
    s_in = swarped_tf[[nlayers+1]]
  }
  ################################
  
  phi_tf = tf$exp(logphi_tf)
  kappa_tf = 2*tf$sigmoid(logitkappa_tf)
  
  # ----------------------------------------------------------------------------
  k1_tf = loc.pairs_t_tf[[0]]; k2_tf = loc.pairs_t_tf[[1]]
  
  s1.pairs_tf = tf$gather_nd(s_in, indices = k1_tf)
  s2.pairs_tf = tf$gather_nd(s_in, indices = k2_tf)
  diff.pairs_tf = tf$subtract(s1.pairs_tf, s2.pairs_tf)
  D.pairs_tf = tf$sqrt(tf$square(diff.pairs_tf[,1]) + tf$square(diff.pairs_tf[,2]))
  
  gamma.pairs_tf = tf$pow(D.pairs_tf/phi_tf, kappa_tf) #0.5*
  
  ones = tf$ones(c(nloc,nloc),dtype=tf$float32) #size of the output matrix
  mask_g = tf$linalg$band_part(ones, 0L, -1L) - tf$linalg$band_part(ones, 0L, 0L) # Mask of upper triangle above diagonal
  non_zero = tf$not_equal(mask_g, 0) #Conversion of mask to Boolean matrix
  gamma.uptri_tf = tf$sparse$to_dense(tf$sparse$SparseTensor(tf$where(non_zero), 
                                                             gamma.pairs_tf, 
                                                             dense_shape=c(nloc, nloc)))
  gamma_tf = gamma.uptri_tf + tf$transpose(gamma.uptri_tf)
  
  D0_tf = tf$sqrt(tf$square(s_in[,1]) + tf$square(s_in[,2]))
  gamma0_tf = tf$pow(D0_tf[D0_tf > 0]/phi_tf, kappa_tf) #0.5*
  indices0 = tf$where(D0_tf > 0)
  indices1 = tf$concat(c(tf$zeros(tf$shape(indices0), dtype=tf$int64), indices0), axis = 1L)
  gammaOrigin_tf = tf$sparse$to_dense(tf$sparse$SparseTensor(indices1, 
                                                             gamma0_tf, 
                                                             dense_shape=c(1L,nloc)))
  gammaOrigin_tf = tf$reshape(gammaOrigin_tf, c(nloc))
  
  #####
  S_tf = tf$tile(tf$expand_dims(gammaOrigin_tf, axis=1L), c(1L, nloc)) +
    tf$transpose(tf$tile(tf$expand_dims(gammaOrigin_tf, axis=1L), c(1L, nloc))) - gamma_tf
  
  Sobs_tf <- tf$multiply(1e-10, tf$eye(ndata, dtype = "float64"))
  SZ_tf <- tf$add(S_tf, Sobs_tf) # + jit
  U_tf <- tf$cholesky_upper(SZ_tf)
  Q_tf <- chol2inv_tf(U_tf)
  
  diagS_tf = tf$reshape(tf$linalg$diag_part(S_tf), c(nloc, 1L))
  qrsum_tf = tf$reduce_sum(Q_tf, 1L)
  qsum_tf = tf$reduce_sum(qrsum_tf)
  qqt_tf = tf$multiply(qrsum_tf, tf$expand_dims(qrsum_tf, axis=1L))
  
  A_tf = Q_tf - qqt_tf/qsum_tf
  B_tf = 2*tf$reshape(qrsum_tf/qsum_tf, c(nloc, 1L)) + 2 + 
    tf$linalg$matmul(Q_tf, diagS_tf) - 
    tf$linalg$matmul(qqt_tf, diagS_tf)/qsum_tf
  
  A_tf = 0.5*(A_tf + tf$transpose(A_tf))
  gradient_tf = -tf$linalg$matmul(A_tf, tf$math$log(z_tf))/z_tf - 0.5/z_tf*B_tf
  diagHessian_tf = -tf$reshape(tf$linalg$diag_part(A_tf), c(nloc, 1L))/z_tf^2 +
    tf$linalg$matmul(A_tf, tf$math$log(z_tf))/z_tf^2 +
    0.5/z_tf^2 * B_tf
  
  weight_tf = weight_fun(z_tf)
  dWeight_tf = dWeight_fun(z_tf)
  
  
  Cost = tf$reduce_sum(2*weight_tf*dWeight_tf*gradient_tf + 
                         weight_tf^2*(diagHessian_tf + 0.5*gradient_tf^2)) / nexc
  
  list(Cost = Cost)
}


WEIGHTS = function(risk_type, p0 = NULL, weight_type = 1) {
  if (risk_type == "sum") {
    if (weight_type == 1) {
      weight_fun = function(z_tf) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L); #zdrxuinv_tf = z_tf
        weight_tf = z_tf*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L); zdrxuinv_tf = z_tf
        dWeight_tf = 1 - tf$math$exp(1 - rxuinv_tf) * (1 - zdrxuinv_tf) }
    } else if (weight_type == 2) { # log
      weight_fun = function(z_tf) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L); #zdrxuinv_tf = z_tf
        part1 = tf$math$log(z_tf + 1)
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L)
        drxuinv_tf = 1
        part1 = tf$math$log(z_tf + 1)
        dpart1 = 1/(z_tf + 1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 3) { #sigmoid
      weight_fun = function(z_tf) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L); #zdrxuinv_tf = z_tf
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L)
        drxuinv_tf = 1
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        dpart1 = 0.1*part1*(1-part1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 4) { # 
      weight_fun = function(z_tf) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L); #zdrxuinv_tf = z_tf
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = tf$reduce_sum(z_tf, 0L)
        drxuinv_tf = 1
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        dpart1 = tf$where(z_tf > 10,
                          (3/10)*tf$math$exp(-3*(z_tf - 10)/10),
                          tf$zeros_like(z_tf))
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    }
    
  } else if (risk_type == "site") {
    if (weight_type == 1) {
      weight_fun = function(z_tf) {
        rxuinv_tf = z_tf[[1]];
        weight_tf = z_tf*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = z_tf[[1]];
        z_tf_row1 = tf$reshape(z_tf[[1]], c(1L, length(z_tf[[1]])))
        rest_zeros = tf$zeros(tf$shape(z_tf) - tf$constant(c(1L,0L)), dtype=dtype)
        zdrxuinv_tf = tf$concat(c(z_tf_row1, rest_zeros), axis = 0L)
        dWeight_tf = 1 - tf$math$exp(1 - rxuinv_tf) * (1 - zdrxuinv_tf) }
    } else if (weight_type == 2) {
      weight_fun = function(z_tf) {
        rxuinv_tf = z_tf[[1]];
        part1 = tf$math$log(z_tf + 1)
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = z_tf[[1]];
        z_tf_row1 = tf$reshape(tf$math$log(z_tf[[1]] + 1), c(1L, length(z_tf[[1]])))
        rest_zeros = tf$zeros(tf$shape(z_tf) - tf$constant(c(1L,0L)), dtype=dtype)
        part1drxuinv_tf = tf$concat(c(z_tf_row1, rest_zeros), axis = 0L)
        dpart1 = 1/(z_tf + 1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1drxuinv_tf*tf$math$exp(1 - rxuinv_tf) }
    } else if (weight_type == 3) {
      weight_fun = function(z_tf) {
        rxuinv_tf = z_tf[[1]];
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = z_tf[[1]];
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        z_tf_row1 = tf$reshape(part1[[1]], c(1L, length(z_tf[[1]])))
        rest_zeros = tf$zeros(tf$shape(z_tf) - tf$constant(c(1L,0L)), dtype=dtype)
        part1drxuinv_tf = tf$concat(c(z_tf_row1, rest_zeros), axis = 0L)
        dpart1 = 0.1*part1*(1-part1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1drxuinv_tf*tf$math$exp(1 - rxuinv_tf) }
    }  else if (weight_type == 4) {
      weight_fun = function(z_tf) {
        rxuinv_tf = z_tf[[1]];
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = z_tf[[1]];
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        z_tf_row1 = tf$reshape(part1[[1]], c(1L, length(z_tf[[1]])))
        rest_zeros = tf$zeros(tf$shape(z_tf) - tf$constant(c(1L,0L)), dtype=dtype)
        part1drxuinv_tf = tf$concat(c(z_tf_row1, rest_zeros), axis = 0L)
        dpart1 = tf$where(z_tf > 10,
                          (3/10)*tf$math$exp(-3*(z_tf - 10)/10),
                          tf$zeros_like(z_tf))
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1drxuinv_tf*tf$math$exp(1 - rxuinv_tf) }
    }
    
  } else if (risk_type == "max") {
    if (weight_type == 1) { # linear
      weight_fun = function(z_tf) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20)
        weight_tf = z_tf*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20)
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20-1) * tf$pow(z_tf, 19)
        zdrxuinv_tf = z_tf*drxuinv_tf
        dWeight_tf = 1 - tf$math$exp(1 - rxuinv_tf) * (1 - zdrxuinv_tf) }
    } else if (weight_type == 2) { # log
      weight_fun = function(z_tf) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        part1 = tf$math$log(z_tf + 1)
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20-1) * tf$pow(z_tf, 19)
        part1 = tf$math$log(z_tf + 1)
        dpart1 = 1/(z_tf + 1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 3) { # mine
      weight_fun = function(z_tf) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20-1) * tf$pow(z_tf, 19)
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        dpart1 = 0.1*part1*(1-part1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 4) { # weight2
      weight_fun = function(z_tf) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20);
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, 20), 0L), 1/20-1) * tf$pow(z_tf, 19)
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        dpart1 = tf$where(z_tf > 10,
                          (3/10)*tf$math$exp(-3*(z_tf - 10)/10),
                          tf$zeros_like(z_tf))
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    }
    
  } else if (risk_type == "sum2") {
    if (weight_type == 1) {
      weight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        weight_tf = z_tf*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p-1) * tf$pow(z_tf, p-1)
        zdrxuinv_tf = z_tf*drxuinv_tf
        dWeight_tf = 1 - tf$math$exp(1 - rxuinv_tf) * (1 - zdrxuinv_tf) }
    } else if (weight_type == 2) {
      weight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        part1 = tf$math$log(z_tf + 1)
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p-1) * tf$pow(z_tf, p-1)
        part1 = tf$math$log(z_tf + 1)
        dpart1 = 1/(z_tf + 1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 3) {
      weight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p-1) * tf$pow(z_tf, p-1)
        part1 = tf$sigmoid(0.1*(z_tf - 20))
        dpart1 = 0.1*part1*(1-part1)
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    } else if (weight_type == 4) {
      weight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        weight_tf = part1*(1 - tf$math$exp(1 - rxuinv_tf)) }
      dWeight_fun = function(z_tf, p = p0) {
        rxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p)
        drxuinv_tf = tf$pow(tf$reduce_sum(tf$pow(z_tf, p), 0L), 1/p-1) * tf$pow(z_tf, p-1)
        part1 = tf$where(z_tf > 10,
                         1-tf$math$exp(-3*(z_tf - 10)/10),
                         tf$zeros_like(z_tf))
        dpart1 = tf$where(z_tf > 10,
                          (3/10)*tf$math$exp(-3*(z_tf - 10)/10),
                          tf$zeros_like(z_tf))
        dWeight_tf = dpart1*(1 - tf$math$exp(1 - rxuinv_tf)) +
          part1*tf$math$exp(1 - rxuinv_tf)*drxuinv_tf }
    }
    
  }
  
  list(weight_fun = weight_fun, dWeight_fun = dWeight_fun)
}

