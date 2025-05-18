tf$cholesky_lower <- tf$linalg$cholesky
tf$cholesky_upper <- function(x) tf$linalg$matrix_transpose(tf$linalg$cholesky(x))
tf$matrix_inverse <- tf$linalg$inv

AFF_1D <- function(a = c(0, 1), dtype = "float32") {
  
  if (!is.numeric(a)) stop("The parameter a needs to be numeric")
  if (!(length(a) == 2)) stop("The parameter a needs to be of length 2")
  
  a0 <- a[1]
  a1 <- a[2]
  
  a0_tf <- tf$Variable(0, name = "a0", dtype = dtype)
  a1_tf <- tf$Variable(1, name = "a1", dtype = dtype)
  
  trans <- function(transeta) {
    transeta
  }
  
  f = function(s_tf, eta_tf) {
    sout_tf <- a0_tf + a1_tf * s_tf
  }
  
  fR = function(s, eta) {
    matrix(a0 + a1 * s[, 1])
  }
  
  list(list(f = f,
            fR = fR,
            trans = trans,
            r = 1L,
            name = "AFF_1D",
            fix_weights = TRUE,
            pars = list(a0_tf, a1_tf)))
  
}


AFF_2D <- function(a = c(0, 1, 0, 0, 0, 1), dtype = "float32") {
  
  if (!is.numeric(a)) stop("The parameter a needs to be numeric")
  if (!(length(a) == 6)) stop("The parameter a needs to be of length 6")
  
  a0 <- a[1]
  a1 <- a[2]
  a2 <- a[3]
  b0 <- a[4]
  b1 <- a[5]
  b2 <- a[6]
  
  a0_tf <- tf$Variable(0, name = "a0", dtype = dtype)
  a1_tf <- tf$Variable(1, name = "a1", dtype = dtype)
  a2_tf <- tf$Variable(0, name = "a2", dtype = dtype)
  b0_tf <- tf$Variable(0, name = "b0", dtype = dtype)
  b1_tf <- tf$Variable(0, name = "b1", dtype = dtype)
  b2_tf <- tf$Variable(1, name = "b2", dtype = dtype)
  
  trans <- function(transeta) {
    transeta
  }
  
  f = function(s_tf, eta_tf) {
    sout1_tf <- tf$reshape(a0_tf + a1_tf * s_tf[, 1] + a2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout2_tf <- tf$reshape(b0_tf + b1_tf * s_tf[, 1] + b2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout_tf <- tf$concat(list(sout1_tf, sout2_tf), axis = 1L)
  }
  
  fMC = function(s_tf, eta_tf) {
    sout1_tf <- tf$reshape(a0_tf + a1_tf * s_tf[, 1] + a2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout2_tf <- tf$reshape(b0_tf + b1_tf * s_tf[, 1] + b2_tf * s_tf[, 2], c(nrow(s_tf[, 1]), 1L))
    sout_tf <- tf$concat(list(sout1_tf, sout2_tf), axis = 1L)
  }
  
  fR = function(s, eta) {
    s1 <- a0 + a1 * s[, 1] + a2 * s[, 2]
    s2 <- b0 + b1 * s[, 1] + b2 * s[, 2]
    matrix(c(s1, s2), nrow = length(s1), byrow=F)
  }
  
  list(list(f = f,
            fMC = fMC,
            fR = fR,
            trans = trans,
            r = 1L,
            name = "AFF_2D",
            fix_weights = TRUE,
            pars = list(a0_tf, a1_tf, a2_tf,
                        b0_tf, b1_tf, b2_tf)))
  
}




AWU <- function(r = 50L, dim = 1L, grad = 200, lims = c(-0.5, 0.5), dtype = "float32") {
  
  ## Parameters appearing in sigmoid (grad, loc)
  theta <- matrix(c(grad, 0), nrow = r - 1, ncol = 2, byrow = TRUE)
  theta[, 2] <- seq(lims[1], lims[2], length.out = (r - 1) + 2)[-c(1, (r - 1) + 2)]
  
  theta_steep_unclipped_tf <- tf$constant(theta[, 1, drop = FALSE], name = "thetasteep", dtype = dtype)
  theta_steep_tf <- tf$clip_by_value(theta_steep_unclipped_tf, 0, 200)
  theta_locs_tf <- tf$constant(theta[, 2, drop = FALSE], name = "thetalocs", dtype = dtype)
  theta_tf <- tf$concat(list(theta_steep_tf, theta_locs_tf), 1L)
  
  f = function(s_tf, eta_tf) {
    PHI_tf <- tf$concat(list(s_tf[, dim, drop = FALSE],
                             sigmoid_tf(s_tf[, dim, drop = FALSE], theta_tf, dtype)), 1L)
    swarped <-  tf$matmul(PHI_tf, eta_tf)
    slist <- lapply(1:ncol(s_tf), function(i) s_tf[, i, drop = FALSE])
    slist[[dim]] <- swarped
    sout_tf <- tf$concat(slist, axis = 1L)
  }
  
  fR = function(s, eta) {
    PHI_list <- list(s[, dim, drop = FALSE],
                     sigmoid(s[, dim, drop = FALSE], theta))
    PHI <- do.call("cbind", PHI_list)
    swarped <-  PHI %*% eta
    slist <- lapply(1:ncol(s), function(i) s[, i, drop = FALSE])
    slist[[dim]] <- swarped
    sout <- do.call("cbind", slist)
  }
  
  fMC = function(s_tf, eta_tf) {
    PHI_tf <- list(s_tf[, , dim, drop = FALSE],
                   sigmoid_tf(s_tf[, , dim, drop = FALSE], theta_tf, dtype)) %>%
      tf$concat(2L)
    swarped <-  tf$matmul(PHI_tf, eta_tf)
    slist <- lapply(1:ncol(s_tf[1, , ]), function(i) s_tf[, , i, drop = FALSE])
    slist[[dim]] <- swarped
    sout_tf <- tf$concat(slist, axis = 2L)
  }
  
  list(list(f = f,
            fR = fR,
            fMC = fMC,
            r = r,
            trans = tf$exp,
            fix_weights = FALSE,
            name = "AWU"))
}





bisquare1D <- function(x, theta) {
  (abs(x - theta[1]) < theta[2]) *
    (1 - (x - theta[1])^2 / theta[2]^2)^2
}

bisquare1D_tf <- function(x, theta, dtype = "float32") {
  
  theta1 <- tf$transpose(theta[, 1, drop = FALSE])
  theta2 <- tf$transpose(theta[, 2, drop = FALSE])
  
  nonzerobit <- tf$cast((tf$abs(x - theta1) < theta2), dtype)*(1 - (x - theta1)^2 / theta2^2)^2
  tf$multiply(x, 0) %>% tf$add(nonzerobit)
}


bisquare2D <- function(x, theta) {
  PHI_list <- list()
  for(i in 1:nrow(theta)) {
    delta <- sqrt((x[, 1] - theta[i, 1])^2 + (x[, 2] - theta[i, 2])^2)
    nonzerobit <- (delta < theta[i, 3])*(1 - delta^2 / theta[i, 3]^2)^2
    PHI_list[[i]] <- nonzerobit
  }
  PHI <- do.call("cbind", PHI_list)
}

bisquare2D_tf <- function(x, theta, dtype = "float32") {
  
  theta11 <- tf$transpose(theta[, 1, drop = FALSE])
  theta12 <- tf$transpose(theta[, 2, drop = FALSE])
  theta2 <- tf$transpose(theta[, 3, drop = FALSE])
  
  ndims <- x$shape$ndims
  
  if(ndims == 2) {
    x11 <- x[, 1, drop = FALSE]
    x12 <- x[, 2, drop = FALSE]
  } else if(ndims == 3) {
    x11 <- x[, , 1, drop = FALSE]
    x12 <- x[, , 2, drop = FALSE]
  }
  
  delta <- tf$sqrt(tf$square(x11 - theta11) + tf$square(x12 - theta12))
  nonzerobit <- tf$cast((delta < theta2), dtype)*(1 - delta^2 / theta2^2)^2
  nonzerobit
}




bisquares1D <- function(r = 30, lims = c(-0.5, 0.5), dtype = "float32") {
  knots_tf <- tf$constant(matrix(seq(lims[1] - 1/r, lims[2] + 1/r, length.out = r)), dtype = dtype)
  
  ## Establish width of bisquare functions on D1
  bisquarewidths_tf <- (tf$multiply((knots_tf[2, 1] - knots_tf[1, 1]), 2)) %>%
    tf$multiply(tf$constant(matrix(rep(1L, r)), dtype = dtype))
  
  ## The parameters are just the centres and widths of the bisquare functions
  theta_tf <- tf$concat(list(knots_tf, bisquarewidths_tf), 1L)
  
  f <- function(s_tf, eta_tf = NULL) {
    ## Evaluate basis functons on warped locations
    PHI_tf <- bisquare1D_tf(s_tf, theta_tf, dtype = dtype)
    if(is.null(eta_tf)) {
      PHI_tf
    } else {
      tf$matmul(PHI_tf, eta_tf)
    }
  }
  
  
  
  list(list(f = f,
            r = r,
            knots_tf = knots_tf))
  
}


bisquares2D <- function(r = 30, lims = c(-0.5, 0.5), dtype = "float32") {
  r1 <- round(sqrt(r))
  r <- as.integer(r1^2)
  knots1D <- seq(lims[1] - 1/r1, lims[2] + 1/r1, length.out = r1)
  knots2D <- as.matrix(expand.grid(s1 = knots1D, s2 = knots1D))
  knots2D_tf <- tf$constant(knots2D, dtype = dtype)
  
  ## Establish width of bisquare functions on D1
  bisquarewidths <- 2*(knots1D[2] - knots1D[1])
  bisquarewidths_tf <- tf$constant(matrix(bisquarewidths, nrow = r), dtype = dtype)
  
  ## The parameters are just the centres and widths of the bisquare functions
  theta <- cbind(knots2D, bisquarewidths)
  theta_tf <- tf$concat(list(knots2D_tf, bisquarewidths_tf), 1L)
  
  f <- function(s_tf, eta_tf = NULL) {
    ## Evaluate basis functons on warped locations
    PHI_tf <- bisquare2D_tf(s_tf, theta_tf)
    if(is.null(eta_tf)) {
      PHI_tf
    } else {
      tf$matmul(PHI_tf, eta_tf)
    }
  }
  
  fR <- function(s, eta = NULL) {
    ## Evaluate basis functons on warped locations
    PHI <- bisquare2D(s, theta)
    if(is.null(eta)) PHI else PHI %*% eta
  }
  
  list(list(f = f,
            fR = fR,
            r = r,
            knots = knots2D,
            knots_tf = knots2D_tf))
}




## Covariance matrix of the weights at the top layer
cov_exp_tf <- function(x1, x2 = x1, sigma2f, alpha, dtype = "float32") {
  
  d <- ncol(x1)
  n1 <- nrow(x1)
  n2 <- nrow(x2)
  square_mat <- tf$cast(tf$math$equal(n1,n2), dtype)
  Dsquared <- tf$constant(matrix(0, n1, n2), 
                          name = 'D', 
                          dtype = dtype)
  
  for(i in 1:d) {
    x1i <- x1[, i, drop = FALSE]
    x2i <- x2[, i, drop = FALSE]
    sep <- x1i - tf$transpose(x2i)
    alphasep <- tf$multiply(alpha[1, i, drop = FALSE], sep)
    alphasep2 <- tf$square(alphasep)
    Dsquared <- tf$add(Dsquared, alphasep2)
  }
  
  Dsquared <- Dsquared + tf$multiply(square_mat, tf$multiply(1e-30, tf$eye(n1)))
  D <- tf$sqrt(Dsquared)
  K <- tf$multiply(sigma2f, tf$exp(-0.5 * D))
  
  return(K)
}

cov_sqexp_tf <- function(x1, x2 = x1, sigma2f, alpha) {
  
  d <- ncol(x1)
  n1 <- nrow(x1)
  n2 <- nrow(x2)
  D <- tf$constant(matrix(0, n1, n2), name='D', dtype = tf$float32)
  
  for(i in 1:d) {
    x1i <- x1[, i, drop = FALSE]
    x2i <- x2[, i, drop = FALSE]
    sep <- x1i - tf$transpose(x2i)
    sep2 <- tf$pow(sep, 2)
    alphasep2 <- tf$multiply(alpha[1, i, drop = FALSE], sep2)
    D <- tf$add(D, alphasep2)
  }
  D <- tf$multiply(-0.5, D)
  K <- tf$multiply(sigma2f, tf$exp(D))
  return(K + tf$diag(rep(0.01, nrow(x1))))
}



# eta_mean, LFTpars, vario
init_learn_rates <- function(sigma2y = 0.0005, covfun = 0.01, sigma2eta = 0.0001,
                             eta_mean = 0.1, eta_mean2 = 0.05,
                             eta_sd = 0.1, LFTpars = 0.01,
                             AFFpars = 0.01, rho = 0.1, vario = 0.1) {
  
  list(sigma2y = sigma2y,
       covfun = covfun,
       sigma2eta = sigma2eta,
       eta_mean = eta_mean,
       eta_mean2 = eta_mean2,
       eta_sd = eta_sd,
       LFTpars = LFTpars,
       AFFpars = AFFpars,
       rho = rho,
       vario = vario)
}




initvars <- function(variogram_logrange = log(0.3),
                     variogram_logitdf = .5,
                     # sigma2y = 0.1,
                     # l_top_layer = 0.5,
                     # sigma2eta_top_layer = 1,
                     # nu = 1.5,
                     transeta_mean_init = list(AWU = -3, #  doesn't matter
                                               RBF = -0.8068528,
                                               RBF1 = -0.8068528,
                                               RBF2 = -0.8068528,
                                               LFT = 1,
                                               AFF_1D = 1,
                                               AFF_2D = 1)) {
  
  list(variogram_logrange = variogram_logrange,
       variogram_logitdf = variogram_logitdf,
       transeta_mean_init = transeta_mean_init)
  # sigma2y = sigma2y,
  # sigma2eta_top_layer = sigma2eta_top_layer,
  # l_top_layer = l_top_layer,
  # nu = nu,
  # transeta_mean_prior = transeta_mean_prior,
  # transeta_sd_init =transeta_sd_init,
  # transeta_sd_prior = transeta_sd_prior)
}





LFT <- function(a = NULL, dtype = "float32") {
  
  if(is.null(a)) {
    a1 <- a4 <- 1 + 0i
    a2 <- a3 <- 0 + 0i
  } else {
    if(!is.complex(a) & !(length(a) == 4))
      stop("a needs to be a vector of 4 complex numbers")
    a1 <- a[1]
    a2 <- a[2]
    a3 <- a[3]
    a4 <- a[4]
  }
  
  a1Re_tf <- tf$Variable(1, name = "a1Re", dtype = dtype)
  a2Re_tf <- tf$Variable(0, name = "a2Re", dtype = dtype)
  a3Re_tf <- tf$Variable(0, name = "a3Re", dtype = dtype)
  a4Re_tf <- tf$Variable(1, name = "a4Re", dtype = dtype)
  
  a1Im_tf <- tf$Variable(0, name = "a1Im", dtype = dtype)
  a2Im_tf <- tf$Variable(0, name = "a2Im", dtype = dtype)
  a3Im_tf <- tf$Variable(0, name = "a3Im", dtype = dtype)
  a4Im_tf <- tf$Variable(0, name = "a4Im", dtype = dtype)
  
  inum <- function(pars){
    a1Re_tf = pars[[1]]; a2Re_tf = pars[[2]]; a3Re_tf = pars[[3]]; a4Re_tf = pars[[4]]
    a1Im_tf = pars[[5]]; a2Im_tf = pars[[6]]; a3Im_tf = pars[[7]]; a4Im_tf = pars[[8]]
    a1_tf <- tf$complex(real = a1Re_tf, imag = a1Im_tf)
    a2_tf <- tf$complex(real = a2Re_tf, imag = a2Im_tf)
    a3_tf <- tf$complex(real = a3Re_tf, imag = a3Im_tf)
    a4_tf <- tf$complex(real = a4Re_tf, imag = a4Im_tf)
    list(a1_tf = a1_tf, a2_tf = a2_tf, a3_tf = a3_tf, a4_tf = a4_tf)
  }
  
  trans <- function(transeta) {
    transeta
  }
  
  f = function(s_tf, a_tf) {
    # a1_tf = a_tf[1,1]; a2_tf = a_tf[2,1]
    # a3_tf = a_tf[3,1]; a4_tf = a_tf[4,1]
    a1_tf = a_tf[[1]]; a2_tf = a_tf[[2]]; a3_tf = a_tf[[3]]; a4_tf = a_tf[[4]]
    z <- tf$complex(real = s_tf[, 1], imag = s_tf[, 2])
    P1 <- tf$multiply(a1_tf, z) %>% tf$add(a2_tf)
    P2 <- tf$multiply(a3_tf, z) %>% tf$add(a4_tf)
    P <- tf$math$divide(P1, P2) %>% tf$expand_dims(1L)
    sout_tf <- tf$concat(list(tf$math$real(P), tf$math$imag(P)), axis = 1L)
  }
  
  fMC = function(s_tf, a_tf) {
    # a1_tf = a_tf[1,1]; a2_tf = a_tf[2,1]
    # a3_tf = a_tf[3,1]; a4_tf = a_tf[4,1]
    a1_tf = a_tf[[1]]; a2_tf = a_tf[[2]]; a3_tf = a_tf[[3]]; a4_tf = a_tf[[4]]
    z <- tf$complex(real = s_tf[, , 1], imag = s_tf[, , 2])
    P1 <- tf$multiply(a1_tf, z) %>% tf$add(a2_tf)
    P2 <- tf$multiply(a3_tf, z) %>% tf$add(a4_tf)
    P <- tf$math$divide(P1, P2) %>% tf$expand_dims(2L)
    sout_tf <- tf$concat(list(tf$math$real(P), tf$math$imag(P)), axis = 2L)
  }
  
  fR = function(s, a) {
    # a1 = a[1,1]; a2 = a[2,1]; a3 = a[3,1]; a4 = a[4,1]
    a1 = a[[1]]; a2 = a[[2]]; a3 = a[[3]]; a4 = a[[4]]
    z <- s[, 1] + s[, 2]*1i
    fz <- (a1*z + a2) / (a3*z + a4)
    cbind(Re(fz), Im(fz))
  }
  
  list(list(f = f,
            fMC = fMC,
            fR = fR,
            trans = trans,
            inum = inum,
            r = 1L,
            name = "LFT",
            fix_weights = TRUE,
            pars = list(a1Re_tf, a2Re_tf, a3Re_tf, a4Re_tf,
                        a1Im_tf, a2Im_tf, a3Im_tf, a4Im_tf)))
  # a1Re_tf = a1Re_tf, a2Re_tf = a2Re_tf, a3Re_tf = a3Re_tf, a4Re_tf = a4Re_tf,
  # a1Im_tf = a1Im_tf, a2Im_tf = a2Im_tf, a3Im_tf = a3Im_tf, a4Im_tf = a4Im_tf
}


RBF <- function(x, theta) {
  
  theta1 <- matrix(rep(1, nrow(x))) %*% theta[1:2]
  theta11 <- theta[1]
  theta12 <- theta[2]
  theta2 <- theta[3]
  
  sep1sq <- (x[, 1, drop = FALSE] - theta11)^2
  sep2sq <- (x[, 2, drop = FALSE] - theta12)^2
  sepsq <- sep1sq + sep2sq
  
  (exp(-theta2 * sepsq) %*% matrix(1, 1, 2))*(x - theta1) + theta1
}


RBF_tf <- function(x, theta) {
  
  theta1 <- theta[, 1:2]
  theta11 <- theta[, 1, drop = FALSE]
  theta12 <- theta[, 2, drop = FALSE]
  theta2 <- theta[, 3, drop = FALSE]
  
  if(length(dim(x)) == 2) {
    sep1sq <- tf$square(x[, 1, drop = FALSE] - theta11)
    sep2sq <- tf$square(x[, 2, drop = FALSE] - theta12)
  } else if(length(dim(x)) == 3) {
    sep1sq <- tf$square(x[, , 1, drop = FALSE] - theta11)
    sep2sq <- tf$square(x[, , 2, drop = FALSE] - theta12)
  }
  sepsq <- sep1sq + sep2sq
  
  tf$exp(-theta2 * sepsq) %>%
    tf$multiply(x - theta1) %>%
    tf$add(theta1)
}




RBF_block <- function(res = 1L, lims = c(-0.5, 0.5), dtype = "float32") {
  
  ## Parameters appearing in sigmoid (grad, loc)
  r <- (3^res)^2
  cx1d <- seq(lims[1], lims[2], length.out = sqrt(r))
  cxgrid <- expand.grid(s1 = cx1d, s2 = cx1d) %>% as.matrix()
  a <- 2*(3^res - 1)^2
  theta <- cbind(cxgrid, a)
  theta_tf <- tf$constant(theta, dtype = dtype)
  
  RBF_list <- list()
  
  
  trans <- function(transeta) {
    tf$exp(-transeta) %>%
      tf$add(tf$constant(1, dtype = dtype)) %>%
      tf$math$reciprocal() %>%
      tf$multiply(tf$constant(1 + exp(3/2)/2, dtype = dtype)) %>%
      tf$add(tf$constant(-1, dtype = dtype))
  }
  
  for(count in 1:r) {
    ff <- function(count) {
      j <- count
      
      f = function(s_tf, eta_tf) {
        PHI_tf <- RBF_tf(s_tf, theta_tf[j, , drop = FALSE])
        swarped <-  tf$multiply(PHI_tf, eta_tf)
        sout_tf <- tf$add(swarped, s_tf)
      }
      
      fMC = function(s_tf, eta_tf) {
        PHI_tf <- RBF_tf(s_tf, theta_tf[j, , drop = FALSE])
        swarped <-  tf$multiply(PHI_tf, eta_tf)
        sout_tf <- tf$add(swarped, s_tf)
      }
      
      fR = function(s, eta) {
        PHI <- RBF(s, theta[j, , drop = FALSE])
        swarped <-  PHI*eta
        sout <- swarped + s
      }
      list(f = f, fMC = fMC, fR = fR)
      
    }
    RBF_list[[count]] <- list(f = ff(count)$f,
                              fMC = ff(count)$fMC,
                              fR = ff(count)$fR,
                              r = 1L,
                              trans = trans,
                              fix_weights = FALSE,
                              name = paste0("RBF", res))
  }
  RBF_list
}





sigmoid <- function(x, theta) {
  PHI <- list()
  for(i in 1:nrow(theta)) {
    PHI[[i]] <- 1 / (1 + exp(-theta[i, 1] * (x - theta[i, 2])))
  }
  do.call("cbind", PHI)
}

sigmoid_tf <- function(x, theta, dtype = "float32") {
  
  theta1 <- tf$transpose(theta[, 1, drop = FALSE])
  theta2 <- tf$transpose(theta[, 2, drop = FALSE])
  
  tf$subtract(x, theta2) %>%
    tf$multiply(tf$constant(-1L, dtype = dtype)) %>%
    tf$multiply(theta1) %>%
    tf$exp() %>%
    tf$add(tf$constant(1L, dtype = dtype)) %>%
    tf$math$reciprocal()
}




tent <- function(x, theta) {
  (abs(x - theta[1]) < theta[2]) *
    ((x < theta[1])*((x - theta[1] + theta[2])/theta[2]) -
       (x > theta[1])*((x - theta[1] - theta[2])/theta[2]))
}

tent_tf <- function(x, theta, dtype = "float32") {
  leftbit <- tf$cast((x <= theta[1] & x >= theta[1] - theta[2]), dtype = dtype)*((x - theta[1] + theta[2])/theta[2])
  rightbit <- -tf$cast((x > theta[1] & x < theta[1] + theta[2]), dtype = dtype)*((x - theta[1] - theta[2])/theta[2])
  tf$multiply(x, 0) %>% tf$add(leftbit) %>% tf$add(rightbit)
}





logdet <- function (R) {
  diagR <- diag(R)
  return(2 * sum(log(diagR)))
}

tr <- function(A) {
  sum(diag(A))
}

safe_chol <- function(A) {
  A <- A + 10^(-6) * diag(nrow(A))
  chol(A)
}

atBa <- function(a, B) {
  t(a) %*% (B %*% a)
}

ABinvAt <- function(A, cholB) {
  tcrossprod(A %*% solve(cholB))
}

AtBA_p_C <- function(A, cholB, C) {
  crossprod(cholB %*% A) + C
}

entropy <- function(s) {
  d <- ncol(s)
  0.5 * sum(colSums(log(s)))
}

get_depvars <- function(f) {
  . <- NULL
  stopifnot(is(f, "formula"))
  if(!attr(terms(f), "response")) {
    depvars <- NULL
  } else {
    gr <- grepl("cbind", as.character(f))
    idx <- which(gr)
    if(length(idx) > 0) {
      terms(f)
      depvars <- ((attr(terms(f), "variables") %>%
                     as.character())[2] %>%
                    strsplit(","))[[1]] %>%
        gsub("cbind\\(|\\)|\\s", "", .)
    } else  {
      depvars <- all.vars(f)[[1]]
    }
  }
  depvars
}

get_depvars_multivar <- function(f) {
  . <- NULL
  stopifnot(is(f, "formula"))
  if(!attr(terms(f), "response")) {
    depvars <- NULL
  } else {
    gr <- grepl("cbind", as.character(f))
    idx <- which(gr)
    if(length(idx) > 0) {
      terms(f)
      depvars <- ((attr(terms(f), "variables") %>%
                     as.character())[2] %>%
                    strsplit(","))[[1]] %>%
        gsub("cbind\\(|\\)|\\s", "", .)
    } else  {
      depvars <- c(all.vars(f)[[1]], all.vars(f)[[2]])
    }
  }
  depvars
}

get_depvars_multivar2 <- function(f) {
  . <- NULL
  stopifnot(is(f, "formula"))
  if(!attr(terms(f), "response")) {
    depvars <- NULL
  } else {
    gr <- grepl("cbind", as.character(f))
    idx <- which(gr)
    if(length(idx) > 0) {
      terms(f)
      depvars <- ((attr(terms(f), "variables") %>%
                     as.character())[2] %>%
                    strsplit(","))[[1]] %>%
        gsub("cbind\\(|\\)|\\s", "", .)
    } else  {
      depvars <- c(all.vars(f)[[1]], all.vars(f)[[2]], all.vars(f)[[3]], all.vars(f)[[4]])
    }
  }
  depvars
}


get_depvars_multivar3 <- function(f, ndepvar) {
  . <- NULL
  stopifnot(is(f, "formula"))
  if(!attr(terms(f), "response")) {
    depvars <- NULL
  } else {
    gr <- grepl("cbind", as.character(f))
    idx <- which(gr)
    if(length(idx) > 0) {
      terms(f)
      depvars <- ((attr(terms(f), "variables") %>%
                     as.character())[2] %>%
                    strsplit(","))[[1]] %>%
        gsub("cbind\\(|\\)|\\s", "", .)
    } else  {
      depvars <- all.vars(f)[1:ndepvar]
    }
  }
  depvars
}



pinvsolve <- function(A, b, reltol = 1e-6) {
  # Compute the SVD of the input matrix A
  A_SVD = svd(A)
  s <- A_SVD$d
  u <- A_SVD$u
  v <- A_SVD$v
  
  # Invert s, clear entries lower than reltol*s[0].
  atol = max(s) * reltol
  s_mask = s[which(s > atol)]
  s_reciprocal <- 1/s_mask
  s_inv = diag(c(s_reciprocal, rep(0, length(s) - length(s_mask))))
  
  # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
  v %*% (s_inv %*% (t(u) %*% b))
}


list_to_listtf <- function(l, name, constant = TRUE, dtype = "float32") {
  stopifnot(is.list(l))
  stopifnot(is.character(name))
  stopifnot(is.logical(constant))
  
  if(constant) tffun <- tf$constant else tffun <- tf$Variable
  lapply(1:length(l), function(i)
    tffun(l[[i]], name = paste0(name, i), dtype = dtype))
}

proc_m.inducing <- function(m.inducing = 10L, nlayers = 1) {
  if(length(m.inducing) == 1)
    m.inducing <- rep(m.inducing, nlayers)
  m.inducing
}

scal_0_5 <- function(s) {
  mins <- min(s)
  maxs <- max(s)
  s <- (s - min(s)) / (maxs - mins) - 0.5
}

scal_0_5_mat <- function(s) {
  mins <- matrix(1, nrow(s), 1) %*% apply(s, 2, min)
  maxs <- matrix(1, nrow(s), 1) %*% apply(s, 2, max)
  s <- (s - mins) / (maxs - mins) - 0.5
}


KL <- function(mu1, S1, mu2, S2) {
  0.5*(sum(diag(solve(S2) %*% S1)) +
         t(mu2 - mu1) %*% solve(S2) %*% (mu2 - mu1) -
         nrow(mu1) +
         determinant(S2)$modulus -
         determinant(S1)$modulus)
}

## Plot warping in ggplot
polygons_from_points <- function(df, every = 3) {
  # df must have s1c and s2c that are integers
  #               s1 and s2
  #               h1 and h2
  
  s1c <- s2c <- NULL
  df <- df %>%  filter((s1c %% every == 0) & (s2c %% every == 0))
  
  cells <- list()
  count <- 0
  for(i in 1:nrow(df)) {
    this_centroid <- df[i,]
    d <- filter(df, (s1c - this_centroid$s1c) < (every + 1) & (s1c - this_centroid$s1c)  >= 0 &
                  (s2c - this_centroid$s2c) < (every + 1) & (s2c - this_centroid$s2c >= 0))
    
    if(nrow(d) == 4)  {
      count <- count + 1
      idx1 <- which(d$s1 == min(d$s1) & d$s2 == min(d$s2))
      idx2 <- which(d$s1 == max(d$s1) & d$s2 == min(d$s2))
      idx3 <- which(d$s1 == max(d$s1) & d$s2 == max(d$s2))
      idx4 <- which(d$s1 == min(d$s1) & d$s2 == max(d$s2))
      
      this_cell <- data.frame(x = d$h1[c(idx1, idx2, idx3, idx4)],
                              y = d$h2[c(idx1, idx2, idx3, idx4)],
                              s1c = d$s1c[c(idx1, idx2, idx3, idx4)],
                              s2c = d$s2c[c(idx1, idx2, idx3, idx4)],
                              id = count)
      cells[[count]] <- this_cell
    }
  }
  data.table::rbindlist(cells)
}

polygons_from_points1 <- function(df, every = 3) {
  # df must have s1c and s2c that are integers
  #               s1 and s2
  #               h1 and h2
  
  s1c <- s2c <- NULL
  df <- df %>%  filter((s1c %% every == 0) & (s2c %% every == 0))
  
  cells <- list()
  count <- 0
  for(i in 1:nrow(df)) {
    this_centroid <- df[i,]
    d <- filter(df, (s1c - this_centroid$s1c) < (every + 1) & (s1c - this_centroid$s1c)  >= 0 &
                  (s2c - this_centroid$s2c) < (every + 1) & (s2c - this_centroid$s2c >= 0))
    
    if(nrow(d) == 4)  {
      count <- count + 1
      minlon_group = sort(d$s1)[1:2]; maxlon_group = sort(d$s1)[3:4]
      minlat_group = sort(d$s2)[1:2]; maxlat_group = sort(d$s2)[3:4]
      idx1 <- which(d$s1 %in% minlon_group & d$s2 %in% minlat_group)
      idx2 <- which(d$s1 %in% maxlon_group & d$s2 %in% minlat_group)
      idx3 <- which(d$s1 %in% maxlon_group & d$s2 %in% maxlat_group)
      idx4 <- which(d$s1 %in% minlon_group & d$s2 %in% maxlat_group)
      
      this_cell <- data.frame(x = d$h1[c(idx1, idx2, idx3, idx4)],
                              y = d$h2[c(idx1, idx2, idx3, idx4)],
                              s1c = d$s1c[c(idx1, idx2, idx3, idx4)],
                              s2c = d$s2c[c(idx1, idx2, idx3, idx4)],
                              id = count)
      cells[[count]] <- this_cell
    }
  }
  data.table::rbindlist(cells)
}





logdet_tf <- function (R) {
  diagR <- tf$linalg$diag_part(R)
  ldet <- tf$math$log(diagR) %>%
    tf$reduce_sum(axis = -1L) %>%
    tf$multiply(2)
  return(ldet)
}

ndim_tf <- function(x) {
  tf$size(tf$shape(x))
}

tr_tf <- function(A) {
  tf$trace(A)
}

safe_chol_tf <- function(A, dtype = "float32") {
  I <- tf$constant(1e-6 * diag(nrow(A)), name = "Imat", dtype = dtype)
  tf$cholesky_upper(tf$add(A, I))
}

atBa_tf <- function(a, B) {
  a %>%
    tf$transpose() %>%
    tf$matmul(B) %>%
    tf$matmul(a)
}

ABinvAt_tf <- function(A, cholB) {
  AcholB <- tf$matmul(A, tf$matrix_inverse(cholB))
  tf$matmul(AcholB, tf$transpose(AcholB))
}

AtBA_p_C_tf <- function(A, cholB, C) {
  cholBA <- tf$matmul(cholB, A)
  tf$matmul(tf$linalg$transpose(cholBA), cholBA) %>%
    tf$add(C)
}


entropy_tf <- function(s) {
  d <- ncol(s)
  s %>% tf$math$log() %>% tf$reduce_sum() %>% tf$multiply(0.5)
}

chol2inv_tf <- function(R) {
  Rinv <- tf$matrix_inverse(R)
  tf$matmul(Rinv, tf$transpose(Rinv))
}

tile_on_dim1 <- function(A, n) {
  m1 <- nrow(A)
  m2 <- ncol(A)
  X <- tf$tile(A, c(n, 1L)) %>%
    tf$reshape(c(n, m1, m2))
  X
}

pinvsolve_tf <- function(A, b, reltol = 1e-6) {
  # Compute the SVD of the input matrix A
  A_SVD = tf$svd(A)
  s <- A_SVD[[1]]
  u <- A_SVD[[2]]
  v <- A_SVD[[3]]
  
  # Invert s, clear entries lower than reltol*s[0].
  atol = tf$multiply(tf$reduce_max(s), reltol)
  s_mask = tf$boolean_mask(s, tf$greater_equal(s, atol))
  s_reciprocal <- tf$math$reciprocal(s_mask)
  s_inv = tf$diag(tf$concat(list(s_reciprocal,
                                 tf$zeros(tf$size(s) - tf$size(s_mask))), 0L))
  
  # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
  tf$matmul(v, tf$matmul(s_inv, tf$matmul(u, b, transpose_a = TRUE)))
}

scale_lims_tf <- function(s_tf) {
  smin_tf <- tf$reduce_min(s_tf, axis = -2L, keepdims = TRUE)
  smax_tf <- tf$reduce_max(s_tf, axis = -2L, keepdims = TRUE)
  
  list(min = smin_tf,
       max = smax_tf)
}

scale_0_5_tf <- function(s_tf, smin_tf, smax_tf, dtype = "float32") {
  s_tf <- (s_tf - smin_tf) /(smax_tf - smin_tf) -
    tf$constant(0.5, dtype = dtype)
}

KL_tf <- function(mu1, S1, mu2, S2) {
  R1 <- tf$cholesky_upper(S1)
  R2 <- tf$cholesky_upper(S2)
  Q2 <- chol2inv_tf(R2)
  k <- tf$shape(mu1)[1] %>% tf$to_float()
  
  Part1 <- tf$matmul(Q2, S1) %>% tf$trace()
  Part2 <- (tf$transpose(mu2 - mu1) %>%
              tf$matmul(Q2) %>%
              tf$matmul(mu2 - mu1))[1,1]
  Part3 <- -k
  Part4 <- logdet_tf(R2) - logdet_tf(R1)
  
  0.5 * (Part1 + Part2 + Part3 + Part4) # tf$constant(0.5, dtype = "float32")
} 

#' @title Set TensorFlow seed
#' @description Set TensorFlow seed in deepspat package
#' @param seed the seed
#' @export
#' @examples
#' set_deepspat_seed(1L)
set_deepspat_seed <- function(seed = 1L) {
  # tf$set_random_seed(seed)
  tf$random$set_seed(seed)
  invisible()
}




tf_status <- new.env()
besselK_tf <- NULL

## Load TensorFlow and add the cholesky functions
.onLoad <- function(libname, pkgname) {
  
  tf <<- reticulate::import("tensorflow", delay_load = TRUE)
  tf$cholesky_lower <- tf$linalg$cholesky
  tf$cholesky_upper <- function(x) tf$linalg$transpose(tf$linalg$cholesky(x))
  
  # Load the bessel function
  
  # Set default tf_status that everything is installed correctly.
  assign("TF", TRUE, envir = tf_status)
  # Check TensorFlow is installed. Update tf_status accordingly.
  # checkTF()
  # If checkTF was not successful, return to avoid printing multiple messages
  if (!get("TF", envir = tf_status)) {
    return()
  }
  # Check TensorFlow Probability is installed, and load in. Update tf_status accordingly.
  
  bessel <- reticulate::import_from_path("besselK", system.file("python", package = "deepspat"))
  besselK_tf <<- bessel$besselK_tf
  
  
}


delta = function(h, tau) {
  if (tau == 0) {
    rep(1, length(h))
  } else {
    pmax(0, (1-(h/tau)^2)^2)
  }
}


EuD = function(s_in1, s_in2 = s_in1) {
  # s_in with shape c(D, 2)
  d <- ncol(s_in1)
  n1 <- nrow(s_in1)
  n2 <- nrow(s_in2)
  square_mat <- tf$cast(tf$math$equal(n1,n2), "float32")
  Dsquared <- tf$constant(matrix(0, n1, n2), 
                          name = 'D', 
                          dtype = tf$float32)
  
  for(i in 1:d) {
    x1i <- s_in1[, i, drop = FALSE]
    x2i <- s_in2[, i, drop = FALSE]
    sep <- x1i - tf$linalg$matrix_transpose(x2i)
    sep2 <- tf$square(sep)
    Dsquared <- tf$add(Dsquared, sep2)
  }
  
  ## Add on a small constant for numeric stability
  Dsquared <- Dsquared + 1e-15
  tf$sqrt(Dsquared)
}
