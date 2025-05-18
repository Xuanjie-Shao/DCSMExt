emp_extdep_est = function(data, coord, model, 
                          risk=NULL, q=NULL, q1=NULL, 
                          exceed_id = NULL) {
  if (model == "MSP-BR") {
    library(SpatialExtremes)
    fmad <- fmadogram(data = t(data), coord = coord)
    
    distances = fmad[,1]
    extcoeffs <- pmin(fmad[,3], 2)
    ec.emp = cbind(extcoeffs, distances)
    print("-----")
    return(ec.emp)
  } else if (model == "r-Pareto") {
    if (is.null(exceed_id)) {
      func_risk = apply(data, 2, risk)
      threshold <- quantile(func_risk, q)
      exceed_id = which(func_risk > threshold)
    }
    
    exceed <- as.matrix(data[, exceed_id])
    
    D = rdist(coord)
    
    u1 = quantile(as.numeric(exceed), q1)

    cep.pairs = do.call("cbind", sapply(1:(nrow(exceed)-1), function(i) {
      # print(i/nrow(exceed))
      sapply((i+1):nrow(exceed), function(j) {
        cep = sum(exceed[i,]>u1 & exceed[j,]>u1) /
          (0.5*sum(exceed[i,]>u1) + 0.5*sum(exceed[j,]>u1))
        cep = ifelse(is.na(cep), 0, cep)
        c(cep, D[i,j])
      }) }))
    str(cep.pairs)
    

    print("-----")
    cep.pairs = t(cep.pairs)
    return(cep.pairs)
  }
}

