rexceed = function(data, risk_fun) {
  func_risk = apply(data, 2, risk_fun)
  threshold <- quantile(as.numeric(func_risk), q)
  id_exc = which(func_risk > threshold)
  rep_exc <- as.matrix(data[,id_exc])
  list(id_exc = id_exc, 
       id_order = order(func_risk,decreasing = T),
       rep_exc_sort = sort(rep_exc,decreasing = T),
       rep_exc = rep_exc,
       threshold = threshold)
}


EC_fun = function(theta, d){
  vario.pair = 2*(d/theta[1])^theta[2]
  2*pnorm(sqrt(vario.pair)/2)
}


pairsamp = function(sites, nitv, frac, type = "stratify") {
  # site.pairs (matrix with 2 columns): all observation pairs
  # D.pairs (matrix DxD): distance matrix for all sites 
  # nitv (number): number of stratification
  
  site.pairs = t(do.call("cbind", sapply(1:(nrow(sites)-1), function(k1){
    sapply((k1+1):nrow(sites), function(k2){ c(k1,k2) } ) } )))
  
  if (type == "stratify") {
    D.pairs = rdist(sites)
    
    # Distance for observation pairs
    dist.pairs = sapply(1:nrow(site.pairs), function(r){
      pair = site.pairs[r,]; D.pairs[pair[1], pair[2]] })
    divide = seq(min(dist.pairs),max(dist.pairs),length.out = nitv + 1)
    divide[length(divide)] = divide[length(divide)] + 0.1
    divide.idx = lapply(1:nitv, function(i) {
      which(divide[i] <= dist.pairs & dist.pairs < divide[i+1])
    })
    sample.idx = c()
    for (i in 1:nitv) {
      sample.idx = c(sample.idx, sample(divide.idx[[i]], max(1,round(length(divide.idx[[i]])*frac))))
    }
  } else if (type == "simple") {
    sample.idx = sample(1:nrow(site.pairs), max(1,round(nrow(site.pairs)*frac)))
  }
  
  list(sample.idx = sample.idx,
       sample.pairs = site.pairs[sample.idx,]-1)
}


find_neighbors <- function(mat, row, col) {
  # Get the number of rows and columns in the matrix
  nrow_mat <- nrow(mat)
  ncol_mat <- ncol(mat)
  
  # Define the potential neighbor offsets
  offsets <- expand.grid(
    row_offset = c(-1, 0, 1),
    col_offset = c(-1, 0, 1)
  )
  
  # Remove the (0, 0) offset (center element itself)
  offsets <- offsets[!(offsets$row_offset == 0 & offsets$col_offset == 0), ]
  
  # Calculate the positions of neighbors
  neighbor_positions <- data.frame(
    row = row + offsets$row_offset,
    col = col + offsets$col_offset
  )
  
  # Filter out neighbors that are out of bounds
  valid_neighbors <- subset(
    neighbor_positions,
    row > 0 & row <= nrow_mat & col > 0 & col <= ncol_mat
  )
  
  # Get the values of the neighbors
  neighbor_values <- apply(valid_neighbors, 1, function(pos) {
    mat[pos[1], pos[2]]
  })
  
  return(as.numeric(neighbor_values))
}

find_neighbors_no_diagonal <- function(mat, row, col) {
  # Get the number of rows and columns in the matrix
  nrow_mat <- nrow(mat)
  ncol_mat <- ncol(mat)
  
  # Define the offsets for non-diagonal neighbors
  offsets <- data.frame(
    row_offset = c(-1, 1, 0, 0),  # Up, Down, Left, Right
    col_offset = c(0, 0, -1, 1)
  )
  
  # Calculate the positions of neighbors
  neighbor_positions <- data.frame(
    row = row + offsets$row_offset,
    col = col + offsets$col_offset
  )
  
  # Filter out neighbors that are out of bounds
  valid_neighbors <- subset(
    neighbor_positions,
    row > 0 & row <= nrow_mat & col > 0 & col <= ncol_mat
  )
  
  # Get the values of the neighbors
  neighbor_values <- apply(valid_neighbors, 1, function(pos) {
    mat[pos[1], pos[2]]
  })
  
  return(neighbor_values)
}
