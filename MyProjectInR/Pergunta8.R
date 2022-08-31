m_ic <- function(seed, m, n, lambda, gama){
  set.seed(seed)
  return(mean(replicate(m, (2*(qnorm((1 + gama)/2)/sqrt(n)))/(mean(rexp(n, lambda))))))
}

m_ic(seed = 781, m = 1000, n = 1123, lambda = 3.45, gama = 0.9)




