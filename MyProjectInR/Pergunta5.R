set.seed(1247)

RGenerate <- c(rexp(637, 0.14))

EmpDist <- ecdf(RGenerate)

EmpDist(8)

pexp(8, 0.14)

EmpDist(8) - pexp(8, 0.14)

