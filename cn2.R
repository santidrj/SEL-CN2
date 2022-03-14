library(data.table)
library(dplyr)
library(rlang)

data <- read.csv(file.path('datasets', 'breast-cancer.csv'), header = T)

# The algorithm assumes that the class is the last column of the dataset.

data <- data %>%
  rename(y = last_col())


most.common <- function(x){
  ## Returns the most common factor in a vector.
  
  dd <- unique(x)
  toString(dd[which.max(tabulate(match(x,dd)))])
}

find.best.complex <- function(E) {
  # paste(colnames(E), collapse = " && ")
  "irradiant == 'no' || irradiant == 'yes'"
}

CN2 <- function(E) {
  aux.copy <- copy(E)
  rule.list <- c()
  best.cpx <- ""
  while (!is.null(best.cpx) && dim(aux.copy)[1] != 0) {
    best.cpx <- find.best.complex(E)
    if (!is.null(best.cpx)) {
      selected <- data %>%
        filter(!! parse_expr(best.cpx))
      aux.copy <- anti_join(aux.copy, selected)
      C <- most.common(selected[, ncol(selected)])
      rule.list <- append(rule.list, paste("If", best.cpx, "then the class is", C, sep = " "))
    }
  }
  rule.list
}

rules <- CN2(data)

