library(data.table)
library(dplyr)
library(rlang)

data <- read.csv(file.path('datasets', 'breast-cancer.csv'), header = T, stringsAsFactors = T)

# The algorithm assumes that the class is the last column of the dataset.

data <- data %>%
  rename(y = last_col())

most.common <- function(x){
  ## Returns the most common factor in a vector.
  
  dd <- unique(x)
  toString(dd[which.max(tabulate(match(x,dd)))])
}

get.selectors <- function(x) {
  selectors <- list()
  x.attr <- names(x)
  for (i in seq(length(x.attr))) {
    values <- x[[i]]
    if (is.factor(values)) {
      
      selectors[[length(selectors)+1]] <- levels(values)
    } else {
      selectors[[length(selectors)+1]] <- unique(values)
    }
    # for (value in values) {
    #   selectors <- append(selectors, paste(name,value, sep = " == "))
    # }
  }
  names(selectors) <- x.attr
  selectors
}

format.complex <- function(x) {
  complex.list <- list()
  for (name in names(x)) {
    for (value in x[name]) {
      complex.list <- append(complex.list, paste(name,value, sep = " == "))
    }
  }
  complex.list
}

find.best.complex <- function(E, selectors) {
  star <- list()
  best.cpx <- NULL
  combinations <- list()
  repeat  {
    if (length(star) == 0) {
      new.star <- selectors
      combinations <- names(selectors)
      complex.list <- format.complex(new.star)
    } else{
      new.combinations <- list()
      for (name in names(star)) {
        new.combinations <- append(new.combinations, as.list(outer(name, names(selectors)[names(selectors) != name], paste)))
        aux <- expand.grid(star[name], selectors[names(selectors) %in% name == FALSE])
      }
    }
    
    star <- new.star

    if (length(star) <= 0) {
      break
    }
  }
  paste(colnames(E), collapse = " && ")
  "irradiant == 'no' || irradiant == 'yes'"
}

CN2 <- function(E, selectors) {
  aux.copy <- copy(E)
  rule.list <- c()
  best.cpx <- ""
  while (!is.null(best.cpx) && dim(aux.copy)[1] != 0) {
    best.cpx <- find.best.complex(E, selectors)
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

selectors <- get.selectors(select(data, !y))
rules <- CN2(data, selectors)

