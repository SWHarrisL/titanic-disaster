# Installs the R packages that titanic_model.R needs.
#
# The rocker/r-ver base image downloads packages from a CRAN snapshot frozen
# on a fixed date, so every build installs exactly the same package versions.

packages <- c("dplyr")

install.packages(packages, Ncpus = parallel::detectCores())

installed <- vapply(packages, requireNamespace, logical(1), quietly = TRUE)
if (!all(installed)) {
  stop("Could not install: ", paste(packages[!installed], collapse = ", "))
}