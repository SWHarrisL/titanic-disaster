# R environment for the Titanic survival model.
#
# Build (run from the repo root):  docker build -t titanic-r -f src/r/Dockerfile .
# Run:                             docker run --rm titanic-r

# Fixed R version; its packages come from a CRAN snapshot frozen on a set date
FROM rocker/r-ver:4.5.1

WORKDIR /app

# Install the R packages first; Docker caches this layer,
# so editing the script later doesn't trigger a reinstall
COPY src/r/install_packages.R src/r/
RUN Rscript src/r/install_packages.R

# Mirror the repo layout so the script finds src/data/ the same way
# inside Docker as it does from the repo root on your own machine
COPY src/data/ src/data/
COPY src/r/titanic_model.R src/r/

CMD ["Rscript", "src/r/titanic_model.R"]