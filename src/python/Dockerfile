# Python environment for the Titanic survival model.
#
# Build (run from the repo root):  docker build -t titanic-python -f src/python/Dockerfile .
# Run:                             docker run --rm titanic-python

FROM python:3.12-slim

# Show print() output immediately and keep pip's notices out of the build log
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# Install the pinned packages first; Docker caches this layer,
# so editing the script later doesn't trigger a reinstall
COPY src/python/requirements.txt src/python/
RUN pip install --no-cache-dir -r src/python/requirements.txt

# Mirror the repo layout so the script finds src/data/ the same way
# inside Docker as on your own machine
COPY src/data/ src/data/
COPY src/python/titanic_model.py src/python/

CMD ["python", "src/python/titanic_model.py"]