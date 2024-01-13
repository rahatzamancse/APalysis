# Use a multi-stage build to keep the final image as small as possible

# Stage 1: Build the React app
FROM node:16-alpine as node-base
WORKDIR /app
COPY src/activation_pathway_analysis_tool_frontend/package.json ./
RUN npm install
COPY src/activation_pathway_analysis_tool_frontend/ .
RUN REACT_APP_BACKEND_PORT=8000 npm run build

# Stage 2: Install all python dependencies
FROM python:3.10-slim as python-base
ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.7.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    \
    # paths
    # this is where our requirements + virtual environment will live
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# `builder-base` stage is used to build deps + create our virtual environment
FROM python-base as builder-base
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for installing poetry
        curl \
        # deps for building python deps
        build-essential

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -

# copy project requirement files here to ensure they will be cached.
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./

# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
RUN poetry install --no-dev --no-root


# `production` image used for runtime
FROM python-base as production
RUN apt-get update && apt-get install -y redis-server && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV FASTAPI_ENV=production
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH

# Preload the model and dataset
RUN python3 -m nltk.downloader wordnet
RUN python3 -c "import tensorflow as tf; tf.keras.applications.VGG16(weights='imagenet')"
RUN python3 -c "import tensorflow_datasets as tfds; tfds.load('imagenette/320px-v2', shuffle_files=False, with_info=True, as_supervised=True, batch_size=None)"

COPY src/activation_pathway_analysis_backend/* ./activation_pathway_analysis_backend/
COPY --from=node-base /app/build /app/activation_pathway_analysis_backend/static
COPY src/run-tf.py .


# Expose the port that FastAPI will run on
EXPOSE 8000

CMD redis-server --daemonize yes && python3 run-tf.py
