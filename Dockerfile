ARG MINICONDA_IMAGE_TAG=4.10.3

FROM continuumio/miniconda3:$MINICONDA_IMAGE_TAG AS base

# install poetry
COPY ./requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# create new environment
# see: https://jcristharif.com/conda-docker-tips.html
# warning: for some reason conda can hang on "Executing transaction" for a couple of minutes
COPY environment.yaml /tmp/environment.yaml
RUN conda env create -f /tmp/environment.yaml && \
    conda clean -afy && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.pyc' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete

# "activate" environment for all commands (note: ENTRYPOINT is separate from SHELL)
SHELL ["conda", "run", "--no-capture-output", "-n", "kilroy-module-huggingface", "/bin/bash", "-c"]

# add poetry files
COPY ./kilroy_module_huggingface/pyproject.toml ./kilroy_module_huggingface/poetry.lock /tmp/kilroy_module_huggingface/
WORKDIR /tmp/kilroy_module_huggingface

FROM base AS test

# install dependencies only (notice that no source code is present yet) and delete cache
RUN poetry install --no-root --only main,test && \
    rm -rf ~/.cache/pypoetry

# add source, tests and necessary files
COPY ./kilroy_module_huggingface/src/ /tmp/kilroy_module_huggingface/src/
COPY ./kilroy_module_huggingface/tests/ /tmp/kilroy_module_huggingface/tests/
COPY ./kilroy_module_huggingface/LICENSE ./kilroy_module_huggingface/README.md /tmp/kilroy_module_huggingface/

# build wheel by poetry and install by pip (to force non-editable mode)
RUN poetry build -f wheel && \
    python -m pip install --no-deps --no-index --no-cache-dir --find-links=dist kilroy-module-huggingface

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "kilroy-module-huggingface", "pytest"]
CMD []

FROM base AS production

# install dependencies only (notice that no source code is present yet) and delete cache
RUN poetry install --no-root --only main && \
    rm -rf ~/.cache/pypoetry

# add source and necessary files
COPY ./kilroy_module_huggingface/src/ /tmp/kilroy_module_huggingface/src/
COPY ./kilroy_module_huggingface/LICENSE ./kilroy_module_huggingface/README.md /tmp/kilroy_module_huggingface/

# build wheel by poetry and install by pip (to force non-editable mode)
RUN poetry build -f wheel && \
    python -m pip install --no-deps --no-index --no-cache-dir --find-links=dist kilroy-module-huggingface

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "kilroy-module-huggingface", "kilroy-module-huggingface"]
CMD []
