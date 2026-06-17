ARG CONDA_ENV=for_container

FROM condaforge/miniforge3
# scope var from global
ARG CONDA_ENV

# I believe this is to avoid permission issues with 
# manipulating added files to places like /opt
RUN old_umask=`umask` \
    && umask 0000 \
    && umask $old_umask

# Singularity uses tini, but raises warnings
# we set it up here correctly for singularity
ADD ./lib/tini /tini
RUN chmod +x /tini
    
# singularity doesn't use the -s flag, and that causes warnings.
# -g kills process group on ctrl+C
ENTRYPOINT ["/tini", "-s", "-g", "--"]

# https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
# create conda env from yaml config
COPY ./genome_qc.yml /opt/genome_qc.yml
# use an external cache for solved environments
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba env create -f /opt/genome_qc.yml \
    && mamba create -n ${CONDA_ENV} -c bioconda snakemake=7.20.0
# add bins to PATH so that the env appears "active"
ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:/app:$PATH
# add local scripts/executables
COPY ./*.py /app/
COPY ./LICENSE /app/
COPY ./README.md /app/
COPY ./Snakefile.genome_qc /app/ws/Snakefile.genome_qc
COPY ./test /app/test
COPY ./config_template.yaml /app/ws/config.yaml
RUN ln -s /app/ws/config.yaml /app/config.yaml
RUN ln -s /app/ws/Snakefile.genome_qc /app/Snakefile.genome_qc
