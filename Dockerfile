FROM jupyter/base-notebook:2023-10-20

# Add these lines to your Dockerfile before installing other packages
RUN pip install plotly==5.15.0 ipywidgets==7.7.2

# Also ensure ipywidgets extension is enabled
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager


# Install conda packages
RUN mamba install -c conda-forge \
    leafmap \
    geopandas \
    localtileserver \
    backports.tarfile \
    pandas \
    plotly \
    statsmodels \
    openpyxl \ 
    -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Copy and install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
# Assuming your data files are in a 'data' directory
RUN mkdir ./data
COPY /data ./data

RUN mkdir ./pages
COPY /pages ./pages

ENV PROJ_LIB='/opt/conda/share/proj'

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

EXPOSE 8765

CMD ["solara", "run", "./pages", "--host=0.0.0.0"]