# BrainScaleS-1 Demos and Tutorials

This repository contains usage examples for BrainScaleS-1.

If you want to execute the notebooks, you can clone them from our github repository and execute them on the EBRAINS Platform (www.ebrains.eu).
Simply use an existing collabatory or create a new one.
There, execute:
`!git clone https://github.com/electronicvisions/brainscales1-demos.git --branch jupyter-notebooks`
in a notebook of your JupyterLab session.

To be able to use the EBRAINS software environment for your notebook, please use the EBRAINS-experimental kernel, which should be set by default.
The currently used kernel is shown in the status line at the bottom of the notebook and in the upper right hand corner of the notebook.


## Build Notebooks locally

Run in container (`singularity shell --app dls /containers/stable/latest`):

```shell
make html
make jupyter
```

To look at the result files: `jupyter-notebook _build/jupyter/jupyter`.
