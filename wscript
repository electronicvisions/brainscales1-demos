import os
from waflib.extras.test_base import summary

def depends(dep):
    pass

def options(opt):
    opt.load("test_base")

def configure(conf):
    conf.load("test_base")

def build(bld):

    srcdir = bld.path.find_dir('.').get_src()
    blddir = bld.path.find_dir('.').get_bld()
    testdir = blddir.find_or_declare('test')

    sphinxbuild = "python -m sphinx"

    # Build jupyter
    bld(name='doc-brainscales1-demos-jupyter',
        rule=f'{sphinxbuild} -M jupyter {srcdir} {blddir}/jupyter',
        always=True)

    # Build HTML
    bld(name='doc-brainscales1-demos-html',
        rule=f'{sphinxbuild} -M html {srcdir} {blddir}/html -W',
        always=True)

    # Patch ebrains kernel
    bld(name = 'patch_kernel',
        rule = f'for fn in {blddir}/jupyter/jupyter/*.ipynb; do patch $fn {srcdir}/ebrains_kernel.patch; done',
        after=["doc-brainscales1-demos-jupyter"],
    )

    bld.add_post_fun(summary)
