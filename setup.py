from __future__ import division
from __future__ import print_function

import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
meta = {}
with open(os.path.join(here, 'bioseq', '__init__.py'), 'r') as f:
    exec(f.read(), meta)

description = """DI-bioseq: OpenDILab Decision Intelligence Biologic Sequence Searching Platform"""

setup(
    name=meta['__TITLE__'],
    version=meta['__VERSION__'],
    description=meta['__DESCRIPTION__'],
    long_description=description,
    author=meta['__AUTHOR__'],
    license='Apache License, Version 2.0',
    keywords='RL Bio-sequence Platform',
    pachages=[
        *find_packages(include=('bioseq')),
    ],
    python_requires='>=3.6',
    install_requires=[
        'di-engine>=0.3',
        'scikit-learn==0.24.2',
    ],
)