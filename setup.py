#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='verse',
    version='0.1',
    description='AutoVerse',
    author='Yangge Li, Katherine Braught, Haoqing Zhu',
    maintainer='Yangge Li, Katherine Braught, Haoqing Zhu',
    maintainer_email='{li213, braught2, haoqing3}@illinois.edu',
    packages=find_packages(exclude=["tests", "demo"]),
    python_requires='>=3.8',
    install_requires=[
        # "numpy",
        # "scipy",
        # "matplotlib",
        # "polytope",
        # "pyvista",
        # "networkx",
        # "sympy",
        # "six",
        # "astunparse",
        # "z3-solver",
        # "plotly",
        # "beautifulsoup4",
        # "lxml",
        # "torch",
        # "tqdm",
        # "intervaltree",
        # "Pympler",
        # "nbformat",
        "ray~=2.2.0",
        "astunparse~=1.6.3",
        "beautifulsoup4~=4.11.1",
        "intervaltree~=3.1.0",
        "lxml~=4.9.1",
        "matplotlib~=3.4.3",
        "numpy~=1.22.4",
        "plotly~=5.8.2",
        "polytope~=0.2.3",
        "Pympler~=1.0.1",
        "pyvista~=0.32.1",
        "scipy~=1.8.1",
        "six~=1.14.0",
        "sympy~=1.6.2",
        "torch~=1.12.1",
        "tqdm~=4.64.1",
        "z3-solver~=4.8.17.0",
        "treelib~=1.6.1",
        "portion~=2.3.1",
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.8',
    ]
)
