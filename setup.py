# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


def open_requirements(fname):
    with open(fname, mode='r') as f:
        requires = f.read().split('\n')
    requires = [e for e in requires if len(e) > 0 and not e.startswith('#')]
    return requires


d = {}
exec(open("delta2D_numcodecs/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

install_requires = open_requirements('requirements.txt')
entry_points = {"numcodecs.codecs": ["delta2D = delta2D_numcodecs:Delta2D"]}

setup(
    name="delta2D_numcodecs",
    version=version,
    author="Alessio Buccino",
    author_email="alessiop.buccino@gmail.com",
    description="Numcodecs implementation of delta filter in 2D.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AllenNeuralDynamics/delta2D-numcodecs",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    entry_points=entry_points,
)
