#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mengutils", # Replace with your own username
    version="1.0.5",
    author="Mengmeng Kuang",
    author_email="kuangmeng@msn.com",
    description="A simple common utils and neural networks package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kuangmeng/MengUtils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)