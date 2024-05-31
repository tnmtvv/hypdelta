from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hyppy",
    version="0.1.0",
    author="Tanya Matveeva",
    author_email="tpmatveeva@hse.ru",
    description="A Python library for computing hyperbolic delta values from distance matrices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tnmtvv/hyppy",
    project_urls={
        "Bug Tracker": "https://github.com/tnmtvv/hyppy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy", "numba", "scipy", "matplotlib"],
)
