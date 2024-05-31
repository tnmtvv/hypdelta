from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="hypdelta",
    version="0.1.0",
    author="Tanya Matveeva",
    author_email="tpmatveeva@hse.ru",
    description="A Python library for calculating delta hyperbolicity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tnmtvv/hypdelta",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.18.0",
        "numba>=0.48.0",
        "scipy>=1.4.0",
    ],
)
