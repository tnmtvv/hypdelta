from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hyppy",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="HypDelta: A tool for calculating delta hyperbolicity of distance matrices.",
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
    package_dir={"": "src"},
    packages=["src/hyppy", "tests"],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.4",
        "scipy>=1.7.0",
        "numba>=0.57.1",
        "pytest==7.4.3",
        "PyYAML==6.0.1",
        "scikit_learn==1.3.0",
        "scipy==1.11.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "flake8>=3.9.2",
            "black>=21.5b1",
        ],
    },
    entry_points={
        "console_scripts": [
            "hypdelta=hyppy.hypdelta:main",
        ],
    },
)
