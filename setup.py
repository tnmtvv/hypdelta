from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hypdelta",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library for calculating delta hyperbolicity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tnmtvv/hypdelta",
    project_urls={
        "Bug Tracker": "https://github.com/tnmtvv/hypdelta/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=["src/hypdelta", "tests"],
    python_requires=">=3.6",
)
