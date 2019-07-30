import setuptools

__version__ = "0.1.1"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Yeti",
    version=__version__,
    author="Sandro Wrzalek",
    author_email="sandro.wrzalek@fu-berlin.de",
    description="Trajectory Interface for biomolecules based on mDTraj",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sawrz/yeti",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
