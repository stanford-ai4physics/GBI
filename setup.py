import re
import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSIONFILE = "paws/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name="paws",
    version=verstr,
    author="Chi Lung Cheng",
    author_email="chi.lung.cheng@cern.ch",
    description="Code base for the Prior-Assisted Weak Supervision (PAWS) method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hep-lbdl/PAWS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'click',
        'pandas',
        'awkward',
        'vector',
        'aliad',
        'quickstats'
    ],
    scripts=['bin/paws'],
    python_requires='>=3.8',
    license="MIT",
)