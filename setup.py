import pathlib

from setuptools import setup

# The directory which contains this file
HERE = pathlib.Path(__file__).parent
PKG_PATH = HERE.resolve().as_posix()

# The readme file
README = (HERE / "README.md").read_text()

setup(
    name="sensipy",
    description="A tool to simulate gravitational wave follow-up observations with gamma-ray observatories.",
    version="2.0",
    package_dir={"": "src"},
    packages=["sensipy"],
    url="https://github.com/astrojarred/sensipy/",
    author="Jarred Green (MPP), Barbara Patricelli (INAF)",
    long_description=README,
    long_description_content_type="text/markdown",
    author_email="jarred.green@inaf.it",
)
