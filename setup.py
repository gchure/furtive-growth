from setuptools import setup, find_packages
import pathlib

__version__ = "0.0.1"

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="growth",
    version=__version__,
    long_description=README,
    description="Python utilities for simulating growth and death in different environments",
    long_description_content_type='text/markdown',
    url="https://github.com/gchure/furtive_growth",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    author="Griffin Chure",
    author_email="griffinchure@gmail.com",
    packages=find_packages(
        exclude=('docs', 'doc', 'sandbox', 'dev', 'growth.egg-info', 'simulations')),
    include_package_data=True,
    install_requires=[
        "matplotlib>=3.7.0",
        "numpy>=1.24.4",
        "pandas>=2.0.3",
        "scipy>=1.10.0",
        "seaborn>=0.12.2",
        "tqdm>=4.64.1",
    ],
)