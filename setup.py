import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

version_dict = {}
with open("zoidberg/_version.py") as f:
    exec(f.read(), version_dict)

name = "zoidberg"
version = version_dict["__version__"]
release = version

setuptools.setup(
    name=name,
    version=version,
    url="https://github.com/boutproject/zoidberg",
    author="Peter Hill",
    author_email="peter.hill@york.ac.uk",
    description="Generate flux-coordinate independent (FCI) grids for BOUT++",
    python_requires=">=3.6",
    install_requires=[
        "boututils",
        "numpy>=1.14.1",
        "scipy>=1.0.0",
        "matplotlib>=3.1.1",
        "netcdf4>=1.4.0",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    packages=setuptools.find_packages(),
    project_urls={
        "Bug Tracker": "https://github.com/boutproject/zoidberg/issues/",
        "Documentation": "https://bout-dev.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/boutproject/zoidberg/",
    },
    command_options={
        "build_sphinx": {
            "project": ("setup.py", name),
            "version": ("setup.py", version),
            "release": ("setup.py", release),
            "source_dir": ("setup.py", "docs"),
        }
    },
)
