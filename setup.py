import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prettyNEAT",
    version="0.0.1",
    author="Dennis G. Wilson",
    author_email="d9w@pm.me",
    description="NeuroEvolution of Augmenting Topologies in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/d9w/prettyNEAT",
    project_urls={
        "Bug Tracker": "https://github.com/d9w/prettyNEAT/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Artificial Life",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "json",
    ],
    python_requires=">=3.5",
)
