import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "jax==0.2.9",
    "sphinx>=3.4.3",
    "sphinx-rtd-theme",
    "nbsphinx",
]


setuptools.setup(
    name="continuation-jax",  # Replace with your own username
    version="0.0.1",
    author="Harsh Nilesh Pathak",
    author_email="harshnpathak@gmail.com",
    description="Continuation Methods for Deep Neural Networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harsh306/continuation-jax",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)
