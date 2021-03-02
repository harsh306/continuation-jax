import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "sphinx_rtd_theme==0.5.1",
    "Sphinx==3.4.3",
    "matplotlib==3.1.2",
    "flax==0.3.0",
    "numpy==1.18.1",
    "jsonlines==2.0.0",
    "jax==0.2.9",
    "jaxlib",
    "nbsphinx",
]


setuptools.setup(
    name="continuation_jax",  # Replace with your own username
    version="0.0.2",
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
