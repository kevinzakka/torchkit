import os

from setuptools import find_packages, setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DESCRIPTION = "torchkit is a lightweight library containing PyTorch utilities useful for day-to-day research."  # noqa: E501
TESTS_REQUIRE = [
    "pytest",
    "black",
    "isort",
    "pytype",
    "flake8",
]
DOCS_REQUIRE = [
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx-rtd-theme",
    "m2r",
    "docutils==0.16",
]


def readme() -> str:
    """Load README for use as package's long description."""
    with open(os.path.join(THIS_DIR, "README.md"), "r") as fp:
        return fp.read()


def get_version() -> str:
    locals_dict = {}
    with open(os.path.join(THIS_DIR, "torchkit", "version.py"), "r") as fp:
        exec(fp.read(), globals(), locals_dict)
    return locals_dict["__version__"]  # pytype: disable=key-error


setup(
    name="torchkit",
    version=get_version(),
    author="Kevin Zakka",
    license="MIT",
    description=DESCRIPTION,
    python_requires=">=3.8",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=1.3",
        "torchvision>=0.4",
        "tensorboard",
        "prettytable",
        "opencv-python",
        "moviepy",
    ],
    extras_require={
        "dev": ["ipdb", "jupyter", *TESTS_REQUIRE, *DOCS_REQUIRE],
        "test": TESTS_REQUIRE,
    },
    tests_require=TESTS_REQUIRE,
    url="https://github.com/kevinzakka/torchkit/",
)
