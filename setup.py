from setuptools import setup, find_packages

setup(
    name="symbolic_regression",
    version="0.1.0",
    description="Symbolic Regression with Genetic Programming",
    author="KevinHaka",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "pysr",
    ],
    python_requires=">=3.8",
)
