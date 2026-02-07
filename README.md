# Symbolic Regression using Genetic Programming

A Python implementation of symbolic regression using genetic programming techniques.

This project was developed as part of a Master's thesis titled "Effective Implementation of Symbolic Regression with Genetic Programming on High-Dimensional Data".

## Description

This project implements various genetic programming methods for symbolic regression, including:
- GP (Genetic Programming)
- GPPI (Genetic Programming with Permutation Importance)
- GPSHAP (Genetic Programming with SHAP)
- GPCMI (Genetic Programming with Conditional Mutual Information)
- RFGP (Residual Fitting Genetic Programming)

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

2. Install the package:

```bash
# From local directory
pip install -e .

# Or directly from GitHub
pip install git+https://github.com/KevinHaka/Symbolic-Regression-Genetic-Programming.git
```

3. Install Julia dependencies for PySR:

```bash
python -c "from pysr import PySRRegressor; PySRRegressor()"
```

## Usage

1. Configure the parameters in [examples/run.py](examples/run.py) according to your needs.

2. (Optional) If you want to receive email notifications, fill in the email settings in [examples/.env](examples/.env)

3. Run the example:

```bash
cd examples
python run.py
```

## References

This project implements methods based on the following papers:

#### GPSHAP
Wang, C., Chen, Q., Xue, B., & Zhang, M. (2024). Improving Generalization of Genetic Programming for High-Dimensional Symbolic Regression with Shapley Value Based Feature Selection. *Data Science and Engineering*. https://doi.org/10.1007/s41019-024-00270-x

#### GPPI
Chen, Q., Zhang, M., & Xue, B. (2017). Feature Selection to Improve Generalization of Genetic Programming for High-Dimensional Symbolic Regression. *IEEE Transactions on Evolutionary Computation*, 21(5), 792-806. https://doi.org/10.1109/TEVC.2017.2683489

#### GPCMI
Kugiumtzis, D. (2013). Direct-coupling information measure from nonuniform embedding. *Physical Review E*, 87(6), 062918. https://doi.org/10.1103/PhysRevE.87.062918

#### PySR
Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. *ArXiv, abs/2305.01582*. https://arxiv.org/abs/2305.01582

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
