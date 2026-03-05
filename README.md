# Numerical Methods

A practical introduction to numerical and linear algebra methods in Python, with implementations and tests for core algorithms used in computer science.

## Course Overview

This repository contains exercises, homework-style implementations, and supporting scripts from the Numerical Methods course at the University of Zurich (Fall Semester 2025).

**Grade: 5.25/6.0 | ECTS: 6**

## Topics Covered

- **Scientific Computing Basics**: Working with NumPy, vectorized operations, and numerical workflows
- **Linear Systems**: Gaussian elimination, pivoting, consistency checks, and structured problem solving
- **Matrix Factorization**: LU decomposition with permutation matrices and determinant computation
- **Linear Algebra Foundations**: Basis transformations, subspace checks, and orthonormalization
- **Eigenvalue Methods**: Power method and inverse power method
- **Data Approximation**: Polynomial interpolation and linear least squares
- **PCA (2D)**: Principal direction estimation from centered data
- **Numerical Solvability Concepts**: Rank, invertibility, and model reconstruction

## Repository Structure

```text
├── Exercise 0/
│   └── exercise_00.py
├── Exercise 1/
│   ├── backend.py
│   └── backend_test.py
├── Exercise 2/
│   ├── backend.py
│   ├── backend_test.py
│   ├── exercise_02.py
│   └── frontend.py
├── Exercise 3/
│   └── Exercise_3_Practice/
│       ├── backend.py
│       ├── backend_test.py
│       ├── exercise_03.py
│       └── frontend.py
├── Exercise 4/
│   └── Exercise_04_Practice/
│       ├── backend.py
│       ├── backend_test.py
│       ├── exercise_04.py
│       └── frontend.py
├── Exercise 5/
│   └── Exercise_05_Practice/
│       ├── backend.py
│       ├── backend_test.py
│       ├── exercise_05.py
│       └── frontend.py
└── Scripts/
    ├── 1.4 Interpolation.ipynb
    ├── 2.5 LU Example.ipynb
    └── 2.9 Solvability.ipynb
```

## Format

Materials are provided as Python modules and Jupyter notebooks:

- **`backend.py`** files for algorithm implementations
- **`backend_test.py`** files for automated validation
- **`frontend.py`** files for visualizations
- **Notebook scripts** for conceptual demonstrations and experimentation

## Key Learning Outcomes

- Implement stable numerical algorithms for solving linear algebra problems
- Analyze matrix properties (rank, solvability, decomposition quality)
- Apply eigenvector-based techniques to extract dominant directions from data
- Use least squares and interpolation for data fitting and approximation
- Build tested, modular scientific Python code

## Notes

This repository reflects hands-on coursework with focus on correctness, numerical stability, and reproducibility.

---

_Course instructors: Renato Pajarola and team_
