import numpy as np

# Task a)
# Implement a method performing least squares approximation of a linear courve.
# Input: Vectors x,y. Both 1D np.array of same size.
# Output: list of factors [m, b] representing the linear courve f(x) = mx + b.
def linearLSQ(x: np.array, y: np.array) -> list:
  X = np.column_stack((np.ones_like(x), x))
  XtX= np.dot(X.T, X)
  Xty = np.dot(X.T, y)
  factors = np.linalg.solve(XtX, Xty)
  return [factors[1], factors[0]]  

# Task b)
# Implement a method, orthogornalizing the given basis.
# Input: sourceBase - list of vectors, as in a)
# Output: orthoronalizedBase - list of vectors, with same size and shape as sourceBase
def orthonormalize(sourceBase: list) -> list:
  orthonormalBase = []
  for v in sourceBase:
    w = v.copy()
    for u in orthonormalBase:
      w = w - np.dot(w, u)/np.dot(u, u) * u
    w = w / np.linalg.norm(w)
    orthonormalBase.append(w)
  return orthonormalBase

