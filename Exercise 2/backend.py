import numpy as np

# Task a)
# Implement a method, calculating the LU factorization of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: Matrices P, L and U - same shape as A each.
def lu(A):
  n = A.shape[0]
  P = np.eye(n)
  L = np.eye(n)
  U = np.copy(A).astype(float)


  for i in range(n-1):
    max_row_index = i
    for r in range(i+1, n):
      if abs(U[r,i]) > abs(U[max_row_index, i]):
        max_row_index = r

    if max_row_index != i:
      U[[i, max_row_index], :] = U[[max_row_index, i], :]
      P[[i, max_row_index], :] = P[[max_row_index, i], :]
      L[[i, max_row_index], :i] = L[[max_row_index, i], :i]

    if abs(U[i, i]) < 1e-12:
      raise ("Zero pivot encountered.")
    
    for r in range(i+1, n):
      L[r, i] = U[r, i] / U[i, i]
      U[r, i:] -= L[r, i] * U[i, i:]

  return P, L, U

# Task b)
# Implement a method, calculating the determinant of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: The determinant - a floating number
def determinant(A):
  n = A.shape[0]
  if n == 1:
        return A[0, 0]
  if n == 2:
        return A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]

  det = 0.0
  for j in range(n):
    if A[0, j] == 0:
      continue  
    sub = np.delete(np.delete(A, 0, axis=0), j, axis=1)
    sign = (-1) ** j
    det += sign * A[0, j] * determinant(sub)

  return det

  