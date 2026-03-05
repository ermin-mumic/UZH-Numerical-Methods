import numpy as np

# Task a)
# Implement the gaussian elimination method, to solve the given system of linear equations;
# Add partial pivoting to increase accuracy and stability of the solution;
# Return the solution for x
# Assume a square matrix
def solveLinearSystem(A, b):
  A = A.astype(float)
  b = b.astype(float)
  n = len(b) # number of rows/columns

  # Forward elimination with partial pivoting
  for i in range(n): # For each column

    # Find the row with highest absolute value in current column i
    max_row = i
    for k in range(i+1,n): # For each row below current row i
      if abs(A[k, i]) > abs(A[max_row, i]):
        max_row = k
    
    # Swap the current row with the max_row
    if max_row != i:
      A[[i, max_row]] = A[[max_row, i]] 
      b[i], b[max_row] = b[max_row], b[i]
    
    # Create zeros in all positions below the pivot
    if abs(A[i, i]) > 1e-10: # Check the pivot element
      for k in range(i+1,n): # For each row below current row i
        factor = A[k, i] / A[i, i]
        A[k, i:] = A[k, i:] - factor * A[i, i:]
        b[k] = b[k] - factor * b[i]
    
  # Back substitution
  x = np.zeros(n)
  for i in range(n-1, -1, -1): # From last row to first row
    x[i] = b[i] 
    for j in range(i+1, n): # For each column to the right of the pivot
      x[i] = x[i] - A[i, j] * x[j] # Subtract known values
    if A[i, i] != 0:
      x[i] = x[i] / A[i, i] # Divide by the pivot element
  
  return x


# Task b)
# Implement a method, checking whether the system is consistent or not;
# Obviously, you're not allowed to use any method solving that problem for you.
# Return either true or false
def isConsistent(A,b):
  A = A.astype(float)
  b = b.astype(float)
  m = A.shape[0] # number of rows
  n = A.shape[1] # number of columns

  if len(b) != m:
    raise ValueError("Incompatible dimensions between A and b.")
  
  augmented = np.column_stack([A, b]) # Create augmented matrix [A|b]
  # Forward elimination on augmented matrix
  for i in range(min(m,n)):
    # Find the row with highest absolute value in current column i
    max_row = i
    for k in range(i+1,m): # For each row below current row i
      if abs(augmented[k, i]) > abs(augmented[max_row, i]):
        max_row = k
    
    # Swap the current row with the max_row
    if max_row != i:
      augmented[[i, max_row]] = augmented[[max_row, i]]

    # Create zeros in all positions below the pivot
    if abs(augmented[i, i]) > 1e-10: # Check the pivot element
      for k in range(i+1, m):  # For each row below current row i
        factor = augmented[k, i] / augmented[i, i] 
        augmented[k, i:] = augmented[k, i:] - factor * augmented[i, i:]  
  
  # Check for inconsistency: [0 0 ... 0 | nonzero]
  for i in range(m):
    if np.allclose(augmented[i, :-1], 0) and abs(augmented[i, -1]) > 1e-10:
      return False 
  
  return True  
    

# Task c)
# Implement a method to compute the daily amounts of chicken breast, brown rice, black beans and avocado to eat to achieve the daily nutritional intake described in the exercise;
# Return a vector x with the grams of chicken breast, brown rice, black beans and avocado to eat each day.
def solveNutrients(A, b):
  
  #Check consistency
  if not isConsistent(A, b):
    raise ValueError("The system is inconsistent.")
  
  # Check if A is square
  m,n = A.shape
  if m != n:
    raise ValueError("Non-square matrix not supported")
  
  solution = solveLinearSystem(A,b)
  solution_in_grams = solution * 10

  return np.round(solution_in_grams, 0)
  

