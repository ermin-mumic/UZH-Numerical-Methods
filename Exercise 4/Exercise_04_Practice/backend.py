import numpy as np

# Task a)
# Implement a method, calculating the largest eigenvector of A with b as an initial guess.
# Input: Matrix A, Vector b. A - 2D numpy array, b - 1D numpy array
# Output: The Eigenvector of A with the largest (absolute) Eigenvalue, given as 1D np.array.
def powerMethod(A: np.array, b: np.array) -> np.array:

  x = b.astype(float) # set intial vector
  x_norm = np.linalg.norm(x) # length of intial vector

  if x_norm == 0:
    raise ValueError("Initial vector b must not be the zero vector.")
  
  x = x / x_norm # normalize initial vector (length = 1)

  # Power method iterations
  for _ in range(1000):
    y = np.dot(A,x) # multiply A with current vector
    y_norm = np.linalg.norm(y) # calculate length of resulting vector

    if y_norm == 0:
      return x
    
    x_new = y / y_norm # normalize resulting vector

    if np.linalg.norm(x_new - x) < 1e-6:
      return x_new
    
    x= x_new
  
  return x


# Task b)
# Implement a method, calculating the smallest eigenvector of A with b as an initial guess.
# Input: Matrix A, Vector b. A - 2D numpy array, b - 1D numpy array
# Output: The Eigenvector of A with the smallest (absolute) Eigenvalue, given as 1D np.array.
def inversePowerMethod(A: np.array, b: np.array) -> np.array:
  
  x = b.astype(float) # set intial vector
  x_norm = np.linalg.norm(x) # length of intial vector

  if x_norm == 0:
    raise ValueError("Initial vector b must not be the zero vector.")
  
  x = x / x_norm # normalize initial vector (length = 1)

  # Inverse Power method iterations
  for _ in range(1000):
    y = np.linalg.solve(A, x) # solve Ay = x for y which is same as y = A^-1 * x
    y_norm = np.linalg.norm(y)

    if y_norm == 0:
      return x

    x_new = y / y_norm

    if np.linalg.norm(x_new - x) < 1e-6:
      return x_new

    x = x_new

  return x


# Task c)
# Implement a method performing a PCA on given data.
# Input: Vectors x, y. Both 1D np.array of same size.
# Output: The Principal direction of the given data, represented as 1D np.array
def linearPCA(x: np.array, y: np.array) -> np.array:
  data = np.column_stack((x, y)) 
  n = data.shape[0] # shape (n, 2) where n is the number of data points

  mean = np.mean(data, axis=0) # axis=0 to get mean of each column (mean x and mean y)
  centered = data - mean # row-wise subtraction of mean (from every x subtract mean x, from every y subtract mean y)

  cov = (centered.T @ centered) / (n - 1) # covariance matrix

  b0 = np.array([1.0, 0.0]) # initial vector for power method
  principal_direction = powerMethod(cov, b0) # use power method to get principal direction

  return principal_direction
