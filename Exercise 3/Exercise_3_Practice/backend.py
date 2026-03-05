import numpy as np

# Task a)
# Implement a method, calculating a base change matrix.
# Input: lists sourceBase and targetBase - lists of vectors (e.g. [np.array([1, 2, 3]), np.array([2, 0, 1]), ...])
# Output: Matrix A - a 2D np.array, with len(sourceBase) x len(targetBase) entries
def changeBase(sourceBase: list, targetBase: list) -> np.array:
  S = np.column_stack(sourceBase)
  T = np.column_stack(targetBase)

  T_inv = np.linalg.inv(T)
  return np.dot(T_inv, S)

# Task b)
# Implement a method, checking if a subBase spans a Subvectorspace of the space spanned by the given base
# Input: lists sourceBase and subSpace - lists of vectors (e.g. [np.array([1, 2, 3]), np.array([2, 0, 1]), ...])
# Output: bool
def spansSubSpace(sourceBase: list, subBase: list) -> bool:
  dim = sourceBase[0].shape[0]

  for v in sourceBase:
    if v.shape[0] != dim:
      return False
    
  for v in subBase:
    if v.shape[0] != dim:
      return False
  
  S = np.column_stack(sourceBase)
  rank_S = np.linalg.matrix_rank(S)

  for v in subBase:
    S_augmented_v = np.column_stack((S, v))
    rank_S_augmented_v = np.linalg.matrix_rank(S_augmented_v)
    if rank_S_augmented_v > rank_S:
      return False

  return True
