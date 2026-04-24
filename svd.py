import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1,2],[3,4],[5,6]])

def svd(A):
    val_ata, vec_ata = np.linalg.eig(A.T@A)
    val_aat, vec_aat = np.linalg.eig(A@A.T)

    # Sort values and vectors
    idx = np.argsort(val_ata)[::-1]
    val_sort = val_ata[idx]
    V = vec_ata[:, idx]

    idx_u = np.argsort(val_aat)[::-1]
    U = vec_aat[:, idx_u]

    singular_values = np.sqrt(np.maximum(val_sort, 0))
    sigma = np.zeros(A.shape)
    np.fill_diagonal(sigma, singular_values)

    return U, sigma, V

print(svd(A))
u,sg,v = svd(A)
print(u@sg@v.T)
print(np.linalg.svd(A))