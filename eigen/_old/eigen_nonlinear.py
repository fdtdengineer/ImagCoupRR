#%%
import numpy as np



if __name__ == "__main__":
    A = np.array([[1, 2], [2, 1]])
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be a square matrix.")
    # debug
    print(np.linalg.eig(A))
    
    omega0 = 2
    #In = np.eye(n)


    def loss(arg):
        omega = arg[0]
        x0 = np.array(arg)
        x0[0] = 1
        v = A @ x0 - omega* x0
        loss = np.conj(v).T @ v

        return loss
    
    # optimization
    from scipy.optimize import minimize

    np.random.seed(0)
    #x0 = np.random.rand(n+1-1)
    x0 = np.array([0, 1])
    res = minimize(loss, x0, tol=1e-20)
    
    omega = res.x[0]
    x = np.concatenate([np.array([1]), res.x[1:]])
    x /= np.linalg.norm(x)
    print("omega: ", omega)
    print("x: ", x)


 

# %%
