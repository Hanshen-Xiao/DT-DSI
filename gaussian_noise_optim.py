import time
import torch

def q2(A):
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    B = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    return B


def mirrorpgd(X, epsilon=None,log_process=False, log_func=None):
    if log_func is None:
        log_func=print
    def log(content):
        if log_process:
            return log_func(content)
    
    log("=Using mPGD=")
    # dealing with non full rank X
    start_time = time.time()
    
    dtype = X.dtype
    X=X.float()
    log(f"X original shape {X.shape}")

    XTX = (X.T @ X).cuda() #+ torch.diag(torch.rand(X.shape[1],device=X.device, dtype=X.dtype)*1e-3)
    eigvals, eigvecs = torch.linalg.eigh(XTX)
    s = torch.sqrt(torch.clamp(eigvals, min=0))
    valid = s > torch.max(s)*1e-3
    basis = X @ eigvecs[:, valid].to(X.device) 
    basis /= s[valid].to(X.device)
    r = basis.shape[1]
    cor = torch.diag(s[valid]) @ eigvecs.T[valid,:]
    if X.shape[1]<30:
        log(f"eigens: {s}")
    # log(f"decom error= {torch.norm(basis @ cor - X)}")
    X = cor

    n = X.shape[1]
    # n contains, r dimension of subspace

    log(f"X rank: {r}, X shape {X.shape} processing time: {time.time()-start_time}")
    log(f"Mean of X norm {torch.mean(torch.norm(X,dim=0))}")
    if n<=50:
        log(f"X norm {torch.norm(X,dim=0)}")
    
    X=X.double()
    log(f"memory consumption {torch.cuda.memory_allocated() / (1024 ** 3):.2f}G")
    
    mu = torch.ones(n, device=X.device, dtype=X.dtype)*10
    if epsilon is None:
        epsilon=torch.ones(n, device=X.device, dtype=X.dtype)
    epsilon=epsilon.double()
    # if n<30:
    #     log(f"epsilon: {epsilon}")

    def sigma_func(mu):
        return q2(X @ torch.diag(mu) @ X.T)
    def f_func(mu, sigma=None):
        if torch.count_nonzero(mu)<r:
            return torch.tensor(-float('inf'), dtype=mu.dtype, device=mu.device)
        if sigma is None:
            sigma_square=X @ torch.diag(mu) @ X.T
            eigenvalues, eigenvectors = torch.linalg.eigh(sigma_square)
            if torch.any(eigenvalues==0):
                return torch.tensor(-float('inf'), dtype=mu.dtype, device=mu.device)
            sqrt_eigenvalues = torch.sqrt(eigenvalues)
            sigma = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
        return 2*torch.trace(sigma)-torch.dot(epsilon, mu)
    def f_grad_func(mu, sigma=None):
        if sigma is None:
            sigma = q2(X @ torch.diag(mu) @ X.T)
        return torch.diagonal(X.T @ torch.linalg.inv(sigma) @ X)-epsilon

    f_val=f_func(mu)
    lr_0=10
    # log(f"lr_0: {lr_0} f_val={f_val}")
    decay=0.1
    iteration=0
    lr=lr_0
    lr_tolerance_reached=False
    while True:
        grad=f_grad_func(mu)
        if torch.isinf(grad).any() or torch.isnan(grad).any():
            raise ValueError(f'grad is inf or nan: {grad}')
        
        kkt_residual = torch.norm(torch.min(torch.abs(grad), mu))
        if kkt_residual < 1e-5:
            err=torch.max(torch.diagonal(X.T @ torch.linalg.inv(sigma_func(mu)) @ X)-epsilon)
            if torch.abs(err)<1e-3:
                log(f"time={time.time()-start_time}, KKT stopping criterion satisfied: number of iterations={iteration}, KKT residual={kkt_residual}, f_val={f_val}, lr={lr}")
                break
        if iteration>1000:
            log(f"Maximum iteration (1000) reached, iteration stops, time={time.time()-start_time}, number of iterations={iteration}, KKT residual={kkt_residual}, f_val={f_val}, lr={lr}")
            break
        if lr_tolerance_reached:
            log(f"lr<1e-100, iteration stops, time={time.time()-start_time}, number of iterations={iteration}, KKT residual={kkt_residual}, f_val={f_val}, lr={lr}")
            break
        if iteration%20==1:
            lr=lr_0
        while True:
            mu_new=torch.exp(torch.log(mu)+lr*(grad))
            f_new=f_func(mu_new)
            if f_new>f_val:
                mu=mu_new
                f_val=f_new
                # log(f"iteration={iteration}, f_val={f_val}, lr={lr}, kkt_residual={kkt_residual}")
                break
            lr*=decay
            if lr<1e-100:
                log(f"lr<1e-100, iteration stops, time={time.time()-start_time}, iteration={iteration}, f_val={f_val}, lr={lr}, kkt_residual={kkt_residual}")
                lr_tolerance_reached=True
                break

        iteration+=1
        freq=50
        if iteration%freq==0:
            log(f"iteration: {iteration}, f_val: {f_val}, lr: {lr}, kkt_residual: {kkt_residual}")

    # log(f"mu: {mu}")
    sigma=sigma_func(mu)
    diag=torch.diagonal(X.T @ torch.linalg.inv(sigma) @ X)
    log(f"max diag: {torch.max(diag)}")
    eigs,_=torch.linalg.eigh(sigma)
    linf=torch.max(eigs)**0.5
    l2=torch.sum(eigs)**0.5
    log(f"Max directional deviation = {linf}")
    log(f"original expected L2 norm = {l2}")
    L=torch.linalg.cholesky(sigma)
    return basis.to(dtype), L.to(dtype), [linf,l2]

def isotropic(X, epsilon=None,log_process=False, log_func=None):
    if log_func is None:
        log_func=print
    def log(content):
        if log_process:
            return log_func(content)
    
    log("=Using isotropic=")
    # dealing with non full rank X

    start_time = time.time()
    X=X.float()
    dtype = X.dtype
    log(f"X original shape {X.shape}")

    XTX = X.T @ X
    eigvals, eigvecs = torch.linalg.eigh(XTX)
    s = torch.sqrt(torch.clamp(eigvals, min=0))
    valid = s > torch.max(s)*1e-3
    basis = (X @ eigvecs[:, valid]) / s[valid]
    r = basis.shape[1]
    cor = torch.diag(s[valid]) @ eigvecs.T[valid,:]
    if True: #X.shape[1]<30:
        log(f"eigens: {s}")
    # log(f"decom error= {torch.norm(basis @ cor - X)}")
    X = cor
    n = X.shape[1]
    # n contains, r dimension of subspace

    log(f"X rank: {r}, X shape {X.shape} processing time: {time.time()-start_time}")
    
    X=X.double()
    log(f"memory consumption {torch.cuda.memory_allocated() / (1024 ** 3):.2f}G")
    
    if epsilon is None:
        epsilon=torch.ones(n, device=X.device, dtype=X.dtype)
    epsilon=epsilon.double()
    if n<30:
        log(f"epsilon: {epsilon}")

    diag=torch.diagonal(X.T @ X)
    std=torch.max(torch.div(diag,epsilon))**0.5

    # log(f"mu: {mu}")
    post_time=time.time()
    diag=(1/std**2)*diag
    log(f"max diag: {torch.max(diag)}")
    log(f"Max directional deviation = {std}")
    # log(f"mu {mu}")
    log(f"post time={time.time()-post_time}s")
    return basis.to(dtype), std.to(dtype)
