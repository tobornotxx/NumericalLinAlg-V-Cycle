import numpy as np
import time
from Operators import compute_residuals_stokes
from TrueSolution import get_exact_solution
from dgs import v_cycle_recursive
from Uzawa import uzawa_step



def solve_problem_1(N, tol=1e-8, max_iter=20):
    h = 1.0 / N
    u = np.zeros((N + 1, N))
    v = np.zeros((N, N + 1))
    p = np.zeros((N, N))
    
    u_ex, v_ex, p_ex, f, g = get_exact_solution(N)

        # === NEW: Initialize u, v with discrete-exact BC ===
    u[0, :]  = u_ex[0, :]      # left wall
    u[-1, :] = u_ex[-1, :]     # right wall
    v[:, 0]  = v_ex[:, 0]      # bottom wall
    v[:, -1] = v_ex[:, -1]     # top wall


    
    # Boundary Conditions
    x_u_internal = (np.arange(1, N) * h)
    bc_b = -2 * np.pi * (1 - np.cos(2 * np.pi * x_u_internal))
    bc_t =  2 * np.pi * (1 - np.cos(2 * np.pi * x_u_internal))
    
    y_v_internal = (np.arange(1, N) * h)
    bc_l = 2 * np.pi * (1 - np.cos(2 * np.pi * y_v_internal))
    bc_r = -2 * np.pi * (1 - np.cos(2 * np.pi * y_v_internal))
    
    bcs = {'b': bc_b, 't': bc_t, 'l': bc_l, 'r': bc_r}
    

    # Init Residual
    r_u, r_v, r_div = compute_residuals_stokes(u, v, p, f, g, h, bcs)
    norm_r0 = np.sqrt(np.sum(r_u**2) + np.sum(r_v**2) + np.sum(r_div**2))
    
    print(f"N={N}, Initial Residual: {norm_r0:.4e}")
    start_time = time.time()
    

    iters = max_iter
    for k in range(max_iter):
        u, v, p = v_cycle_recursive(u, v, p, f, g, None, h, bcs, 3, 3, 4)
        
        p = p - np.mean(p)
        
        r_u, r_v, r_div = compute_residuals_stokes(u, v, p, f, g, h, bcs)
        norm_r = np.sqrt(np.sum(r_u**2) + np.sum(r_v**2) + np.sum(r_div**2))
        
        ratio = norm_r / norm_r0
        print(f"Iter {k+1}: Norm = {norm_r:.4e}, Ratio = {ratio:.4e}")
        
        if np.isnan(norm_r) or np.isinf(norm_r):
            print("Diverged!")
            break
            
        if ratio <= tol:
            iters = k + 1
            break
            
    cpu_time = time.time() - start_time
    
    err_u_sq = np.sum((u - u_ex)**2)
    err_v_sq = np.sum((v - v_ex)**2)
    error_L2 = np.sqrt(h*h * (err_u_sq + err_v_sq))
    
    print(f"Done N={N}. Iter={iters}, Time={cpu_time:.2f}s, Error={error_L2:.4e}")
    return iters, cpu_time, error_L2, u, v, p, u_ex, v_ex, p_ex



def solve_problem_2(N, alpha=1.0, tol=1e-8, max_iter=10000):
    '''
    使用 Uzawa Iteration Method 求解
    '''
    from Operators import compute_residuals_stokes
    
    h = 1.0 / N
    u = np.zeros((N + 1, N))
    v = np.zeros((N, N + 1))
    p = np.zeros((N, N))
    
    # 获取真解用于设置边界和计算误差
    u_ex, v_ex, p_ex, f, g = get_exact_solution(N)

    # 初始化边界条件 (Dirichlet)
    u[0, :]  = u_ex[0, :]
    u[-1, :] = u_ex[-1, :]
    v[:, 0]  = v_ex[:, 0]
    v[:, -1] = v_ex[:, -1]
    
    # 准备 Neumann 边界数据
    x_u_internal = (np.arange(1, N) * h)
    bc_b = -2 * np.pi * (1 - np.cos(2 * np.pi * x_u_internal))
    bc_t =  2 * np.pi * (1 - np.cos(2 * np.pi * x_u_internal))
    
    y_v_internal = (np.arange(1, N) * h)
    bc_l = 2 * np.pi * (1 - np.cos(2 * np.pi * y_v_internal))
    bc_r = -2 * np.pi * (1 - np.cos(2 * np.pi * y_v_internal))
    
    bcs = {'b': bc_b, 't': bc_t, 'l': bc_l, 'r': bc_r}
    
    # 计算初始残差
    r_u, r_v, r_div = compute_residuals_stokes(u, v, p, f, g, h, bcs)
    norm_r0 = np.sqrt(np.sum(r_u**2) + np.sum(r_v**2) + np.sum(r_div**2))
    
    print(f"Uzawa (N={N}): Initial Residual = {norm_r0:.4e}")
    
    start_time = time.time()
    iters = max_iter
    
    for k in range(max_iter):
        # Uzawa 迭代步
        u, v, p = uzawa_step(u, v, p, f, g, h, bcs, alpha)
        
        # 强制压力零均值
        p -= np.mean(p)
        
        # 检查残差
        r_u, r_v, r_div = compute_residuals_stokes(u, v, p, f, g, h, bcs)
        norm_r = np.sqrt(np.sum(r_u**2) + np.sum(r_v**2) + np.sum(r_div**2))
        
        ratio = norm_r / norm_r0
        
        if k % 100 == 0:
            print(f"Iter {k}: Norm = {norm_r:.4e}, Ratio = {ratio:.4e}")
            
        if ratio <= tol:
            iters = k + 1
            print(f"Converged at iter {k+1}, Ratio = {ratio:.4e}")
            break
            
        if np.isnan(norm_r) or norm_r > 1e10:
            print("Diverged!")
            break
            
    cpu_time = time.time() - start_time
    
    # 计算误差 e_N (Discrete L2 norm)
    # e_N = sqrt( h^2 * (sum(du^2) + sum(dv^2)) )
    # 注意：这里的误差公式通常只针对内部点，但为了简单起见，全场求和（边界误差为0）也可以
    err_u_sq = np.sum((u - u_ex)**2)
    err_v_sq = np.sum((v - v_ex)**2)
    error_L2 = np.sqrt(h*h * (err_u_sq + err_v_sq))
    
    print(f"Done Uzawa N={N}. Iter={iters}, Time={cpu_time:.2f}s, Error={error_L2:.4e}")
    
    return iters, cpu_time, error_L2, u, v, p, u_ex, v_ex, p_ex