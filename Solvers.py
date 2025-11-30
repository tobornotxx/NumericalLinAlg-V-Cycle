import numpy as np
import time
from Operators import compute_residuals_stokes
from Smoother import dgs_step
from Transfer import restrict_residuals, prolongate_error
from TrueSolution import get_exact_solution

def enforce_dirichlet_bc(u, v):
    '''
    强制实施 Dirichlet 边界条件 (u=0, v=0)
    注意：只处理 Dirichlet 边界。
    u: (N+1, N). Left(0) and Right(N) are Walls (Dirichlet u=0).
    v: (N, N+1). Bottom(0) and Top(N) are Walls (Dirichlet v=0).
    '''
    u[0, :] = 0.0
    u[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    return u, v

def v_cycle_recursive(u, v, p, f, g, g_div, h, bcs, nu1, nu2, min_N):
    N = p.shape[0]
    
    # Coarse Grid Solve
    if N <= min_N:
        if g_div is not None:
            g_div -= np.mean(g_div)
            # print(f"current centered g_div:{g_div}")

        for _ in range(100): 
            u, v, p = dgs_step(u, v, p, f, g, h, bcs, g_div)
            p -= np.mean(p)
        u, v = enforce_dirichlet_bc(u, v)
        return u, v, p

    # 1. Pre-Smooth
    # 在进入平滑前计算一次残差范数
    r_u_pre, r_v_pre, r_div_pre = compute_residuals_stokes(u, v, p, f, g, h, bcs)
    norm_pre = np.linalg.norm(r_u_pre) + np.linalg.norm(r_v_pre) + np.linalg.norm(r_div_pre)


    # 1. Pre-Smooth
    for _ in range(nu1):
        u, v, p = dgs_step(u, v, p, f, g, h, bcs, g_div)
    u, v = enforce_dirichlet_bc(u, v) # Important!
    p -= np.mean(p)

    # 2. Residuals
    r_u, r_v, r_div_computed = compute_residuals_stokes(u, v, p, f, g, h, bcs)
    
    print(f'r_div_comp[0,0]:{r_div_computed[0,0]:.2e}')
    if g_div is not None:
        r_div_real = r_div_computed + g_div
    else:
        r_div_real = r_div_computed

    # 3. Restrict
    f_c_inner, g_c_inner, g_div_c = restrict_residuals(r_u, r_v, r_div_real)

    g_div_c -= np.mean(g_div_c)
    
    # Setup Coarse Problem
    Nc = N // 2
    f_c = np.zeros((Nc + 1, Nc))
    f_c[1:-1, :] = f_c_inner
    g_c = np.zeros((Nc, Nc + 1))
    g_c[:, 1:-1] = g_c_inner
    
    # Error equations have zero BCs
    bcs_c = {'b': np.zeros(Nc - 1), 't': np.zeros(Nc - 1), 
             'l': np.zeros(Nc - 1), 'r': np.zeros(Nc - 1)}
    
    u_c = np.zeros((Nc + 1, Nc))
    v_c = np.zeros((Nc, Nc + 1))
    p_c = np.zeros((Nc, Nc))
    
    # 4. Recursion
    e_u, e_v, e_p = v_cycle_recursive(u_c, v_c, p_c, f_c, g_c, g_div_c, 
                                      2*h, bcs_c, nu1, nu2, min_N)
    
    # 5. Prolongate & Correct
    corr_u, corr_v, corr_p = prolongate_error(e_u, e_v, e_p)
    u += corr_u
    v += corr_v
    p += corr_p
    
    # Correction might be noisy at boundaries, clamp it immediately
    u, v = enforce_dirichlet_bc(u, v)
    
    # 6. Post-Smooth
    for _ in range(nu2):
        u, v, p = dgs_step(u, v, p, f, g, h, bcs, g_div)
        
    u, v = enforce_dirichlet_bc(u, v)
    p -= np.mean(p)

    # 在平滑后计算残差范数
    r_u_post, r_v_post, r_div_post = compute_residuals_stokes(u, v, p, f, g, h, bcs)
    norm_post = np.linalg.norm(r_u_post) + np.linalg.norm(r_v_post) + np.linalg.norm(r_div_post)
    
    print(f"[Level N={N}] Smoothing Check: Pre={norm_pre:.4e} -> Post={norm_post:.4e}")
    # if norm_post > norm_pre:
    #     print("!!! SMOOTHER IS DIVERGING !!!")
        
    return u, v, p

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
    

    # # --- DEBUG STEP 1 (Detailed): Boundary Residual Check ---
    # print("\n=== DEBUG: Detailed Boundary Residual Check ===")
    # u[:] = u_ex[:]
    # v[:] = v_ex[:]
    # p[:] = p_ex[:]

    # # 计算代入精确解后的残差
    # r_u, r_v, r_div = compute_residuals_stokes(u, v, p, f, g, h, bcs)

    # # 1. 检查 u 的上下边界 (Neumann/Dirichlet interface?)
    # # u 的 shape 是 (N+1, N)，Dim 1 是 Y 轴。
    # # u[:, 0] 是 Bottom, u[:, -1] 是 Top。
    # print(f"r_u (Internal Max): {np.max(np.abs(r_u[:, 1:-1])):.4e}")
    # print(f"r_u (Bottom Row 0) Max: {np.max(np.abs(r_u[:, 0])):.4e}")
    # print(f"r_u (Top Row -1)   Max: {np.max(np.abs(r_u[:, -1])):.4e}")
    # print("Max internal")
    # print(np.max(np.abs(r_u[1:-1,1:-1])))


    # # 2. 检查 v 的左右边界
    # # v 的 shape 是 (N, N+1)，Dim 0 是 X 轴。
    # # v[0, :] 是 Left, v[-1, :] 是 Right。
    # print(f"r_v (Internal Max): {np.max(np.abs(r_v[1:-1, :])):.4e}")
    # print(f"r_v (Left Col 0)   Max: {np.max(np.abs(r_v[0, :])):.4e}")
    # print(f"r_v (Right Col -1) Max: {np.max(np.abs(r_v[-1, :])):.4e}")

    # # 3. 检查 Divergence
    # print(f"r_div Max: {np.max(np.abs(r_div)):.4e}")

    # print("=== DEBUG END ===\n")

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