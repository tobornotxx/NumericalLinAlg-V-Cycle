import numpy as np
from Operators import compute_residuals_stokes
from Smoother import dgs_step
from Transfer import restrict_residuals, prolongate_error


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