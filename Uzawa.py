import numpy as np
# ==========================================
# Question 2: Uzawa Iteration Method Helpers
# ==========================================

def get_rhs_with_bcs(f, g, bcs, h):
    '''
    准备带边界条件的 RHS (对应 f 和 g)
    提取自 compute_residuals_stokes 的逻辑，用于 Uzawa 的速度求解步
    '''
    # 复制内部源项
    rhs_u = f[1:-1, :].copy()
    rhs_v = g[:, 1:-1].copy()

    # 加入边界条件到右边项
    rhs_u[:, 0] += bcs['b'] / h
    rhs_u[:, -1] += bcs['t'] / h
    rhs_v[0, :] += bcs['l'] / h
    rhs_v[-1, :] += bcs['r'] / h
    
    return rhs_u, rhs_v

def apply_laplacian_u_direction(p_dir, h):
    '''
    辅助函数：计算 A * p_dir，其中 p_dir 是 CG 中的搜索方向（内部点）
    需要将 p_dir 填充回 (N+1, N) 的全网格形式，边界为 0 (齐次 Dirichlet)
    '''
    from Operators import apply_laplacian_u
    N_minus_1, N = p_dir.shape
    p_full = np.zeros((N_minus_1 + 2, N))
    p_full[1:-1, :] = p_dir
    # 边界保持为 0
    return apply_laplacian_u(p_full, h)

def cg_solve_u(u, rhs, h, max_iter=200, tol=1e-10):
    '''
    共轭梯度法求解 -Laplacian(u) = rhs
    u: 初值 (N+1, N). in-place更新内部.
    rhs: (N-1, N)
    '''
    from Operators import apply_laplacian_u
    
    # 计算初始残差 r = b - Ax
    # 注意：apply_laplacian_u 处理了内部 stencil，这里使用的是当前的 u (包含边界)
    r = rhs - apply_laplacian_u(u, h)
    
    p = r.copy()
    rsold = np.sum(r * r)
    iter = max_iter
    
    for i in range(max_iter):
        # print(f'Current rsold:{rsold}')
        if rsold < tol**2:
            iter = i
            break
        
        # CG法计算逻辑
        # 计算 Ap。注意 p 只是方向，代表 update 量，因此其边界条件是 0
        Ap = apply_laplacian_u_direction(p, h)
        
        alpha = rsold / (np.sum(p * Ap) + 1e-30)
        
        # 更新 u 的内部
        u[1:-1, :] += alpha * p
        r -= alpha * Ap
        
        rsnew = np.sum(r * r)
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
        
    # print(f'CG for u converged after {max_iter} iters')
    return u, iter

def apply_laplacian_v_direction(p_dir, h):
    '''
    辅助函数：计算 A * p_dir for v
    '''
    from Operators import apply_laplacian_v
    N, N_minus_1 = p_dir.shape
    p_full = np.zeros((N, N_minus_1 + 2))
    p_full[:, 1:-1] = p_dir
    return apply_laplacian_v(p_full, h)

def cg_solve_v(v, rhs, h, max_iter=200, tol=1e-10):
    '''
    共轭梯度法求解 -Laplacian(v) = rhs
    v: Initial guess (N, N+1). Updates interior in-place.
    rhs: (N, N-1)
    '''
    from Operators import apply_laplacian_v
    
    r = rhs - apply_laplacian_v(v, h)
    
    p = r.copy()
    rsold = np.sum(r * r)
    iter = max_iter
    
    for i in range(max_iter):
        if rsold < tol**2:
            iter = i
            break
            
        Ap = apply_laplacian_v_direction(p, h)
        
        alpha = rsold / (np.sum(p * Ap) + 1e-30)
        
        v[:, 1:-1] += alpha * p
        r -= alpha * Ap
        
        rsnew = np.sum(r * r)
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    # print(f'CG for v converged after {max_iter} iters')
    return v, iter

def uzawa_step(u, v, p, f, g, h, bcs, alpha: float = 1.0, max_cg_iter: int = 100, cg_tol: float = 1e-10):
    '''
    Uzawa Iteration Step
    1. Solve A u_{k+1} = f - B p_k
    2. Update p_{k+1} = p_k + alpha * B^T u_{k+1} (where B^T u is -div u)
    '''
    from Operators import apply_gradient_p, apply_divergence_uv
    
    # 1. 准备速度方程的 RHS
    # F_u = f + BCs - grad_x(p)
    # F_v = g + BCs - grad_y(p)
    
    # 获取包含 BC 贡献的 f, g 部分
    rhs_u_base, rhs_v_base = get_rhs_with_bcs(f, g, bcs, h)
    
    # 计算压力梯度
    gp_x, gp_y = apply_gradient_p(p, h)
    
    # 组装最终 RHS
    rhs_u_total = rhs_u_base - gp_x
    rhs_v_total = rhs_v_base - gp_y
    
    # 2. 求解速度子问题 (使用 CG)
    # 这里 CG 的容差需要足够小，以保证外层 Uzawa 收敛
    u, u_iter = cg_solve_u(u, rhs_u_total, h, max_iter=max_cg_iter, tol=cg_tol)
    v, v_iter = cg_solve_v(v, rhs_v_total, h, max_iter=max_cg_iter, tol=cg_tol)

    # print(f"CG for u used {u_iter} iterations, for v used {v_iter} iterations")
    
    # 3. 更新压力
    # p_{k+1} = p_k - alpha * div(u_{k+1})
    # 注意：div 函数返回的是 div(u)，对应方程中的 B^T U = -div U。
    # Uzawa 更新: p += alpha * (defect). 
    # Stokes: div u = 0. Defect = 0 - div u = -div u.
    div_val = apply_divergence_uv(u, v, h)
    p = p - alpha * div_val
    
    return u, v, p