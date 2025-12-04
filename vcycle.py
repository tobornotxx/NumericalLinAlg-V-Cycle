import numpy as np
from Operators import compute_residuals_stokes
from Smoother import dgs_step
from Transfer import restrict_residuals, prolongate_error
from InexactUzawa import dgs_velocity_step


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
    
    # print(f'r_div_comp[0,0]:{r_div_computed[0,0]:.2e}')
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
    
    # print(f"[Level N={N}] Smoothing Check: Pre={norm_pre:.4e} -> Post={norm_post:.4e}")
    # if norm_post > norm_pre:
    #     print("!!! SMOOTHER IS DIVERGING !!!")
        
    return u, v, p

def v_cycle_velocity(u, v, f_minus_bp, h, bcs,
                     nu1=2, nu2=2, min_N=4):
    """
    V-cycle for the velocity-only subproblem A U = F - B P_k.

    参数
    ----
    u: ndarray (N+1, N)       — 当前速度 u 分量（网格边界包含）
    v: ndarray (N, N+1)       — 当前速度 v 分量
    f_minus_bp: tuple (f_u, f_v)
        f_u shape (N+1, N), f_v shape (N, N+1)
        这是 RHS = F - B P_k（pressure 已被吸收到 RHS）
    h: float                  — 网格步长
    bcs: dict                 — 边界条件字典，保持跟你现有代码相同语义
    nu1, nu2: int             — pre-/post- smooth 次数
    min_N: int               — 最粗网格阈值（与你的 v_cycle_recursive 保持一致）
    
    返回
    ----
    u, v  — 更新后的速度场（近似解）
    """

    # shapes and sizes
    f_u, f_v = f_minus_bp
    N = f_u.shape[0] - 1         # 注意：f_u.shape == (N+1, N) 这里 N 表示单元格个数
    # consistency check (optional)
    # f_v should be (N, N+1)

    # assert f_v.shape[0] == N and f_v.shape[1] == N+1, "f_v shape mismatch"

    # --- Coarse grid direct-ish solve (base case) ---
    if N <= min_N:
        # 在最粗网格上，使用较多次的 DGS velocity-step 迭代作为近似解器
        # 这里用 100 次与您原来 v_cycle_recursive 的 coarse solve 保持一致
        for _ in range(100):
            u, v = dgs_velocity_step(u, v, f_minus_bp, h, bcs)
        u, v = enforce_dirichlet_bc(u, v)
        return u, v

    # -------------------------
    # 1) Pre-smooth (nu1 次)
    # -------------------------
    for _ in range(nu1):
        u, v = dgs_velocity_step(u, v, f_minus_bp, h, bcs)
    u, v = enforce_dirichlet_bc(u, v)

    # -------------------------
    # 2) Compute residual r = RHS - A*u
    #    We use compute_residuals_stokes with pressure = 0
    #    because compute_residuals_stokes returns r_u = f - (A u + B p).
    #    Passing p=0 and f=f_minus_bp yields r_u = f_minus_bp - A u (期望的残差).
    # -------------------------
    p_zero = np.zeros((N, N))  # pressure not used here, shape as usual
    r_u, r_v, r_div = compute_residuals_stokes(u, v, p_zero, f_u, f_v, h, bcs)
    # r_div is ignored for velocity-only transfer, but restrict_residuals signature expects it.
    # If restrict_residuals expects a "real" divergence, pass zeros; we pass r_div (should be consistent).
    # In your original v_cycle_recursive you passed r_u, r_v, r_div_real into restrict_residuals.

    # -------------------------
    # 3) Restrict residuals to coarse grid
    #    restrict_residuals returns (f_c_inner, g_c_inner, g_div_c)
    #    where f_c_inner has shape (Nc-1, Nc) (inner cells) as in your code
    # -------------------------
    f_c_inner, g_c_inner, g_div_c = restrict_residuals(r_u, r_v, r_div)

    # prepare coarse arrays (same layout as in v_cycle_recursive)
    Nc = N // 2
    f_c = np.zeros((Nc + 1, Nc))
    f_c[1:-1, :] = f_c_inner     # place inner restricted RHS
    g_c = np.zeros((Nc, Nc + 1))
    g_c[:, 1:-1] = g_c_inner

    # coarse-level boundary conditions: zero (error equations have zero BCs)
    bcs_c = {'b': np.zeros(Nc - 1), 't': np.zeros(Nc - 1),
             'l': np.zeros(Nc - 1), 'r': np.zeros(Nc - 1)}

    # initialize coarse corrections (zero initial guess)
    u_c = np.zeros((Nc + 1, Nc))
    v_c = np.zeros((Nc, Nc + 1))

    # -------------------------
    # 4) Recurse: solve A_c e = r_c approximately
    # -------------------------
    e_u_c, e_v_c = v_cycle_velocity(u_c, v_c, (f_c, g_c), 2*h, bcs_c,
                                   nu1=nu1, nu2=nu2, min_N=min_N)

    # -------------------------
    # 5) Prolongate correction and apply to fine grid
    # -------------------------
    # prolongate_error returns (corr_u, corr_v, corr_p)
    corr_u, corr_v, _ = prolongate_error(e_u_c, e_v_c, np.zeros((Nc, Nc)))
    # Add corrections (shapes must match)
    u += corr_u
    v += corr_v

    # clamp boundaries, avoid introducing boundary noise
    u, v = enforce_dirichlet_bc(u, v)

    # -------------------------
    # 6) Post-smooth (nu2 次)
    # -------------------------
    for _ in range(nu2):
        u, v = dgs_velocity_step(u, v, f_minus_bp, h, bcs)

    u, v = enforce_dirichlet_bc(u, v)

    return u, v
