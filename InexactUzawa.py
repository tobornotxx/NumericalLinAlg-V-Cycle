import numpy as np
from Operators import compute_residuals_stokes
from Transfer import restrict_residuals, prolongate_error

def smooth_velocity(u, v, f_minus_bp, h, bcs):
    """
    DGS smoother for the velocity subproblem AU = F - B*P_k
    Only updates u, v; pressure is fixed.
    
    Parameters:
        u: (N+1, N) velocity x-component
        v: (N, N+1) velocity y-component
        f_minus_bp: tuple (f_u, f_v) right-hand side
        h: grid spacing
        bcs: boundary conditions dictionary
    """
    f_u, f_v = f_minus_bp
    N = f_u.shape[0]
    h2 = h*h

    # --- Update u ---
    u_pad = np.pad(u, ((0,0),(1,1)), mode='constant')
    dp_dx_dummy = np.zeros_like(u)  # Pressure term absorbed in RHS
    rhs_u = f_u[1:-1, :] * h2

    nx, ny_pad = u_pad[1:-1,:].shape
    ix, iy = np.indices((nx, ny_pad))

    for parity in [0,1]:
        if 'b' in bcs: u_pad[1:-1,0] = u_pad[1:-1,1] + h*bcs['b']
        if 't' in bcs: u_pad[1:-1,-1] = u_pad[1:-1,-2] + h*bcs['t']

        mask = ((ix+1) + (iy-1)) % 2 == parity
        mask[:,0] = False
        mask[:,-1] = False

        u_center = u_pad[1:-1,1:-1]
        u_up    = u_pad[1:-1,2:]
        u_down  = u_pad[1:-1,0:-2]
        u_left  = u_pad[0:-2,1:-1]
        u_right = u_pad[2:,1:-1]

        mask_center = mask[:,1:-1]

        sum_neighbors = u_up + u_down + u_left + u_right
        val_new = (sum_neighbors + rhs_u) / 4.0

        u_center[mask_center] = val_new[mask_center]
        u_pad[1:-1,1:-1] = u_center

    u[:] = u_pad[:,1:-1]

    # --- Update v ---
    v_pad = np.pad(v, ((1,1),(0,0)), mode='constant')
    dp_dy_dummy = np.zeros_like(v)
    rhs_v = f_v[:, 1:-1] * h2

    nx_pad, ny = v_pad[:,1:-1].shape
    ix, iy = np.indices((nx_pad, ny))

    for parity in [0,1]:
        if 'l' in bcs: v_pad[0,1:-1] = v_pad[1,1:-1] + h*bcs['l']
        if 'r' in bcs: v_pad[-1,1:-1] = v_pad[-2,1:-1] + h*bcs['r']

        mask = ((ix-1) + (iy+1)) % 2 == parity
        mask[0,:] = False
        mask[-1,:] = False

        v_center = v_pad[1:-1,1:-1]
        mask_center = mask[1:-1,:]

        v_up    = v_pad[1:-1,2:]
        v_down  = v_pad[1:-1,0:-2]
        v_left  = v_pad[0:-2,1:-1]
        v_right = v_pad[2:,1:-1]

        sum_neighbors = v_up + v_down + v_left + v_right
        val_new = (sum_neighbors + rhs_v) / 4.0

        v_center[mask_center] = val_new[mask_center]
        v_pad[1:-1,1:-1] = v_center

    v[:] = v_pad[1:-1,:]

    return u, v


def apply_distributive_correction_velocity(u, v, h):
    """
    Distributive correction for velocity subproblem only.
    Does NOT update pressure.
    """
    N = u.shape[0]-1
    ix, iy = np.indices((N,N))

    for parity in [0,1]:
        # Compute divergence
        div = (u[1:, :] - u[:-1,:])/h + (v[:,1:] - v[:,:-1])/h
        r = -div

        mask_all = (ix + iy) % 2 == parity

        # Internal points
        mask_int = np.zeros_like(mask_all, dtype=bool)
        mask_int[1:-1,1:-1] = True
        mask = mask_all & mask_int

        if np.any(mask):
            rows, cols = np.where(mask)
            vals = r[rows,cols]
            delta = vals*h/4.0
            u[rows,cols]   -= delta
            u[rows+1,cols] += delta
            v[rows,cols]   -= delta
            v[rows,cols+1] += delta

    return u, v


def dgs_velocity_step(u, v, f_minus_bp, h, bcs):
    """
    One DGS iteration for AU = F - B P_k
    """
    u, v = smooth_velocity(u, v, f_minus_bp, h, bcs)
    u, v = apply_distributive_correction_velocity(u, v, h)

    # Enforce Dirichlet BCs
    u[0,:] = 0.0
    u[-1,:] = 0.0
    v[:,0] = 0.0
    v[:,-1] = 0.0

    return u, v


# 需要事先定义 dgs_velocity_step(u,v,f_minus_bp,h,bcs)
#   它应当实现单步 DGS 磨光（只更新 u,v，不更新 p）
#   并且在最后强制 Dirichlet 边界（u 左右为0，v 上下为0）。

def enforce_dirichlet_bc_uv(u, v):
    """和你现有代码一致的 Dirichlet 强制（u左右0, v上下0）"""
    u[0, :] = 0.0
    u[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    return u, v


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
        u, v = enforce_dirichlet_bc_uv(u, v)
        return u, v

    # -------------------------
    # 1) Pre-smooth (nu1 次)
    # -------------------------
    for _ in range(nu1):
        u, v = dgs_velocity_step(u, v, f_minus_bp, h, bcs)
    u, v = enforce_dirichlet_bc_uv(u, v)

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
    u, v = enforce_dirichlet_bc_uv(u, v)

    # -------------------------
    # 6) Post-smooth (nu2 次)
    # -------------------------
    for _ in range(nu2):
        u, v = dgs_velocity_step(u, v, f_minus_bp, h, bcs)

    u, v = enforce_dirichlet_bc_uv(u, v)

    return u, v
