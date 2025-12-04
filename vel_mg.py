# vel_mg.py
import numpy as np
from Smoother import smooth_momentum    # 你的红黑 Gauss-Seidel smoother（只做 momentum 部分）
from Operators import apply_laplacian_u, apply_laplacian_v
from Transfer import restrict_residuals, prolongate_error

def vel_smoother(u, v, f_rhs, g_rhs, h, bcs, iterations=2):
    """
    Velocity-only smoother using your existing smooth_momentum (red-black GS on momentum).
    We call smooth_momentum which expects p (pressure). For velocity-only solve we pass p=0.
    f_rhs and g_rhs are full-grid arrays with interior filled (same shapes as u and v).
    """
    # create zero pressure (shape (N,N))
    N = f_rhs.shape[1]
    p_zero = np.zeros((N, N))
    for _ in range(iterations):
        # smooth_momentum updates u and v using -Laplacian + grad(p) and f,g; p==0 => pure velocity smoothing
        u, v = smooth_momentum(u, v, p_zero, f_rhs, g_rhs, h, bcs)
        # enforce Dirichlet velocity faces remain same (they should be set externally)
        # (we don't overwrite Dirichlet values here; assume caller initialized them)
        u[0, :] = u[0, :]
        u[-1, :] = u[-1, :]
        v[:, 0] = v[:, 0]
        v[:, -1] = v[:, -1]
    return u, v

def vel_compute_residuals(u, v, f_rhs, g_rhs, h):
    """
    Compute velocity-only residuals:
      r_u = interior_rhs - (Laplacian_u(u))
      r_v = interior_rhs - (Laplacian_v(v))
    Shapes:
      u: (N+1, N) -> r_u: (N-1, N)
      v: (N, N+1) -> r_v: (N, N-1)
    """
    Lu = apply_laplacian_u(u, h)   # (N-1, N)
    Lv = apply_laplacian_v(v, h)   # (N, N-1)

    rhs_u = f_rhs[1:-1, :].copy()
    rhs_v = g_rhs[:, 1:-1].copy()

    r_u = rhs_u - Lu
    r_v = rhs_v - Lv
    return r_u, r_v

def vel_v_cycle(u, v, f_rhs, g_rhs, h, bcs, nu1=2, nu2=2, min_N=4):
    """
    Velocity-only V-cycle using your Transfer.restrict_residuals and Transfer.prolongate_error.
    Returns updated (u,v) and a dummy p (for compatibility).
    """
    # N is number of cells in each direction for pressure; f_rhs shape (N+1, N)
    N = f_rhs.shape[1]
    # base coarse solve
    if N <= min_N:
        # apply a bunch of smooths on coarse to act as direct solve
        for _ in range(100):
            u, v = vel_smoother(u, v, f_rhs, g_rhs, h, bcs, iterations=4)
        return u, v, np.zeros((N, N))

    # pre-smooth
    u, v = vel_smoother(u, v, f_rhs, g_rhs, h, bcs, iterations=nu1)

    # compute residuals (interior)
    r_u, r_v = vel_compute_residuals(u, v, f_rhs, g_rhs, h)
    # r_div not used for velocity-only restrict, but restrict_residuals expects it; pass zeros
    r_div_dummy = np.zeros((N, N))

    # use your provided transfer operator
    rc_u, rc_v, rc_div = restrict_residuals(r_u, r_v, r_div_dummy)

    # build coarse RHS full-shaped arrays
    Nc = N // 2
    f_c = np.zeros((Nc + 1, Nc))
    g_c = np.zeros((Nc, Nc + 1))

    # put restricted residuals into coarse RHS interior
    # rc_u shape expected (Nc-1, Nc), rc_v shape (Nc, Nc-1)
    f_c[1:-1, :] = rc_u
    g_c[:, 1:-1] = rc_v

    # initialize coarse corrections
    u_c = np.zeros((Nc + 1, Nc))
    v_c = np.zeros((Nc, Nc + 1))

    # coarse homogeneous BCs (error equation)
    bcs_c = {'b': np.zeros(Nc - 1), 't': np.zeros(Nc - 1), 'l': np.zeros(Nc - 1), 'r': np.zeros(Nc - 1)}

    # recursion
    e_u_c, e_v_c, _ = vel_v_cycle(u_c, v_c, f_c, g_c, 2*h, bcs_c, nu1, nu2, min_N)

    # prolongate using your provided prolongate_error (returns full-size corrections)
    corr_u, corr_v, _ = prolongate_error(e_u_c, e_v_c, np.zeros((Nc, Nc)))

    # correct fine-grid
    u += corr_u
    v += corr_v

    # post-smooth
    u, v = vel_smoother(u, v, f_rhs, g_rhs, h, bcs, iterations=nu2)

    return u, v, np.zeros((N, N))
