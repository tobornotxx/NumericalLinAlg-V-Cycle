# 实现第一问的DGS磨光，分两步，先更新速度（smooth_momentum），然后更新压力(apply_distributive_correction)
import numpy as np
import matplotlib.pyplot as plt
import time
from Operators import apply_divergence_uv, apply_laplacian_u, apply_laplacian_v

def smooth_momentum(u, v, p, f, g, h, bcs):
    N = p.shape[0]
    h2 = h * h
    
    # --- Update u ---
    # 填充u的上下两行，填充值为0，保证网格尺寸兼容
    u_pad = np.pad(u, ((0,0), (1,1)), mode='constant')
    
    dp_dx = (p[1:, :] - p[:-1, :]) * h #这里单独计算是为了防止数值溢出

    rhs_u = f[1:-1, :] * h2
    
    nx, ny_pad = u_pad[1:-1, :].shape
    ix, iy = np.indices((nx, ny_pad))
    
    # 红黑Gauss-Seidel迭代，交替更新，当更新红点时，黑点值fix；反之亦然。
    for parity in [0, 1]:
        # 用边界条件值更新边界点（分别为bottom和top）
        if 'b' in bcs:
            u_pad[1:-1, 0] = u_pad[1:-1, 1] + h * bcs['b']
        if 't' in bcs:
            u_pad[1:-1, -1] = u_pad[1:-1, -2] + h * bcs['t']

        # 计算红黑掩码
        mask = ((ix + 1) + (iy - 1)) % 2 == parity

        # 锁定边界点，防止更新
        mask[:, 0] = False
        mask[:, -1] = False
        
        u_center = u_pad[1:-1, 1:-1]
        
        # 使用切片的方法计算避免循环，对于内点，4u = u_up + u_down + u_left + u_right - dp/dx * h + rhs * h^2
        u_up    = u_pad[1:-1, 2:]
        u_down  = u_pad[1:-1, 0:-2]
        u_left  = u_pad[0:-2, 1:-1]
        u_right = u_pad[2:, 1:-1]
        
        mask_center = mask[:, 1:-1]
        
        sum_neighbors = u_up + u_down + u_left + u_right
        val_new = (sum_neighbors - dp_dx[:, :] + rhs_u) / 4.0

        u_center[mask_center] = val_new[mask_center]
        u_pad[1:-1, 1:-1] = u_center

    u[:] = u_pad[:, 1:-1] # 去掉新加的边界

    # --- Update v ---
    v_pad = np.pad(v, ((1,1), (0,0)), mode='constant')
    
    dp_dy = (p[:, 1:] - p[:, :-1]) * h
    rhs_v = g[:, 1:-1] * h2
    
    nx_pad, ny = v_pad[:, 1:-1].shape
    ix, iy = np.indices((nx_pad, ny))

    for parity in [0, 1]:
        if 'l' in bcs:
            v_pad[0, 1:-1] = v_pad[1, 1:-1] + h * bcs['l']
        if 'r' in bcs:
            v_pad[-1, 1:-1] = v_pad[-2, 1:-1] + h * bcs['r']
            
        mask = ((ix - 1) + (iy + 1)) % 2 == parity
        mask[0, :] = False
        mask[-1, :] = False


        
        v_center = v_pad[1:-1, 1:-1]
        mask_center = mask[1:-1, :]
        
        v_up    = v_pad[1:-1, 2:]
        v_down  = v_pad[1:-1, 0:-2]
        v_left  = v_pad[0:-2, 1:-1]
        v_right = v_pad[2:, 1:-1]

        sum_neighbors = v_up + v_down + v_left + v_right

        val_new = (sum_neighbors - dp_dy[:, :] + rhs_v) / 4.0
        
        v_center[mask_center] = val_new[mask_center]
        v_pad[1:-1, 1:-1] = v_center
        
    v[:] = v_pad[1:-1, :]

    return u, v


def apply_distributive_correction(u, v, p, h, g_div=None):
    """
    压力分布式更新
    防止了负索引导致的越界等情况
    g_div: 散度残差
    """
    N = p.shape[0]
    ix, iy = np.indices((N, N))

    # 同理红黑 Gauss-Seidel
    for parity in [0, 1]:
        # 计算散度
        div = apply_divergence_uv(u, v, h)
        r = (g_div - div) if (g_div is not None) else -div


        mask_all = (ix + iy) % 2 == parity

        # ============================================================
        # 1. 内点，四个邻居
        # ============================================================
        mask_int = np.zeros_like(mask_all, dtype=bool)
        mask_int[1:-1, 1:-1] = True
        # 同样不允许修改边界点
        mask = mask_all & mask_int

        if np.any(mask):
            rows, cols = np.where(mask)
            vals = r[rows, cols]
            delta = vals * h / 4.0

            # 内部方程，delta = div residual/4
            # u左侧-delta，右侧+delta。v更新规则类似

            u[rows, cols]   -= delta
            u[rows+1, cols] += delta
            v[rows, cols]   -= delta
            v[rows, cols+1] += delta

            # p中心点 + res, 四周 - res / 4
            p[rows, cols]   += vals
            p[rows-1, cols] -= vals/4
            p[rows+1, cols] -= vals/4
            p[rows, cols-1] -= vals/4
            p[rows, cols+1] -= vals/4

        # ============================================================
        # 2. 上边界  j = N-1
        # ============================================================
        mask_top = np.zeros_like(mask_all, dtype=bool)
        mask_top[1:-1, N-1] = True
        mask = mask_all & mask_top

        if np.any(mask):
            rows, cols = np.where(mask)
            vals = r[rows, cols]
            delta = vals * h / 3.0

            u[rows, cols]   -= delta
            u[rows+1, cols] += delta
            v[rows, cols]   -= delta

            for rr, cc, val in zip(rows, cols, vals):
                p[rr, cc] += val
                p[rr-1, cc] -= val/3
                p[rr+1, cc] -= val/3
                p[rr, cc-1] -= val/3  # 没有上方邻居

        # ============================================================
        # 3. 下边界 j = 0
        # ============================================================
        mask_bot = np.zeros_like(mask_all, dtype=bool)
        mask_bot[1:-1, 0] = True
        mask = mask_all & mask_bot

        if np.any(mask):
            rows, cols = np.where(mask)
            vals = r[rows, cols]
            delta = vals * h / 3.0

            u[rows, cols]   -= delta
            u[rows+1, cols] += delta
            v[rows, cols+1] += delta

            for rr, cc, val in zip(rows, cols, vals):
                p[rr, cc] += val
                p[rr-1, cc] -= val/3
                p[rr+1, cc] -= val/3
                p[rr, cc+1] -= val/3

        # ============================================================
        # 4. 左边界 i = 0
        # ============================================================
        mask_left = np.zeros_like(mask_all, dtype=bool)
        mask_left[0, 1:-1] = True
        mask = mask_all & mask_left

        if np.any(mask):
            rows, cols = np.where(mask)
            vals = r[rows, cols]
            delta = vals * h / 3.0

            u[rows+1, cols] += delta
            v[rows, cols]   -= delta
            v[rows, cols+1] += delta

            for rr, cc, val in zip(rows, cols, vals):
                p[rr, cc] += val
                p[rr+1, cc] -= val/3
                p[rr, cc-1] -= val/3
                p[rr, cc+1] -= val/3

        # ============================================================
        # 5. 右边界 i = N-1
        # ============================================================
        mask_right = np.zeros_like(mask_all, dtype=bool)
        mask_right[N-1, 1:-1] = True
        mask = mask_all & mask_right

        if np.any(mask):
            rows, cols = np.where(mask)
            vals = r[rows, cols]
            delta = vals * h / 3.0

            u[rows, cols] -= delta
            v[rows, cols] -= delta
            v[rows, cols+1] += delta

            for rr, cc, val in zip(rows, cols, vals):
                p[rr, cc] += val
                p[rr-1, cc] -= val/3
                p[rr, cc-1] -= val/3
                p[rr, cc+1] -= val/3

        # ============================================================
        # 6. 角落点更新
        #    只有两个邻居点，权重为 h/2
        # ============================================================

        # (0,0)
        if mask_all[0,0]:
            r0 = r[0,0]; d = r0*h/2
            if u.shape[0] > 1: u[1,0] += d
            if v.shape[1] > 1: v[0,1] += d
            p[0,0] += r0
            if N > 1:
                p[1,0] -= r0/2
                p[0,1] -= r0/2

        # (N-1,0)
        if mask_all[N-1,0]:
            r0 = r[N-1,0]; d = r0*h/2
            u[N-2,0] -= d
            v[N-1,1] += d
            p[N-1,0] += r0
            p[N-2,0] -= r0/2
            p[N-1,1] -= r0/2

        # (0,N-1)
        if mask_all[0,N-1]:
            r0 = r[0,N-1]; d = r0*h/2
            u[1,N-1] += d
            v[0,N-1] -= d
            p[0,N-1] += r0
            p[1,N-1] -= r0/2
            p[0,N-2] -= r0/2

        # (N-1,N-1)
        if mask_all[N-1,N-1]:
            r0 = r[N-1,N-1]; d = r0*h/2
            u[N-2,N-1] -= d
            v[N-1,N-1] -= d
            p[N-1,N-1] += r0
            p[N-2,N-1] -= r0/2
            p[N-1,N-2] -= r0/2


    return u, v, p


def dgs_step(u, v, p, f, g, h, bcs, g_div=None):
    # 1. 速度迭代
    u, v = smooth_momentum(u, v, p, f, g, h, bcs)

    # 2. 压力更新
    u, v, p = apply_distributive_correction(u, v, p, h, g_div)
 
    # 后处理，中心化p，处理边界问题
    p -= np.mean(p)
    u[0, :] = 0.0
    u[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0

    return u, v, p



# 以下是测试代码，检验DGS是否能正确迭代到近似收敛，直接运行本文件可以看到结果和图片，然而，与最终结果无关。


def make_forced_problem(N):
    # grid spacing
    h = 1.0 / N
    # face and center coordinates
    xf = np.linspace(0.0, 1.0, N+1)                # x for u (faces)
    ycf = (np.arange(N) + 0.5) / N                 # y for u (cell centers vertically)
    Xf_u, Yf_u = np.meshgrid(xf, ycf, indexing='ij')  # shapes (N+1, N)

    xcf = (np.arange(N) + 0.5) / N                 # x for v (cell centers horizontally)
    yf = np.linspace(0.0, 1.0, N+1)                # y for v (faces)
    Xf_v, Yf_v = np.meshgrid(xcf, yf, indexing='ij')  # shapes (N, N+1)

    # cell centers for p
    xc = (np.arange(N) + 0.5) / N
    yc = (np.arange(N) + 0.5) / N
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')    # (N,N)

    # exact analytic solution evaluated on staggered grid (for comparisons)
    u_exact_faces = (1.0 - np.cos(2*np.pi*Xf_u)) * np.sin(2*np.pi*Yf_u)     # shape (N+1, N)
    v_exact_faces = - (1.0 - np.cos(2*np.pi*Yf_v)) * np.sin(2*np.pi*Xf_v)   # shape (N, N+1)
    p_exact_centers = (Xc**3) / 3.0 - 1.0/12.0

    # build f on u-face locations: f = -Delta u + dp/dx
    # analytic expression (derived): -Delta u = -4*pi^2*(2*cos(2pi x) - 1) * sin(2pi y)
    f = -4.0 * np.pi**2 * (2.0 * np.cos(2.0*np.pi*Xf_u) - 1.0) * np.sin(2.0*np.pi*Yf_u)
    # add dp/dx = x^2 (derivative of p_exact = x^2)
    f += Xf_u**2

    # build g on v-face locations: g = -Delta v + dp/dy (dp/dy = 0 here)
    g = 4.0 * np.pi**2 * (2.0 * np.cos(2.0*np.pi*Yf_v) - 1.0) * np.sin(2.0*np.pi*Xf_v)

    return {
        'u_exact_faces': u_exact_faces,
        'v_exact_faces': v_exact_faces,
        'p_exact': p_exact_centers,
        'f': f,
        'g': g,
        'Xc': Xc, 'Yc': Yc
    }

def run_dgs_manufactured(N=64, max_iter=800, omega=0.9, n_smooth=1, verbose=True):
    h = 1.0 / N
    prob = make_forced_problem(N)
    f = prob['f']; g = prob['g']
    u_exact_faces = prob['u_exact_faces']; v_exact_faces = prob['v_exact_faces']
    p_exact = prob['p_exact']

    # initial guess zeros
    u = np.zeros_like(u_exact_faces)
    v = np.zeros_like(v_exact_faces)
    p = np.zeros_like(p_exact)

    bcs = {'b': 0.0, 't': 0.0, 'l': 0.0, 'r': 0.0}

    divergences = []
    errors_u = []; errors_v = []
    t0 = time.time()
    for it in range(1, max_iter+1):
        u, v, p = dgs_step(u, v, p, f, g, h, bcs, g_div=None)

        # divergence at cell centers
        div = (u[1:, :] - u[:-1, :]) / h + (v[:, 1:] - v[:, :-1]) / h
        divergences.append(np.linalg.norm(div.ravel()))

        # compute error against exact (interpolate to cell centers)
        u_c = 0.5 * (u[:-1, :] + u[1:, :])    # shape (N,N) - numeric u at cell centers
        v_c = 0.5 * (v[:, :-1] + v[:, 1:])    # shape (N,N) - numeric v at cell centers
        # exact at cell centers:
        xc = (np.arange(N) + 0.5) / N
        yc = (np.arange(N) + 0.5) / N
        Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
        u_exact_centers = (1.0 - np.cos(2*np.pi*Xc)) * np.sin(2*np.pi*Yc)
        v_exact_centers = - (1.0 - np.cos(2*np.pi*Yc)) * np.sin(2*np.pi*Xc)

        errors_u.append(np.linalg.norm((u_c - u_exact_centers).ravel()))
        errors_v.append(np.linalg.norm((v_c - v_exact_centers).ravel()))

        if verbose and (it % 50 == 0 or it == max_iter-1):
            print(f"iter {it:4d}, div L2 = {divergences[-1]:.3e}, err_u = {errors_u[-1]:.3e}, err_v = {errors_v[-1]:.3e}")

    # plotting
    iters = np.arange(len(divergences))
    plt.figure(figsize=(6,4))
    plt.semilogy(iters, divergences, label='div L2')
    plt.semilogy(iters, errors_u, label='err u (L2)')
    plt.semilogy(iters, errors_v, label='err v (L2)')
    plt.legend(); plt.grid(True)
    plt.title('DGS convergence (manufactured sol)')
    plt.xlabel('iteration'); plt.tight_layout()

    # final fields: plot u and v numeric vs exact (cell centers)
    u_c = 0.5 * (u[:-1, :] + u[1:, :])
    v_c = 0.5 * (v[:, :-1] + v[:, 1:])
    xc = (np.arange(N) + 0.5) / N
    yc = (np.arange(N) + 0.5) / N
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
    u_exact_centers = (1.0 - np.cos(2*np.pi*Xc)) * np.sin(2*np.pi*Yc)
    v_exact_centers = - (1.0 - np.cos(2*np.pi*Yc)) * np.sin(2*np.pi*Xc)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title('u numeric (cell centers)')
    im = plt.imshow(u_c.T, origin='lower', extent=[0,1,0,1])
    plt.colorbar(im)
    plt.subplot(1,2,2)
    plt.title('u exact (cell centers)')
    im2 = plt.imshow(u_exact_centers.T, origin='lower', extent=[0,1,0,1])
    plt.colorbar(im2)
    plt.tight_layout()

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title('v numeric (cell centers)')
    im = plt.imshow(v_c.T, origin='lower', extent=[0,1,0,1])
    plt.colorbar(im)
    plt.subplot(1,2,2)
    plt.title('v exact (cell centers)')
    im2 = plt.imshow(v_exact_centers.T, origin='lower', extent=[0,1,0,1])
    plt.colorbar(im2)
    plt.tight_layout()

    plt.show()

    return {
        'u': u, 'v': v, 'p': p,
        'u_exact_faces': u_exact_faces, 'v_exact_faces': v_exact_faces, 'p_exact': p_exact,
        'divergences': np.array(divergences), 'errors_u': np.array(errors_u), 'errors_v': np.array(errors_v)
    }

if __name__ == '__main__':
    res = run_dgs_manufactured(N=128, max_iter=1000)