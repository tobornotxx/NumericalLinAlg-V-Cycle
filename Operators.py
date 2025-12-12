import numpy as np


def apply_laplacian_u(u, h):
    '''
    Apply Laplacian operator to v
    Shape of u: (N+1, N)
    使用差分格式计算 Laplacian
    返回 -Laplacian(u)， shape (N-1, N)
    '''
    # x的内部点，二阶差分格式：
    dxxu = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / h**2
    u_internal = u[1:-1, :] # x方向内部，去掉边界
    dyyu_internal = (u_internal[:, 2:] - 2 * u_internal[:, 1:-1] + u_internal[:, :-2]) / h**2

    # 边界处理，底部，j=0,离散形式退化为 (u_{i,1} - u_{i,0})/h^2，一阶近似
    # 边界条件满足的方程是V-Cycle Slides 35-36页给出的
    dyyu_bottom = (u_internal[:, 1] - u_internal[:, 0]) / h**2

    # 顶部，j=N-1,离散形式退化为 -(u_{i,N-1} - u_{i,N-2})/h^2
    dyyu_top = (u_internal[:, -2] - u_internal[:, -1]) / h**2

    # 分别用内部和边界值填充组装
    dyyu = np.zeros_like(dxxu)
    dyyu[:, 1:-1] = dyyu_internal
    dyyu[:, 0] = dyyu_bottom
    dyyu[:, -1] = dyyu_top

    # 注意：这里返回的是 -Laplacian(u)
    return -(dxxu + dyyu)

def apply_laplacian_v(v,h):
    '''
    Apply Laplacian operator to v
    Shape of v: (N, N+1)
    使用差分格式计算 Laplacian
    返回 -Laplacian(v)， shape (N, N-1)
    '''
    # y可以直接差分计算
    dyyv = (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) / h**2

    # y的内部点：
    v_internal = v[:, 1:-1] # y方向内部
    dxxv_internal = (v_internal[2:, :] - 2 * v_internal[1:-1, :] + v_internal[:-2, :]) / h**2

    # 左侧，i=0,离散形式退化为 (v_{1,j} - v_{0,j})/h^2
    dxxv_left = (v_internal[1, :] - v_internal[0, :]) / h**2

    # 右侧，i=N-1,离散形式退化为 -(v_{N-1,j} - v_{N-2,j})/h^2
    dxxv_right = (v_internal[-2, :] - v_internal[-1, :]) / h**2

    dxxv = np.zeros_like(dyyv)
    dxxv[1:-1, :] = dxxv_internal
    dxxv[0, :] = dxxv_left
    dxxv[-1, :] = dxxv_right

    # 注意：这里返回的是 -Laplacian(v)
    return -(dxxv + dyyv)

def apply_gradient_p(p,h):
    '''
    计算压力p的梯度
    输入：p (N, N)
    输出：
        gp_x: (N-1, N)对应u位置的梯度
        gp_y: (N, N-1)对应v位置的梯度   
    '''
    # dp/dx: (p[i,j] - p[i-1,j])/h
    # 在 u_{i,j} 的位置，右边是 p[i,j] (对应 numpy index i)，左边是 p[i-1,j] (对应 numpy index i-1)
    gp_x = (p[1:, :] - p[:-1, :]) / h

    # dp/dy: (p[i,j] - p[i,j-1])/h
    # 在 v_{i,j} 的位置，右边是 p[i,j] (对应 numpy index j)，左边是 p[i,j-1] (对应 numpy index j-1)
    gp_y = (p[:, 1:] - p[:, :-1]) / h
    return gp_x, gp_y

def apply_divergence_uv(u, v, h):
    '''
    计算速度场的散度 div (uv)
    输入：
        u: (N+1, N)
        v: (N, N+1)
    输出：
        div_uv: (N, N)
    '''
    # div (uv) = dux/dx + dvy/dy
    # dux/dx = (u[i+1,j] - u[i,j])/h

    dux_dx = (u[1:, :] - u[:-1, :]) / h

    # dvy/dy = (v[i,j+1] - v[i,j])/h

    dvy_dy = (v[:, 1:] - v[:, :-1]) / h

    return dux_dx + dvy_dy

def compute_residuals_stokes(u, v, p, f, g, h, bcs):
    '''
    计算 Stokes 方程的残差
    r_u = f - (-Laplacian(u) + grad(p))
    r_v = g - (-Laplacian(v) + grad(p))
    r_div = 0 - div(uv)
    注意：Neumann边界条件产生的b/h, t/h等在这里作为RHS修正项引入
    输入：
        u: (N+1, N)
        v: (N, N+1)
        p: (N, N)
        f: (N+1, N)
        g: (N, N+1)
        h: float
        bcs: Dict, 包含b, t, l, r四个一维数组
        b, t长N, X方向
        l, r长N, Y方向
    输出：
        r_u: (N-1, N)
        r_v: (N, N-1)
        r_div: (N, N)
    '''
    N = p.shape[0]

    # 首先计算AU+BP
    Lu = apply_laplacian_u(u, h) #(N-1, N)
    Lv = apply_laplacian_v(v, h) #(N, N-1)
    gp_x, gp_y = apply_gradient_p(p, h) #(N-1, N), (N, N-1)
    div = apply_divergence_uv(u, v, h) #(N, N)

    # 准备RHS(f, g)并取内部
    rhs_u = f[1:-1, :].copy()
    rhs_v = g[:, 1:-1].copy()

    # 加入边界条件
    # Reference: Slides V-Cycle P35 Equation 2, 3; P36 Equation 2, 3

    rhs_u[:, 0] += bcs['b'] / h
    rhs_u[:, -1] += bcs['t'] / h
    rhs_v[0, :] += bcs['l'] / h
    rhs_v[-1, :] += bcs['r'] / h

    # 计算残差
    # r_u = rhs_u - (Lu + gp_x)
    r_u = rhs_u - (Lu + gp_x)

    # r_v = rhs_v - (Lv + gp_y)
    r_v = rhs_v - (Lv + gp_y)

    # r_div = 0 - div
    r_div = - div

    return r_u, r_v, r_div