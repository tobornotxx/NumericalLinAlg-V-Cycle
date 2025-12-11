import numpy as np

def create_grids(N):
    '''
    创建网格
    输入：
        N: int, 网格点数
    输出：
        xp: (N, N)
        yp: (N, N)
        xu: (N+1, N)
        yu: (N+1, N)
        xv: (N, N+1)
        yv: (N, N+1)
    '''
    h = 1.0/N
    xp = (np.arange(1, N+1) - 0.5) * h
    yp = (np.arange(1, N+1) - 0.5) * h

    xu = np.arange(0, N+1) * h
    yu = (np.arange(1, N+1) - 0.5) * h

    xv = (np.arange(1, N+1) - 0.5) * h
    yv = np.arange(0, N+1) * h

    # 用ij方式编号可以保证x方向对应第一个维度，y方向对应第二个维度
    xp, yp = np.meshgrid(xp, yp, indexing='ij')
    xu, yu = np.meshgrid(xu, yu, indexing='ij')
    xv, yv = np.meshgrid(xv, yv, indexing='ij')

    return xp, yp, xu, yu, xv, yv

def get_exact_solution(N):
    '''
    获取精确解
    输入：
        N: int, 网格点数
    输出：
        u_exact: (N+1, N)
        v_exact: (N, N+1)
        p_exact: (N, N)
        f: (N+1, N)
        g: (N, N+1)
    '''
    Xp, Yp, Xu, Yu, Xv, Yv = create_grids(N)

    # Get u, v, p
    u_exact = (1 - np.cos(2 * np.pi * Xu)) * np.sin(2 * np.pi * Yu)
    v_exact =   -(1 - np.cos(2 * np.pi * Yv)) * np.sin(2 * np.pi * Xv)
    p_exact = Xp ** 3 / 3.0 - 1.0 / 12.0

    # 强制消去平均值
    p_exact = p_exact - np.mean(p_exact)

    # 计算外力项f, g
    f = - 4 * (np.pi ** 2) * (2 * np.cos(2 * np.pi * Xu) - 1) * np.sin(2 * np.pi * Yu) + Xu ** 2
    g = 4 * (np.pi ** 2) * (2 * np.cos(2 * np.pi * Yv) - 1) * np.sin(2 * np.pi * Xv) 
    return u_exact, v_exact, p_exact, f, g



if __name__ == "__main__":
    print(create_grids(4))
    print(get_exact_solution(4))