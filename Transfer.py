import numpy as np
# 参考往年大作业的作业文档中的限制-提升算子图例。

def restrict_residuals(r_u, r_v, r_div):
    '''
    限制算子 (Fine -> Coarse)
    U, V: Nearest 2 (/4) + Next Nearest 4 (/8). 
    P (Div): Nearest 4 (/4). (Average)
    '''
    N = r_div.shape[0]
    Nc = N // 2
    
    # --- 1. Restrict r_u (N-1, N) -> (Nc-1, Nc) ---
    # r_u 位于垂直边，对应 Python 索引行(1,3,5...)
    rc_u_center = r_u[1::2, :] 
    
    # 垂直方向 (Nearest 2): 系数 1/4 = 0.25
    # 列索引 0::2 和 1::2 对应粗网格单元内的左右两个细网格面，对于粗网格面来说它们是"最近的"
    term1 = 0.25 * (rc_u_center[:, 0::2] + rc_u_center[:, 1::2])
    
    # 周围四个 (Next Nearest 4): 系数 1/8 = 0.125
    left_cols = r_u[0:-1:2, :] 
    term2_left = 0.125 * (left_cols[:, 0::2] + left_cols[:, 1::2])
    
    right_cols = r_u[2::2, :]
    term2_right = 0.125 * (right_cols[:, 0::2] + right_cols[:, 1::2])
    
    rc_u = term1 + term2_left + term2_right
    
    # --- 2. Restrict r_v (N, N-1) -> (Nc, Nc-1) ---
    # r_v 位于水平边，对应 Python 索引列(1,3,5...)
    rc_v_center = r_v[:, 1::2]
    
    # 水平方向 (Nearest 2)
    term1_v = 0.25 * (rc_v_center[0::2, :] + rc_v_center[1::2, :])
    
    # 上下邻居 (Next Nearest 4)
    bot_rows = r_v[:, 0:-1:2]
    term2_bot = 0.125 * (bot_rows[0::2, :] + bot_rows[1::2, :])
    
    top_rows = r_v[:, 2::2]
    term2_top = 0.125 * (top_rows[0::2, :] + top_rows[1::2, :])
    
    rc_v = term1_v + term2_bot + term2_top
    
    # --- 3. Restrict r_div (N, N) -> (Nc, Nc) ---
    # 图片要求：Nearest four / 4 (即求平均)
    temp = r_div[0::2, :] + r_div[1::2, :] # Sum vertical
    rc_div_sum = temp[:, 0::2] + temp[:, 1::2] # Sum horizontal
    
    rc_div = rc_div_sum / 4.0 # Average
    
    return rc_u, rc_v, rc_div


def prolongate_error(e_u_coarse, e_v_coarse, e_p_coarse):
    '''
    提升算子 (Coarse -> Fine)
    U: 粗网格线(Nearest), 细网格线(Average of 2) -> 线性插值
    V: 同上
    P: Nearest -> 常数插值
    '''
    Nc = e_p_coarse.shape[0]
    N = Nc * 2
    
    e_u_fine = np.zeros((N + 1, N))
    e_v_fine = np.zeros((N, N + 1))
    e_p_fine = np.zeros((N, N))
    
    # --- 1. Prolongate U ---
    # e_u_coarse shape: (Nc+1, Nc)
    # 步骤 A: 水平方向扩展 (Nearest)
    # 粗网格的第 j 列对应细网格的 2j 和 2j+1 列
    u_expanded = np.repeat(e_u_coarse, 2, axis=1) # (Nc+1, 2Nc)
    
    # 步骤 B: 垂直方向插值
    # 粗网格线 (Rows 0, 2, 4...) -> 直接复制 (Nearest)
    e_u_fine[0::2, :] = u_expanded
    
    # 细网格线 (Rows 1, 3, 5...) -> 上下平均 (Nearest 2 / 2)
    e_u_fine[1::2, :] = 0.5 * (u_expanded[:-1, :] + u_expanded[1:, :])
    
    # --- 2. Prolongate V ---
    # e_v_coarse shape: (Nc, Nc+1)
    # 步骤 A: 垂直方向扩展 (Nearest)
    # 粗网格的第 i 行对应细网格的 2i 和 2i+1 行
    v_expanded = np.repeat(e_v_coarse, 2, axis=0) # (2Nc, Nc+1)
    
    # 步骤 B: 水平方向插值
    # 粗网格线 (Cols 0, 2, 4...) -> 直接复制
    e_v_fine[:, 0::2] = v_expanded
    
    # 细网格线 (Cols 1, 3, 5...) -> 左右平均
    e_v_fine[:, 1::2] = 0.5 * (v_expanded[:, :-1] + v_expanded[:, 1:])
    
    # --- 3. Prolongate P ---
    # 图片要求: Nearest (即最近邻/常数插值)
    # 每个粗网格单元的值赋给对应的 2x2 细网格单元
    # 使用 kron 进行块复制
    e_p_fine = np.kron(e_p_coarse, np.ones((2,2)))
    
    return e_u_fine, e_v_fine, e_p_fine