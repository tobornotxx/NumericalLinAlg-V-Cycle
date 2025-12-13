import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Solvers import solve_problem_1, solve_problem_2, solve_problem_3
import itertools

def estimate_convergence_order(df: pd.DataFrame, refinement_ratio: float = 2.0):
    """
    通过对整个误差轨迹的对数进行线性回归来估计收敛阶。

    参数:
    df (pd.DataFrame): DataFrame 中必须包含一个名为 'Error L2' 的列。
    refinement_ratio (float): 每次迭代的细化率，
                               例如，如果每次步长减半，则细化率为 2。
                               您的原始代码中 np.log2 隐含了该值为 2。
    """
    # 过滤掉非正数的误差值，因为无法取对数
    df_positive_error = df[df['Error L2'] > 0].copy()

    if len(df_positive_error) < 2:
        print("\n数据点不足（少于2个），无法估计收敛阶。")
        return

    log_error = np.log(df_positive_error['Error L2'].values)
    iterations = np.arange(len(log_error))

    # 使用 polyfit 进行一阶多项式拟合（即线性拟合）
    # 返回值是 [斜率, 截距]
    slope, intercept = np.polyfit(iterations, log_error, 1)

    # 从斜率计算收敛阶 p
    # slope = -p * log(refinement_ratio)
    order = -slope / np.log(refinement_ratio)

    return order

def plot_solution(N, u, v, p, u_ex, v_ex, p_ex, method_idx: int):
    '''
    画图的辅助函数，做一个数值解和真解的对比图并存储。
    '''
    h = 1.0 / N
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # 绘制 U（插值到中心）
    u_center = 0.5 * (u[:-1, :] + u[1:, :])
    im0 = axs[0,0].imshow(u_center.T, origin='lower', extent=[0,1,0,1])
    axs[0,0].set_title('Computed U')
    plt.colorbar(im0, ax=axs[0,0])
    
    # u_ex是直接对中心计算的，直接绘图即可
    u_ex_c = 0.5 * (u_ex[:-1, :] + u_ex[1:, :])
    im1 = axs[1,0].imshow(u_ex_c.T, origin='lower', extent=[0,1,0,1])
    axs[1,0].set_title('Exact U')
    plt.colorbar(im1, ax=axs[1,0])

    # 绘制 V（插值到中心）
    v_center = 0.5 * (v[:, :-1] + v[:, 1:])
    im2 = axs[0,1].imshow(v_center.T, origin='lower', extent=[0,1,0,1])
    axs[0,1].set_title('Computed V')
    plt.colorbar(im2, ax=axs[0,1])
    
    # 绘制v_ex真解
    v_ex_c = 0.5 * (v_ex[:, :-1] + v_ex[:, 1:])
    im3 = axs[1,1].imshow(v_ex_c.T, origin='lower', extent=[0,1,0,1])
    axs[1,1].set_title('Exact V')
    plt.colorbar(im3, ax=axs[1,1])

    # 绘制 P
    im4 = axs[0,2].imshow(p.T, origin='lower', extent=[0,1,0,1])
    axs[0,2].set_title('Computed P')
    plt.colorbar(im4, ax=axs[0,2])
    
    # 绘制p_ex真解
    im5 = axs[1,2].imshow(p_ex.T, origin='lower', extent=[0,1,0,1])
    axs[1,2].set_title('Exact P')
    plt.colorbar(im5, ax=axs[1,2])

    plt.tight_layout()
    plt.savefig(f'stokes_solution_N={N}_method={method_idx}.png')
    plt.close()

def main():
    Ns = [64, 128, 256, 512, 1024, 2048] 
    results_1 = []

    print("Start Problem 1 (V-Cycle with DGS smoother)...")
    
    for N in Ns:
        iters, cpu_time, err, u, v, p, u_ex, v_ex, p_ex = solve_problem_1(N)
        results_1.append({
            'N': N,
            'Iterations': iters,
            'CPU Time': cpu_time,
            'Error L2': err
        })
        
        plot_solution(N, u, v, p, u_ex, v_ex, p_ex, 1)
        
    df = pd.DataFrame(results_1)
    print("\nResults Summary:")
    print(df)
    
    if len(df) >= 2:
        error_ratio = df['Error L2'].iloc[-2] / df['Error L2'].iloc[-1]
        order = np.log2(error_ratio)
        print(f"\nEstimated Convergence Order: {order:.2f}")

    results2 = []

    print("Start Problem 2 (Uzawa Iteration Method)...")
    
    for N in Ns:
        iters, cpu_time, err, u, v, p, u_ex, v_ex, p_ex = solve_problem_2(N)
        results2.append({
            'N': N,
            'Iterations': iters,
            'CPU Time': cpu_time,
            'Error L2': err
        })
        
        plot_solution(N, u, v, p, u_ex, v_ex, p_ex, 2)
        
    df = pd.DataFrame(results2)
    print("\nResults Summary:")
    print(df)
    
    if len(df) >= 2:
        error_ratio = df['Error L2'].iloc[-2] / df['Error L2'].iloc[-1]
        order = np.log2(error_ratio)
        print(f"\nEstimated Convergence Order: {order:.2f}")
    
    results3 = []
    print("Start Problem 3 (Inexact Uzawa)...")
    
    for N in Ns:
        iters, cpu_time, err, u, v, p, u_ex, v_ex, p_ex = solve_problem_3(N)
        results3.append({
            'N': N,
            'Iterations': iters,
            'CPU Time': cpu_time,
            'Error L2': err
        })
        
        plot_solution(N, u, v, p, u_ex, v_ex, p_ex, 3)
        
    df = pd.DataFrame(results3)
    print("\nResults Summary:")
    print(df)
    
    if len(df) >= 2:
        order = estimate_convergence_order(df, 2.0)
        print(f"\nEstimated Convergence Order: {order:.2f}")

def main_param_grid_search():
    """
    对 solve_problem_1 和 solve_problem_3 的超参数执行网格搜索，
    以找到最优的参数组合。
    """
    # 为网格搜索选择一个固定的、有代表性的 N
    N_test = 1024
    print(f"\n{'='*20} Starting Hyperparameter Grid Search (N={N_test}) {'='*20}\n")

    # --- Part 1: Grid Search for solve_problem_1 ---
    print("--- Grid Search for Problem 1 (V-Cycle) ---")
    param_grid_1 = {
        'nu': [2, 3, 4, 5],       # nu1 = nu2
        'min_N': [2, 4, 8]
    }
    
    search_results_1 = []
    
    # 生成参数组合
    keys_1, values_1 = zip(*param_grid_1.items())
    param_combinations_1 = [dict(zip(keys_1, v)) for v in itertools.product(*values_1)]
    
    for params in param_combinations_1:
        nu = params['nu']
        min_N = params['min_N']
        print(f"Testing Problem 1 with nu1=nu2={nu}, min_N={min_N}...")
        
        iters, cpu_time, err, _, _, _, _, _, _ = solve_problem_1(N_test, nu1=nu, nu2=nu, min_N=min_N)
        
        search_results_1.append({
            'nu': nu,
            'min_N': min_N,
            'Iterations': iters,
            'CPU Time': cpu_time,
            'Error L2': err
        })

    df_search_1 = pd.DataFrame(search_results_1)
    print("\nGrid Search Results for Problem 1:")
    print(df_search_1)
    
    # 找到最优参数（首先按迭代次数排序，然后按CPU时间排序）
    best_params_1 = df_search_1.sort_values(by=['Iterations', 'CPU Time']).iloc[0]
    print("\nBest Parameters for Problem 1:")
    print(best_params_1)
    print("-" * 50)

    # --- Part 2: Grid Search for solve_problem_3 ---
    print("\n--- Grid Search for Problem 3 (Inexact Uzawa) ---")
    param_grid_3 = {
        'alpha': [0.5, 1.0, 1.5],
        'nu': [2, 3, 4, 5],      # nu1 = nu2
        'min_N': [2, 4, 8],
        'cg_tol': [1e-2, 1e-3, 1e-4]
    }
    
    search_results_3 = []

    # 生成参数组合
    keys_3, values_3 = zip(*param_grid_3.items())
    param_combinations_3 = [dict(zip(keys_3, v)) for v in itertools.product(*values_3)]

    for params in param_combinations_3:
        alpha = params['alpha']
        nu = params['nu']
        min_N = params['min_N']
        cg_tol = params['cg_tol']
        print(f"Testing Problem 3 with alpha={alpha}, nu1=nu2={nu}, min_N={min_N}, cg_tol={cg_tol:.1e}...")
        
        iters, cpu_time, err, _, _, _, _, _, _ = solve_problem_3(
            N_test, alpha=alpha, nu1=nu, nu2=nu, min_N=min_N, cg_tol=cg_tol
        )
        
        search_results_3.append({
            'alpha': alpha,
            'nu': nu,
            'min_N': min_N,
            'cg_tol': cg_tol,
            'Iterations': iters,
            'CPU Time': cpu_time,
            'Error L2': err
        })

    df_search_3 = pd.DataFrame(search_results_3)
    # 设置显示选项以避免科学记数法截断
    pd.set_option('display.float_format', '{:.2e}'.format)
    print("\nGrid Search Results for Problem 3:")
    print(df_search_3.to_string()) # 使用 to_string() 确保所有行都被打印
    
    # 找到最优参数
    best_params_3 = df_search_3.sort_values(by=['Iterations', 'CPU Time']).iloc[0]
    print("\nBest Parameters for Problem 3:")
    print(best_params_3)
    print("-" * 50)


if __name__ == "__main__":
    # main()
    main_param_grid_search()