import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Solvers import solve_problem_1, solve_problem_2, solve_problem_3

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
    # results_1 = []

    # print("Start Problem 1 (V-Cycle with DGS smoother)...")
    
    # for N in Ns:
    #     iters, cpu_time, err, u, v, p, u_ex, v_ex, p_ex = solve_problem_1(N)
    #     results_1.append({
    #         'N': N,
    #         'Iterations': iters,
    #         'CPU Time': cpu_time,
    #         'Error L2': err
    #     })
        
    #     plot_solution(N, u, v, p, u_ex, v_ex, p_ex, 1)
        
    # df = pd.DataFrame(results_1)
    # print("\nResults Summary:")
    # print(df)
    
    # if len(df) >= 2:
    #     error_ratio = df['Error L2'].iloc[-2] / df['Error L2'].iloc[-1]
    #     order = np.log2(error_ratio)
    #     print(f"\nEstimated Convergence Order: {order:.2f}")

    # results2 = []

    # print("Start Problem 2 (Uzawa Iteration Method)...")
    
    # for N in Ns:
    #     iters, cpu_time, err, u, v, p, u_ex, v_ex, p_ex = solve_problem_2(N)
    #     results2.append({
    #         'N': N,
    #         'Iterations': iters,
    #         'CPU Time': cpu_time,
    #         'Error L2': err
    #     })
        
    #     plot_solution(N, u, v, p, u_ex, v_ex, p_ex, 2)
        
    # df = pd.DataFrame(results2)
    # print("\nResults Summary:")
    # print(df)
    
    # if len(df) >= 2:
    #     error_ratio = df['Error L2'].iloc[-2] / df['Error L2'].iloc[-1]
    #     order = np.log2(error_ratio)
    #     print(f"\nEstimated Convergence Order: {order:.2f}")
    
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
        error_ratio = df['Error L2'].iloc[-2] / df['Error L2'].iloc[-1]
        order = np.log2(error_ratio)
        print(f"\nEstimated Convergence Order: {order:.2f}")

if __name__ == "__main__":
    main()