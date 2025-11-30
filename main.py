import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Solvers import solve_problem_1

def plot_solution(N, u, v, p, u_ex, v_ex, p_ex):
    h = 1.0 / N
    Xp, Yp = np.meshgrid(np.linspace(h/2, 1-h/2, N), np.linspace(h/2, 1-h/2, N), indexing='ij')
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot U (interpolated to center for viz)
    u_center = 0.5 * (u[:-1, :] + u[1:, :])
    im0 = axs[0,0].imshow(u_center.T, origin='lower', extent=[0,1,0,1])
    axs[0,0].set_title('Computed U')
    plt.colorbar(im0, ax=axs[0,0])
    
    im1 = axs[1,0].imshow(u_ex.T, origin='lower', extent=[0,1,0,1]) # u_ex shape is (N+1, N), plot directly? 
    # u_ex is on faces, let's just plot u_center_ex
    u_ex_c = 0.5 * (u_ex[:-1, :] + u_ex[1:, :])
    axs[1,0].imshow(u_ex_c.T, origin='lower', extent=[0,1,0,1])
    axs[1,0].set_title('Exact U')

    # Plot V
    v_center = 0.5 * (v[:, :-1] + v[:, 1:])
    im2 = axs[0,1].imshow(v_center.T, origin='lower', extent=[0,1,0,1])
    axs[0,1].set_title('Computed V')
    plt.colorbar(im2, ax=axs[0,1])
    
    v_ex_c = 0.5 * (v_ex[:, :-1] + v_ex[:, 1:])
    axs[1,1].imshow(v_ex_c.T, origin='lower', extent=[0,1,0,1])
    axs[1,1].set_title('Exact V')

    # Plot P
    im4 = axs[0,2].imshow(p.T, origin='lower', extent=[0,1,0,1])
    axs[0,2].set_title('Computed P')
    plt.colorbar(im4, ax=axs[0,2])
    
    im5 = axs[1,2].imshow(p_ex.T, origin='lower', extent=[0,1,0,1])
    axs[1,2].set_title('Exact P')

    plt.tight_layout()
    plt.savefig(f'stokes_solution_N{N}.png')
    plt.close()

def main():
    Ns = [256] # 可以尝试 128
    results = []
    
    print("Start Problem 1 (V-Cycle with DGS)...")
    
    for N in Ns:
        iters, cpu_time, err, u, v, p, u_ex, v_ex, p_ex = solve_problem_1(N)
        results.append({
            'N': N,
            'Iterations': iters,
            'CPU Time': cpu_time,
            'Error L2': err
        })
        
        plot_solution(N, u, v, p, u_ex, v_ex, p_ex)
        
    df = pd.DataFrame(results)
    print("\nResults Summary:")
    print(df)
    
    if len(df) >= 2:
        error_ratio = df['Error L2'].iloc[-2] / df['Error L2'].iloc[-1]
        order = np.log2(error_ratio)
        print(f"\nEstimated Convergence Order: {order:.2f}")

if __name__ == "__main__":
    main()