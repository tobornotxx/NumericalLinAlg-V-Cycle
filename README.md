# 数值代数大作业README文档

实现了以下的求解Stokes方程方法：

1. 基于DGS磨光（红黑更新）的V-Cycle算法；
2. 基于Uzawa Iteration Method；
3. 使用DGS磨光的V-Cycle作为预条件子的Inexact Uzawa Method；

包含以下的文件：

1. main.py: 
   1. `def plot_solution(N, u, v, p, u_ex, v_ex, p_ex, method_idx: int)`的画图辅助函数，做真解和数值解的可视化对比
   2. `def main()`的主函数，组织计算流程。
2. TrueSolution.py:
   1. `def create_grids(N)`返回形状分别为N, N; N+1, N; N, N+1的p, u, v函数的网格坐标x和y
   2. `def get_exact_solution(N)`返回在对应网格坐标上的u, v, p, f, g的精确数值
3. Operators.py:
   1. `def apply_laplacian_u(u, h)`返回-Laplace u
   2. `apply_laplacian_v(v,h)`返回-Laplace v
   3. `apply_gradient_p(p,h)`返回p的梯度
   4. `apply_divergence_uv(u, v, h)`求速度场的散度
   5. `def compute_residuals_stokes(u, v, p, f, g, h, bcs)`计算残差，真解减去数值解
4. Smoother.py
   1. `def smooth_momentum(u, v, p, f, g, h, bcs)`DGS算法的第一步，使用一步Gauss-Seidel迭代更新一次速度，使用红黑更新顺序
   2. `def apply_distributive_correction(u, v, p, h, g_div=None)`DGS算法第二步，基于当前残差分布式更新速度与压力分量
   3. `def dgs_step(u, v, p, f, g, h, bcs, g_div=None)`一步DGS迭代
   4. 一些测试收敛性的函数。建立一个测试问题并尝试用DGS求解。
5. Transfer.py
   1. `def restrict_residuals(r_u, r_v, r_div)`将细网格残差限制到更粗的网格上
   2. `def prolongate_error(e_u_coarse, e_v_coarse, e_p_coarse)`将粗网格残差提升到细网格上
6. vcycle.py
   1. `def enforce_dirichlet_bc(u, v)`边界处理辅助函数，修改边界值保证计算收敛性
   2. `def v_cycle_recursive(u, v, p, f, g, g_div, h, bcs, nu1, nu2, min_N)`用DGS Vcycle直接求解Stokes方程的函数逻辑
   3. `def v_cycle_velocity(u, v, f_minus_bp, h, bcs, nu1=2, nu2=2, min_N=4)`用DGS Vcycle做Uzawa求解第一步预条件的逻辑。
7. Uzawa.py
   1. `def get_rhs_with_bcs(f, g, bcs, h)`辅助函数，计算Uzawa的CG求解方程的右侧项
   2. `def apply_laplacian_u_direction(p_dir, h), def apply_laplacian_v_direction(p_dir, h)`辅助函数，计算共轭梯度法中的Ap即Laplace u和Laplace v
   3. `def cg_solve_u(u, rhs, h, max_iter=200, tol=1e-10), def cg_solve_v(v, rhs, h, max_iter=200, tol=1e-10)`共轭梯度法求解速度分量u和v
   4. `def uzawa_step(u, v, p, f, g, h, bcs, alpha)`Uzawa法的运行函数逻辑。
8. InexactUzawa.py
   1. `def smooth_velocity(u, v, f_minus_bp, h, bcs)` Inexact Uzawa中预优的V-Cycle求解第一步，Gauss-Seidel更新速度
   2. `def apply_distributive_correction_velocity(u, v, h)`第二步，分布式更新速度
   3. `def dgs_velocity_step(u, v, f_minus_bp, h, bcs)`一次子问题DGS磨光迭代。
9. Solver.py:
   1. `def solve_problem_1(N, tol=1e-8, max_iter=10000)`第一问的循环启动与终止，残差计算等
   2. `def solve_problem_2(N, alpha=1.0, tol=1e-8, max_iter=10000)`第二问的循环启动终止，残差计算等
   3. `def solve_problem_3(N,alpha = 1.0, tol=1e-8, max_iter=100)`第三问的循环启动终止，残差计算等