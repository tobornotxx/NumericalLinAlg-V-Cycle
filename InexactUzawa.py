import numpy as np

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
    u[0, :] = 0.0
    u[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    return u, v


