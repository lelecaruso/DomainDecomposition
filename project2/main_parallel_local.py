from math import pi
import time
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, csc_matrix
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
from utils import mesh, mass, stiffness, plot_mesh, point_source, boundary

# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
j = rank  # Subdomain index
J = size  # Total subdomains


def local_mesh(Lx, Ly, nx, ny, j, J):
    ny_loc = (ny - 1) // J + 1
    Ly_loc = Ly / J
    vtx_loc, elt_loc = mesh(nx, ny_loc, Lx, Ly_loc)
    vtx_loc[:, 1] += j * Ly_loc
    return vtx_loc, elt_loc


def local_boundary(nx, ny, j, J):
    ny_loc = (ny - 1) // J + 1
    phys = []
    artf = []
    # Bottom boundary
    for i in range(nx - 1):
        edge = [i, i + 1]
        if j == 0:
            phys.append(edge)
        else:
            artf.append(edge)
    # Top boundary
    offset = (ny_loc - 1) * nx
    for i in range(nx - 1):
        edge = [offset + i, offset + i + 1]
        if j == J - 1:
            phys.append(edge)
        else:
            artf.append(edge)
    # Left-Right boundary (always physical)
    for k in range(ny_loc - 1):
        phys.append([k * nx, (k + 1) * nx])
        phys.append([k * nx + nx - 1, (k + 1) * nx + nx - 1])
    return np.array(phys, dtype=int), np.array(artf, dtype=int)


def Bj_matrix(nx, ny, j, J, belt_artf):
    ny_loc = (ny - 1) // J + 1
    nv_loc = nx * ny_loc
    interface_nodes = np.unique(belt_artf)
    nbelt_nodes = len(interface_nodes)
    rows = np.arange(nbelt_nodes)
    cols = interface_nodes
    data = np.ones(nbelt_nodes)
    Bj = csr_matrix((data, (rows, cols)), shape=(nbelt_nodes, nv_loc))
    return Bj


def Tj_matrix(vtxj, beltj_artf, Bj, k):
    M_local = mass(vtxj, beltj_artf)
    Tj_interface = Bj @ M_local @ Bj.T
    return k * Tj_interface


def Sj_factorization(Aj, Tj, Bj):
    Tj_expanded = Bj.T @ Tj @ Bj
    Sj = Aj - 1j * Tj_expanded
    Sj_fact = spla.splu(csc_matrix(Sj))
    return Sj_fact


def Rj_matrix(nx, ny_glob, j, J):
    ny = ny_glob // J + 1
    first_vertex_in_omega_j = nx * (ny - 1) * j
    tot_vertex_in_omega_j = nx * ny
    Rj = np.zeros((tot_vertex_in_omega_j, nx * ny_glob))
    for row in range(Rj.shape[0]):
        Rj[row, first_vertex_in_omega_j + row] = 1
    return Rj


# --- DISTRIBUTED PARALLEL OPERATORS ---


def apply_Pi_local(v_local, nx, rank, size):
    """
    Exchange interface data with neighbors using Sendrecv.
    v_local: contains bottom interface data (first nx) then top (next nx)
    """
    v_exchanged = np.zeros_like(v_local)

    # 1. BOTTOM EXCHANGE (with rank-1)
    if rank > 0:
        # My bottom interface is at index [0:nx]
        my_bottom = v_local[0:nx]
        neighbor_below = rank - 1
        recv_buffer = np.zeros(nx, dtype=np.complex128)

        # Tags: 100 for Top->Bottom, 200 for Bottom->Top
        comm.Sendrecv(
            sendbuf=my_bottom,
            dest=neighbor_below,
            sendtag=100,
            recvbuf=recv_buffer,
            source=neighbor_below,
            recvtag=200,
        )
        v_exchanged[0:nx] = recv_buffer

    # 2. TOP EXCHANGE (with rank+1)
    if rank < size - 1:
        # If I have a bottom, my top starts at nx. If I'm Rank 0, it starts at 0.
        start_idx = nx if rank > 0 else 0
        my_top = v_local[start_idx : start_idx + nx]
        neighbor_above = rank + 1
        recv_buffer = np.zeros(nx, dtype=np.complex128)

        comm.Sendrecv(
            sendbuf=my_top,
            dest=neighbor_above,
            sendtag=200,
            recvbuf=recv_buffer,
            source=neighbor_above,
            recvtag=100,
        )
        v_exchanged[start_idx : start_idx + nx] = recv_buffer

    return v_exchanged


def S_local(v_local, Bj, Tj, Sj_fact):
    # Operator S = I + 2j * Bj * Sj^-1 * Bj.T * Tj
    w_vol = Bj.T @ (Tj @ v_local)
    y_vol = Sj_fact.solve(w_vol)
    return v_local + 2j * (Bj @ y_vol)


def distributed_fixed_point(
    g_local, Bj, Tj, Sj_fact, nx, rank, size, w=0.5, maxit=500, tol=1e-8
):
    x_local = np.zeros_like(g_local)
    residuals = []

    for it in range(maxit):
        # Step 1: Apply S locally
        Sx = S_local(x_local, Bj, Tj, Sj_fact)
        # Step 2: Apply Pi (Communication)
        PiSx = apply_Pi_local(Sx, nx, rank, size)
        # Step 3: (I + PiS)x
        Ax = x_local + PiSx

        # Step 4: Residual and Global Reduction
        diff = g_local - Ax
        local_sq_norm = np.sum(np.abs(diff) ** 2)
        global_res = np.sqrt(comm.allreduce(local_sq_norm, op=MPI.SUM))
        residuals.append(global_res)

        if global_res < tol:
            break

        x_local += w * diff

    return x_local, residuals


def u_global_gather(nx, ny, J, u_local, rank):
    u_list = comm.gather(u_local, root=0)
    if rank == 0:
        vtx_global, elt_global = mesh(nx, ny, 1, 2)
        u_final = np.zeros(vtx_global.shape[0], dtype=np.complex128)
        counts = np.zeros(vtx_global.shape[0])
        for r_idx, u_r in enumerate(u_list):
            Rj = Rj_matrix(nx, ny, r_idx, J)
            u_final += Rj.T @ u_r
            counts += np.sum(Rj, axis=0)
        return u_final / counts
    return None


if __name__ == "__main__":
    Lx, Ly = 1, 2
    nx, ny = int(1 + Lx * 64), int(1 + Ly * 64)
    k, ns = 16, 8
    np.random.seed(1234)
    sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in range(ns)]

    # Setup local matrices
    vtx_loc, elt_loc = local_mesh(Lx, Ly, nx, ny, rank, size)
    belt_phys, belt_art = local_boundary(nx, ny, rank, size)

    Aj = (
        stiffness(vtx_loc, elt_loc)
        - k**2 * mass(vtx_loc, elt_loc)
        - 1j * k * mass(vtx_loc, belt_phys)
    )
    Bj = Bj_matrix(nx, ny, rank, size, belt_art)
    Tj = Tj_matrix(vtx_loc, belt_art, Bj, k)
    Sj_fact = Sj_factorization(Aj, Tj, Bj)

    # Setup distributed RHS g
    bj = mass(vtx_loc, elt_loc) @ point_source(sp, k)(vtx_loc)
    yj_interf = Bj @ Sj_fact.solve(bj)
    g_local = -2j * apply_Pi_local(yj_interf, nx, rank, size)

    # Solve
    comm.Barrier()
    t0 = time.perf_counter()
    sol_int, res_hist = distributed_fixed_point(
        g_local, Bj, Tj, Sj_fact, nx, rank, size
    )
    solve_time = time.perf_counter() - t0

    # Reconstruct volume solution
    u_local = Sj_fact.solve(bj + Bj.T @ (Tj @ sol_int))
    u_final = u_global_gather(nx, ny, size, u_local, rank)

    if rank == 0:
        print(f"Converged in {len(res_hist)} iterations. Time: {solve_time:.4f}s")
        plt.figure()
        plot_mesh(mesh(nx, ny, Lx, Ly)[0], mesh(nx, ny, Lx, Ly)[1], np.real(u_final))
        plt.title("Parallel Fixed Point Solution")
        plt.savefig("solution_mpi_distributed.png")
        plt.show()
