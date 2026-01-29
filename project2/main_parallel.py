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
from code import mesh, mass, stiffness, plot_mesh, point_source, boundary
from local import local_mesh, local_boundary, Bj_matrix, Tj_matrix, Sj_factorization, Rj_matrix, bj_vector

# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
j = rank  # Subdomain index
J = size  # Total subdomains

def Pi_local(v_local, nx, rank, size):
    v_exchanged = np.zeros_like(v_local)

    #bottom swap (with rank-1)
    if rank > 0:
        # My bottom interface is at index [0:nx]
        bottom = v_local[0:nx]
        neighbor_below = rank - 1
        recv_buffer = np.zeros(nx, dtype=np.complex128)

        # Tags: 1 for Top->Bottom, 2 for Bottom->Top
        comm.Sendrecv(
            sendbuf=bottom,dest=neighbor_below,sendtag=1,
            recvbuf=recv_buffer, source=neighbor_below, recvtag=2,)
        v_exchanged[0:nx] = recv_buffer

    #top swap (with rank+1)
    if rank < size - 1:
        # If I have a bottom, my top starts at nx. If I'm Rank 0, it starts at 0.
        start_idx = nx if rank > 0 else 0
        top = v_local[start_idx : start_idx + nx]
        neighbor_above = rank + 1
        recv_buffer = np.zeros(nx, dtype=np.complex128)

        comm.Sendrecv(
            sendbuf=top,dest=neighbor_above,sendtag=2,
            recvbuf=recv_buffer,source=neighbor_above,recvtag=1,)
        v_exchanged[start_idx : start_idx + nx] = recv_buffer

    return v_exchanged

def S_local(v_local, Bj, Tj, Sj_fact):
    # Operator S = I + 2j * Bj * Sj^-1 * Bj.T * Tj
    w_vol = Bj.T @ (Tj @ v_local)
    y_vol = Sj_fact.solve(w_vol)
    return v_local + 2j * (Bj @ y_vol)

def uj_solution(Sj_fact, bj, Bj, Tj, sol_int):
    return  Sj_fact.solve(bj + Bj.T @ (Tj @ sol_int))

def g_vector_local(yj_interf, nx, rank, size):
    return -2j * Pi_local(yj_interf, nx, rank, size)

def distributed_fixed_point(g_local, Bj, Tj, Sj_fact, nx, rank, size, w=0.99, maxit=2000, tol=1e-3):
    x_local = np.zeros_like(g_local)
    residuals = []

    for it in range(maxit):
        # Apply S locally
        Sx = S_local(x_local, Bj, Tj, Sj_fact)
        # Apply Pi (Communication)
        PiSx = Pi_local(Sx, nx, rank, size)
        # (I + PiS)x
        Ax = x_local + PiSx

        # Residual and Global Reduction
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
    nx, ny = int(1 + Lx * 32), int(1 + Ly * 32)
    k, ns = 16, 8
    np.random.seed(1234)
    sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in range(ns)]

    # Setup local matrices
    vtx_loc, elt_loc = local_mesh(Lx, Ly, nx, ny, rank, size)
    belt_phys, belt_art = local_boundary(nx, ny, rank, size)

    Al = stiffness(vtx_loc, elt_loc)
    Ml = mass(vtx_loc, elt_loc)
    Mli = mass(vtx_loc, belt_phys)
    Aj = ( Al - k**2 * Ml - 1j * k * Mli)
    Bj = Bj_matrix(nx, ny, rank, size, belt_art)
    Tj = Tj_matrix(vtx_loc, belt_art, Bj, k)
    Sj_fact = Sj_factorization(Aj, Tj, Bj)

    # Setup distributed RHS g
    bj = bj_vector(vtx_loc, elt_loc, sp, k)
    yj_interf = Bj @ Sj_fact.solve(bj)
    g_local = g_vector_local(yj_interf, nx, rank, size)

    # Solve
    comm.Barrier()
    t0 = time.perf_counter()
    sol_int, res_hist = distributed_fixed_point( g_local, Bj, Tj, Sj_fact, nx, rank, size )
    solve_time = time.perf_counter() - t0

    # Reconstruct volume solution
    u_local = uj_solution(Sj_fact, bj, Bj, Tj, sol_int)
    u_final = u_global_gather(nx, ny, size, u_local, rank)

    if rank == 0:
        print(f"Converged in {len(res_hist)} iterations. Time: {solve_time:.4f}s")
        plt.figure()
        plot_mesh(mesh(nx, ny, Lx, Ly)[0], mesh(nx, ny, Lx, Ly)[1], np.real(u_final))
        plt.title("Parallel Fixed Point Solution")
        plt.savefig("../plots/mpi_sol_real.png")

        plt.figure()
        plot_mesh(mesh(nx, ny, Lx, Ly)[0], mesh(nx, ny, Lx, Ly)[1], np.abs(u_final))
        plt.title("Parallel Fixed Point Solution")
        plt.savefig("../plots/mpi_sol_abs.png")

        plt.figure()
        plt.semilogy(res_hist)
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.grid(True, which="both")
        plt.savefig("../plots/mpi_res_fp.png", dpi=300, bbox_inches="tight")
        plt.close()
