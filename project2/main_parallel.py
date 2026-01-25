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
    # Number of points per subdomain in y (with interface sharing)
    ny_loc = (ny - 1) // J + 1
    # Local vertical size
    Ly_loc = Ly / J
    # Build local mesh
    vtx_loc, elt_loc = mesh(nx, ny_loc, Lx, Ly_loc)
    # Shift to global y-position
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
    for k in range(ny_loc - 1):
        phys.append([k * nx + nx - 1, (k + 1) * nx + nx - 1])
    return np.array(phys, dtype=int), np.array(artf, dtype=int)


def Rj_matrix(nx, ny_glob, j, J):
    assert j < J
    assert (ny_glob - 1) % J == 0
    ny = ny_glob // J + 1
    first_vertex_in_omega_j = nx * (ny - 1) * j
    tot_vertex_in_omega_j = nx * ny
    Rj = np.zeros((tot_vertex_in_omega_j, nx * ny_glob))
    row = 0
    col = first_vertex_in_omega_j
    while row < Rj.shape[0]:
        Rj[row, col] = 1
        col += 1
        row += 1
    return Rj


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


def Cj_matrix(nx, ny_glob, j, J):
    # Maps local interface vector to the Global Interface Vector
    assert 0 <= j < J
    global_size = 2 * nx * (J - 1)
    has_bottom_interface = j > 0
    has_top_interface = j < J - 1
    num_local_interfaces = has_bottom_interface + has_top_interface
    local_size = num_local_interfaces * nx
    Cj = np.zeros((local_size, global_size))
    local_offset = 0

    if has_bottom_interface:
        interface_number = j - 1
        # View from above (second block of the interface)
        global_start = 2 * interface_number * nx + nx
        for i in range(nx):
            Cj[local_offset + i, global_start + i] = 1
        local_offset += nx

    if has_top_interface:
        interface_number = j
        # View from below (first block of the interface)
        global_start = 2 * interface_number * nx
        for i in range(nx):
            Cj[local_offset + i, global_start + i] = 1
        local_offset += nx
    return Cj


def Aj_matrix(vtxj, eltj, beltj_phys, kappa):
    Kj = stiffness(vtxj, eltj)
    Mj = mass(vtxj, eltj)
    Mbj = mass(vtxj, beltj_phys)
    Aj = Kj - kappa**2 * Mj - 1j * kappa * Mbj
    return Aj


def Tj_matrix(vtxj, beltj_artf, Bj, k):
    M_local = mass(vtxj, beltj_artf)
    Tj_interface = Bj @ M_local @ Bj.T
    return k * Tj_interface


def Sj_factorization(Aj, Tj, Bj):
    Tj_expanded = Bj.T @ Tj @ Bj
    Sj = Aj - 1j * Tj_expanded
    Sj_fact = spla.splu(csc_matrix(Sj))
    return Sj_fact


def bj_vector(vtxj, eltj, sp, k):
    bj = np.zeros(vtxj.shape[0], dtype=np.complex128)
    Mj = mass(vtxj, eltj)
    bj = Mj @ point_source(sp, k)(vtxj)
    return bj


# --- PARALLEL OPERATORS ---


def Pi_operator(nx, J):
    """
    Global Pi Operator.
    Even though we are in parallel, the vector 'x' passed to this
    is the global interface vector (replicated on all ranks).
    Therefore, we use the global swapping logic.
    """
    vector_size = 2 * nx * (J - 1)

    def matvec(x):
        result = np.zeros_like(x)
        for interface_idx in range(J - 1):
            idx1_start = 2 * interface_idx * nx
            idx1_end = idx1_start + nx
            idx2_start = idx1_end
            idx2_end = idx2_start + nx

            # Swap halves
            result[idx1_start:idx1_end] = x[idx2_start:idx2_end]
            result[idx2_start:idx2_end] = x[idx1_start:idx1_end]
        return result

    return spla.LinearOperator((vector_size, vector_size), matvec=matvec)


def g_vector(nx, ny, J, Bj, Cj, bj, Sj_fact):
    # Compute local contribution: Cj.T @ Bj @ Sj^{-1} @ bj
    g_size = (J - 1) * nx * 2

    y_j = Sj_fact.solve(bj)
    # Extract interface values using B_j
    y_j_interface = Bj @ y_j
    # C_j maps local interface to global interface
    local_contribution = Cj.T @ y_j_interface

    # Accumulate contributions from all ranks
    # The global vector has size S = 2 * nx * (J-1)
    global_accumulated = np.zeros_like(local_contribution, dtype=np.complex128)
    comm.Allreduce(local_contribution, global_accumulated, op=MPI.SUM)

    # Apply global Pi operator and scaling
    # g = -2i * Pi * (Sum of local contributions)
    Pi = Pi_operator(nx, J)
    exchanged = Pi @ global_accumulated
    g = -2j * exchanged

    return g


def S_operator(nx, ny, J, Bj, Tj, Cj, Sj_fact):
    vector_size = 2 * nx * (J - 1)

    def matvec(x):
        # x is the Global Interface Vector (replicated)

        # Restrict to local interface and map to full interface sizes
        x_local_interface = Cj @ x
        x_local_full = Bj.T @ Tj @ x_local_interface  # size S

        # Local Solve
        y = Sj_fact.solve(x_local_full)

        # Construct local update for the interface
        # The term is: Cj.T @ (x_j + 2j * Bj @ y)
        y_interface = x_local_interface + 2j * Bj @ y
        # expand to global interface size
        local_vec = Cj.T @ y_interface

        # Accumulate result from all ranks
        result = np.zeros_like(x, dtype=np.complex128)
        comm.Allreduce(local_vec, result, op=MPI.SUM)

        return result

    return spla.LinearOperator(
        (vector_size, vector_size), matvec=matvec, dtype=np.complex128
    )


def interface_operator(nx, ny, J, Bj, Tj, Cj, Sj_fact):
    vector_size = 2 * nx * (J - 1)
    S = S_operator(nx, ny, J, Bj, Tj, Cj, Sj_fact)
    Pi = Pi_operator(nx, J)

    def matvec(x):
        # (I + Pi S) x
        return x + Pi.matvec(S.matvec(x))

    return spla.LinearOperator(
        (vector_size, vector_size), matvec=matvec, dtype=np.complex128
    )


def uj_solution(Sj_fact, Bj, Cj, Tj, bj, sol_interface, J):
    # This computes the solution on the specific rank's subdomain
    # sol_interface is the Global interface solution
    rhs = Bj.T @ Tj @ (Cj @ sol_interface) + bj
    u_local = Sj_fact.solve(rhs)
    return u_local


def fixed_point(w, starting, g_vector, I_PIS, maxit=500, tol=1e-8):
    sol = starting
    residuals = []
    res = tol + 1
    it = 0
    while res > tol and it < maxit:
        diff = g_vector - I_PIS.matvec(sol)
        sol = sol + w * diff
        res = np.linalg.norm(diff)
        residuals.append(res)
        it += 1

    return sol, residuals


def u_global_gather(nx, ny, J, u_local, rank):
    """
    Gather local solutions from all ranks and reconstruct the global solution.
    """
    # Gather all local u vectors to rank 0
    # We gather as a list of numpy arrays
    u_list = comm.gather(u_local, root=0)

    if rank == 0:
        # Build the global Reconstruct matrix R
        R = None
        for r_idx in range(J):
            Rj = Rj_matrix(nx, ny, r_idx, J)
            if R is None:
                R = Rj.T
            else:
                R = np.hstack((R, Rj.T))
        R = R.T

        # Concatenate received solutions
        uu_global_concat = np.hstack(u_list)

        Ru = R.T @ uu_global_concat
        d = np.sum(R * R, axis=0)
        u_final = Ru / d
        return u_final


if __name__ == "__main__":
    np.random.seed(1234)
    Lx = 1
    Ly = 2
    nx = int(1 + Lx * 32)
    ny = int(1 + Ly * 32)
    k = 16
    ns = 8
    # Ensure source points are same on all ranks
    sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in np.arange(ns)]

    # 1. Local Setup
    vtx_loc, elt_loc = local_mesh(Lx, Ly, nx, ny, j, J)
    belt_phys, belt_art = local_boundary(nx, ny, j, J)

    M = mass(vtx_loc, elt_loc)
    Mb = mass(vtx_loc, belt_phys)
    K = stiffness(vtx_loc, elt_loc)
    Aj = K - k**2 * M - 1j * k * Mb

    Bj = Bj_matrix(nx, ny, j, J, belt_art)
    Cj = Cj_matrix(nx, ny, j, J)
    Tj = Tj_matrix(vtx_loc, belt_art, Bj, k)
    Sj_fact = Sj_factorization(Aj, Tj, Bj)
    bj = bj_vector(vtx_loc, elt_loc, sp, k)

    #  Construct Global/Parallel Operators
    g = g_vector(nx, ny, J, Bj, Cj, bj, Sj_fact)

    I_PIS = interface_operator(nx, ny, J, Bj, Tj, Cj, Sj_fact)

    # Solver (Fixed Point)
    global_interface_size = 2 * nx * (J - 1)
    starting_sol = np.zeros(global_interface_size, dtype=np.complex128)

    comm.Barrier()
    t0 = time.perf_counter()

    interf_sol_fp, residuals_fp = fixed_point(0.5, starting_sol, g, I_PIS)

    comm.Barrier()
    tfp = time.perf_counter() - t0

    # 4. Recover Local Solution
    u_local = uj_solution(Sj_fact, Bj, Cj, Tj, bj, interf_sol_fp, J)

    # 5. Gather and Plot (Rank 0 only)
    u_final = u_global_gather(nx, ny, J, u_local, rank)

    if rank == 0:
        print(f"Parallel Fixed Point Converged in {len(residuals_fp)} iterations.")
        print(f"Elapsed time (MPI size {J}): {tfp*1000:.2f} ms")

        # Plot Residuals
        plt.figure()
        plt.semilogy(residuals_fp)
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.grid(True, which="both")
        plt.title(f"Parallel FP Residuals (J={J})")
        plt.savefig("residualsFP_MPI.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot Solution Real Part
        vtx_global, elt_global = mesh(nx, ny, Lx, Ly)
        plt.figure()
        plot_mesh(vtx_global, elt_global, np.real(u_final))
        plt.colorbar()
        plt.title("Global Solution (Real Part)")
        plt.savefig("solution_real_MPI.png", dpi=300, bbox_inches="tight")
        plt.close()
