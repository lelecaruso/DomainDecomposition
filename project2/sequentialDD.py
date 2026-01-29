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

# Global Variables
na = np.newaxis
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def local_mesh(Lx, Ly, nx, ny, j, J):
    # Get MPI parameters
    # J = size
    # j = rank
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
    # important to notice that only the first subdomain has physical bottom boundary
    for i in range(nx - 1):
        edge = [i, i + 1]
        if j == 0:
            phys.append(edge)
        else:
            artf.append(edge)
    # Top boundary
    # important to notice that only the last subdomain has physical top boundary
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
    """
    Maps the local nodes (Omega_j) to the artificial interface (Sigma_j).
    Returns a matrix of size (nbelt_art x nv_loc).
    """
    ny_loc = (ny - 1) // J + 1
    nv_loc = nx * ny_loc
    # Number of nodes on the artificial interface
    # belt_artf contains pair of vertices, so we extract unique nodes
    interface_nodes = np.unique(belt_artf)
    nbelt_nodes = len(interface_nodes)

    rows = np.arange(nbelt_nodes)  # interface index
    cols = interface_nodes  # local node indices: the actual node numbers in the subdomain mesh
    data = np.ones(nbelt_nodes)

    # Bj: (interface_nodes x local_nodes)
    Bj = csr_matrix((data, (rows, cols)), shape=(nbelt_nodes, nv_loc))
    return Bj


def Cj_matrix(nx, ny_glob, j, J):
    assert 0 <= j < J

    global_size = 2 * nx * (J - 1)

    # Figure out which interfaces this subdomain has
    has_bottom_interface = j > 0
    has_top_interface = j < J - 1

    num_local_interfaces = has_bottom_interface + has_top_interface
    local_size = num_local_interfaces * nx

    Cj = np.zeros((local_size, global_size))

    local_offset = 0  # Where to write in the local vector

    # Bottom interface (if it exists)
    if has_bottom_interface:
        interface_number = j - 1  # Interface between subdomain (j-1) and j

        # Subdomain j is ABOVE this interface, so we want the "view from above"
        # which is stored in the second half of this interface's global data
        global_start = 2 * interface_number * nx + nx  # Second block

        for i in range(nx):
            Cj[local_offset + i, global_start + i] = 1

        local_offset += nx

    # Top interface (if it exists)
    if has_top_interface:
        interface_number = j  # Interface between subdomain j and (j+1)

        # Subdomain j is BELOW this interface, so we want the "view from below"
        # which is stored in the first half of this interface's global data
        global_start = 2 * interface_number * nx  # First block

        for i in range(nx):
            Cj[local_offset + i, global_start + i] = 1

        local_offset += nx

    return Cj


def Aj_matrix(vtxj, eltj, beltj_phys, kappa):
    """
    given the global assembly of A matrix:
    M = mass(vtx_loc, elt_loc)
    Mb = mass(vtx_loc, belt)
    K = stiffness(vtx_loc, elt_loc)
    A = K - k**2 * M - 1j * k * Mb  # matrix of linear system
    """
    #  Compute the local stiffness matrix
    Kj = stiffness(vtxj, eltj)

    # Compute the local mass matrix
    Mj = mass(vtxj, eltj)

    # 3. Compute the local boundary mass matrix for physical boundaries
    # This represents the Robin/absorbing boundary condition (impedance)
    Mbj = mass(vtxj, beltj_phys)

    # 4. Assemble the local operator: A = K - k^2 * M - i * k * Mb
    # Note: We use 1j for the imaginary unit
    Aj = Kj - kappa**2 * Mj - 1j * kappa * Mbj

    return Aj


def Tj_matrix(vtxj, beltj_artf, Bj, k):

    # Mass matrix on local nodes (nv_loc x nv_loc)
    M_local = mass(vtxj, beltj_artf)

    # Project to interface space: Bj @ M @ Bj.T
    # Resulting Tj is (nbelt_art x nbelt_art)
    Tj_interface = Bj @ M_local @ Bj.T

    return k * Tj_interface


def Sj_factorization(Aj, Tj, Bj):
    # Expand Tj back to local dimensions (nv_loc x nv_loc)
    # Since Bj is real, Bj.T is sufficient for the adjoint
    Tj_expanded = Bj.T @ Tj @ Bj

    # Assemble the complex local operator
    Sj = Aj - 1j * Tj_expanded

    # Factorize (splu requires CSC format)
    Sj_fact = spla.splu(csc_matrix(Sj))

    return Sj_fact


def bj_vector(vtxj, eltj, sp, k):

    # Number of local vertices
    nv_loc = vtxj.shape[0]

    # Initialize local RHS vector
    bj = np.zeros(nv_loc, dtype=np.complex128)
    Mj = mass(vtxj, eltj)
    # Evaluate point sources at local vertices
    bj = Mj @ point_source(sp, k)(vtxj)  # linear system RHS (source term)

    return bj


def g_vector(nx, ny, J, Bj_list, Cj_list, bj_list, Sj_fact_list):
    # Initialize global interface vector
    g_size = (J - 1) * nx * 2
    g = np.zeros(g_size, dtype=np.complex128)
    # Loop over all subdomains
    for j in range(J):
        y_j = Sj_fact_list[j].solve(bj_list[j])
        # Extract interface values using B_j
        y_j_interface = Bj_list[j] @ y_j
        # C_j maps local interface to global interface
        local_contribution = Cj_list[j].T @ y_j_interface
        # This exchanges information between subdomains
        Pi = Pi_operator(nx, J)
        exchanged = Pi @ local_contribution
        g += -2j * exchanged

    return g


def S_operator(nx, ny, J, Bj_list, Tj_list, Cj_list, Sj_fact_list):
    vector_size = 2 * nx * (J - 1)  # Each interface seen from both sides

    def matvec(x):
        result = np.zeros_like(x, dtype=np.complex128)

        for j in range(J):
            x_local_interface = Cj_list[j] @ x
            x_local_full = Bj_list[j].T @ Tj_list[j] @ x_local_interface

            y = Sj_fact_list[j].solve(x_local_full)
            y_interface = x_local_interface + 2j * Bj_list[j] @ y
            result += Cj_list[j].T @ y_interface

        return result

    return spla.LinearOperator((vector_size, vector_size), matvec=matvec)


def Pi_operator(nx, J):
    vector_size = 2 * nx * (J - 1)  # Each interface seen from both sides

    def matvec(x):
        result = np.zeros_like(x)

        for interface_idx in range(J - 1):
            # Each interface has 2*nx values (nx from each side)
            idx1_start = 2 * interface_idx * nx
            idx1_end = idx1_start + nx
            idx2_start = idx1_end
            idx2_end = idx2_start + nx

            # Swap the two halves
            result[idx1_start:idx1_end] = x[idx2_start:idx2_end]
            result[idx2_start:idx2_end] = x[idx1_start:idx1_end]

        return result

    return spla.LinearOperator((vector_size, vector_size), matvec=matvec)


def interface_operator(nx, ny, J, Bj_list, Tj_list, Cj_list, Sj_fact_list):

    vector_size = 2 * nx * (J - 1)

    S = S_operator(nx, ny, J, Bj_list, Tj_list, Cj_list, Sj_fact_list)
    Pi = Pi_operator(nx, J)

    def matvec(x):
        return x + Pi.matvec(S.matvec(x))

    return spla.LinearOperator(
        (vector_size, vector_size), matvec=matvec, dtype=np.complex128
    )


def uj_solution(Sj_fact_list, Bj_list, Cj_list, Tj_list, bj_list, sol, J):
    sols = None
    for j in range(J):
        x = Sj_fact_list[j].solve(
            Bj_list[j].T @ Tj_list[j] @ (Cj_list[j] @ sol) + bj_list[j]
        )
        if sols is None:
            sols = x
        else:
            sols = np.hstack((sols, x))
    return sols


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


def u_global(uu_gmres):
    # build the pseudoinverse to compute global u
    R = None
    for j in range(J):
        Rj = Rj_matrix(nx, ny, j, J)
        if R is None:
            R = Rj.T
        else:
            R = np.hstack((R, Rj.T))
    R = R.T

    Ru = R.T @ uu_gmres
    d = np.sum(R * R, axis=0)
    u = Ru / d

    return u


if __name__ == "__main__":
    np.random.seed(1234)
    Lx = 1
    Ly = 2
    nx = int(1 + Lx * 64)
    ny = int(1 + Ly * 64)
    j = 2
    J = 4
    k = 16
    ns = 8
    sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in np.arange(ns)]

    Bj_list = []
    Cj_list = []
    Tj_list = []
    Sj_fact_list = []
    bj_list = []
    Rj_list = []

    for j in range(J):
        vtx_loc, elt_loc = local_mesh(Lx, Ly, nx, ny, j, J)
        belt_phys, belt_art = local_boundary(nx, ny, j, J)
        M = mass(vtx_loc, elt_loc)
        Mb = mass(vtx_loc, belt_phys)
        K = stiffness(vtx_loc, elt_loc)
        Aj = K - k**2 * M - 1j * k * Mb
        Bj = Bj_matrix(nx, ny, j, J, belt_art)
        Cj = Cj_matrix(nx, ny, j, J)
        Tj = Tj_matrix(vtx_loc, belt_art, Bj, k)
        Sj = Sj_factorization(Aj, Tj, Bj)

        bj = bj_vector(vtx_loc, elt_loc, sp, k)

        Bj_list.append(Bj)
        Cj_list.append(Cj)
        Tj_list.append(Tj)
        Sj_fact_list.append(Sj)
        bj_list.append(bj)

    g = g_vector(nx, ny, J, Bj_list, Cj_list, bj_list, Sj_fact_list)
    I_PIS = interface_operator(nx, ny, J, Bj_list, Tj_list, Cj_list, Sj_fact_list)

    residuals_gmres = []
    residuals_fp = []

    # GMRES
    def callback(x):
        residuals_gmres.append(x)

    t0 = time.perf_counter()
    interf_sol_gmres, _ = spla.gmres(
        I_PIS,
        g,
        atol=1e-8,
        restart=50,
        maxiter=500,
        callback=callback,
        callback_type="pr_norm",
    )
    tgmrs = time.perf_counter() - t0

    uu_gmres = uj_solution(
        Sj_fact_list, Bj_list, Cj_list, Tj_list, bj_list, interf_sol_gmres, J
    )
    u_gmres = u_global(uu_gmres)

    # FIXED POINT
    starting_sol = np.zeros(interf_sol_gmres.shape[0])

    t0 = time.perf_counter()
    interf_sol_fp, residuals_fp = fixed_point(0.5, starting_sol, g, I_PIS)
    tfp = time.perf_counter() - t0

    uu_fp = uj_solution(
        Sj_fact_list, Bj_list, Cj_list, Tj_list, bj_list, interf_sol_fp, J
    )
    u_fp = u_global(uu_fp)

    # error between solvers
    print(
        "Gmres vs Fp RELATIVE ERROR NORM = ",
        np.linalg.norm(u_gmres - u_fp) / np.linalg.norm(u_gmres),
    )
    print(f"Elapsed time: \ngmres -> { tgmrs*1000 } ms \nfp ----> { tfp*1000 } ms  \n")

    # plot both residuals
    plt.figure()
    plt.semilogy(residuals_gmres)
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.grid(True, which="both")
    plt.savefig("residualsGMRES.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.semilogy(residuals_fp)
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.grid(True, which="both")
    plt.savefig("residualsFP.png", dpi=300, bbox_inches="tight")
    plt.close()


"""
        # --- Plot 1: mesh ---
        plt.figure()
        plot_mesh(vtx_loc, elt_loc)  # slow for fine meshes
        plt.savefig("mesh.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- Plot 2: real part ---
        plt.figure()
        plot_mesh(vtx_loc, elt_loc, np.real(x))
        plt.colorbar()
        plt.savefig("solution_real.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- Plot 3: magnitude ---
        plt.figure()
        plot_mesh(vtx_loc, elt_loc, np.abs(x))
        plt.colorbar()
        plt.savefig("solution_abs.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --- Plot 4: residual history ---
        plt.figure()
        plt.semilogy(residuals)
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.grid(True, which="both")
        plt.savefig("residuals.png", dpi=300, bbox_inches="tight")
        plt.close()
"""
