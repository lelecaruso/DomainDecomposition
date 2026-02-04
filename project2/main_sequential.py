"""
========================================================================
                        Domain Decomposition Project

                Davide Villani     -     Emanuele Caruso
========================================================================
"""


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
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.sparse.linalg as spla

    np.random.seed(1234)

    Lx, Ly = 1, 2
    k = 16
    ns = 8

    Js = [2, 4, 8, 12, 16]

    STUDY = "fixed_dofs_per_sub"      

    if STUDY == "fixed_domain":
        nx = int(1 + Lx * 64)
        ny = int(1 + Ly * 64)
    else:
        dofs_per_sub = 8000          # approx per subdomain

    sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in range(ns)]

    gmres_its = []
    total_dofs = []

    for J in Js:
        print(f"\n=== Number of subdomains J = {J} ===")

        if STUDY == "fixed_dofs_per_sub":
            ndofs_tot = J * dofs_per_sub
            nx = int(np.sqrt(ndofs_tot / Ly))
            ny = int(Ly * nx)

        Bj_list = []
        Cj_list = []
        Tj_list = []
        Sj_fact_list = []
        bj_list = []

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

        def callback(res):
            residuals_gmres.append(res)

        spla.gmres(
            I_PIS,
            g,
            atol=1e-8,
            restart=50,
            maxiter=500,
            callback=callback,
            callback_type="pr_norm",
        )

        gmres_its.append(len(residuals_gmres))
        total_dofs.append(nx * ny)

        print(f"GMRES iterations = {gmres_its[-1]}")
        print(f"Total DoFs = {nx * ny}")
