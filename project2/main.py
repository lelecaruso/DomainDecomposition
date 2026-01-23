#! /usr/bin/python3
from math import pi
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, csc_matrix
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
from utils import mesh, mass, stiffness, plot_mesh, point_source

#Global Variables
na = np.newaxis
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def local_mesh(Lx, Ly, nx, ny):
    # Get MPI parameters
    J = size
    j = rank
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
        phys.append([k * nx + nx - 1,
                     (k + 1) * nx + nx - 1])
    return np.array(phys, dtype=int), np.array(artf, dtype=int)


def Rj_matrix(nx, ny, j, J):
    ny_loc = (ny - 1) // J + 1
    nv_loc = nx * ny_loc
    nv = nx * ny
    #  Define the mapping (Local index i maps to Global index k)
    # Local indices are simply 0, 1, 2... nv_loc-1
    rows = np.arange(0, nv_loc, 1, dtype=np.int64)
    # Global indices are shifted based on the subdomain index 'j'
    # The shift (ny_loc - 1) * nx moves the "window" down the global vector
    cols = rows + j * (ny_loc - 1) * nx
    # Fill the matrix with 1s
    # Each row of Rj has a single '1' at the column corresponding to the global node
    data = np.ones(nv_loc, dtype=np.float64)
    # Create the Sparse Matrix
    # Shape is (local_nodes, global_nodes)
    Rj = csr_matrix((data, (rows, cols)), shape=(nv_loc, nv), dtype=np.float64)
    return Rj

def Cj_matrix(nx, ny, j, J):
    #define size of the matrix Cj which is:
    #the sum of the size of all interfaces x size of one interface
    # for each rank
    ny_loc = (ny - 1) // J + 1
    nv_loc = nx * ny_loc
    sizeCrow = (J - 1) * nx
    sizeCcol = nx if j == 0 or j == J - 1 else 2 * nx   
    rows = np.arange(0, sizeCrow, 1, dtype=np.int64)
    cols = np.zeros(sizeCcol, dtype=np.int64)
    if j == J - 1:
        cols = np.arange(0, nx, 1, dtype=np.int64)
    elif j == 0:
        cols = np.arange((ny_loc - 1) * nx, ny_loc * nx, 1, dtype=np.int64)
    else:
        cols[:nx] = np.arange(0, nx, 1, dtype=np.int64)
        cols[nx:] = np.arange((ny_loc - 1) * nx, ny_loc * nx, 1, dtype=np.int64)
    data = np.ones(sizeCrow, dtype=np.float64)
    Cj = csr_matrix((data, (rows, cols)), shape=(sizeCrow, sizeCcol), dtype=np.float64)
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


def Tj_matrix(vtxj, beltj_artf, Bj, k):
    """
    Constructs the mass matrix reduced to the interface space.
    """
    # Mass matrix on local nodes (nv_loc x nv_loc)
    M_local = mass(vtxj, beltj_artf)

    # Project to interface space: Bj @ M @ Bj.T
    # Resulting Tj is (nbelt_art x nbelt_art)
    Tj_interface = Bj @ M_local @ Bj.T

    return k * Tj_interface


def Sj_factorization(Aj, Tj, Bj):
    """
    Constructs Sj = Aj - i * (Bj.T @ Tj @ Bj) and factorizes it.
    """
    # Expand Tj back to local dimensions (nv_loc x nv_loc)
    # Since Bj is real, Bj.T is sufficient for the adjoint
    Tj_expanded = Bj.T @ Tj @ Bj

    # Assemble the complex local operator
    Sj = Aj - 1j * Tj_expanded

    # Factorize (splu requires CSC format)
    Sj_fact = spla.splu(csc_matrix(Sj))

    return Sj_fact




if __name__ == "__main__":
    np.random.seed(1234)
    Lx = 1
    Ly = 2
    nx = int(1 + Lx * 3)
    ny = int(1 + Ly * 4)

    vtx_loc, elt_loc = local_mesh(Lx, Ly, nx, ny)
    belt_phys, belt_artf = local_boundary(nx, ny, rank, size)
    #Rj = Rj_matrix(nx, ny, rank, size)
    #Bj = Bj_matrix(nx, ny, rank, size, belt_artf)
    Cj = Cj_matrix(nx, ny, rank, size)
    print(f"[Rank {rank}] " f"Vertices: {belt_artf.shape[0]}, "f"tot vtx: {nx * ny}, Bj = \n{Cj.todense()}")
    belt = belt_phys

    print()

    k = 16           # Wavenumber of the problem
    ns = 8           # Number of point sources + random position and weight below
    sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in np.arange(ns)]
    #vtx, elt = mesh(nx, ny, Lx, Ly)
    M = mass(vtx_loc, elt_loc)
    Mb = mass(vtx_loc, belt)
    K = stiffness(vtx_loc, elt_loc)
    A = K - k**2 * M - 1j*k*Mb      # matrix of linear system 
    b = M @ point_source(sp,k)(vtx_loc) # linear system RHS (source term)
    x = spla.spsolve(A, b)          # solution of linear system via direct solver

    # GMRES
    residuals = [] # storage of GMRES residual history
    def callback(x):
        residuals.append(x)
    y, _ = spla.gmres(A, b, rtol=1e-12, callback=callback, callback_type='pr_norm', maxiter=200)
    print("Total number of GMRES iterations = ", len(residuals))
    print("Direct vs GMRES error            = ", la.norm(y - x))

    if( rank == 0 ):
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
