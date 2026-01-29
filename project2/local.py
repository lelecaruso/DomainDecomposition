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

def bj_vector(vtx_loc, elt_loc, sp, k):
    return mass(vtx_loc, elt_loc) @ point_source(sp, k)(vtx_loc)

