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

na = np.newaxis
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def mesh(nx,ny,Lx,Ly):
   i = np.arange(0,nx)[na,:] * np.ones((ny,1), np.int64)
   j = np.arange(0,ny)[:,na] * np.ones((1,nx), np.int64)
   p = np.zeros((2,ny-1,nx-1,3), np.int64)
   q = i+nx*j
   p[:,:,:,0] = q[:-1,:-1]
   p[0,:,:,1] = q[1: ,1: ]
   p[0,:,:,2] = q[1: ,:-1]
   p[1,:,:,1] = q[:-1,1: ]
   p[1,:,:,2] = q[1: ,1: ]
   v = np.concatenate(((Lx/(nx-1)*i)[:,:,na], (Ly/(ny-1)*j)[:,:,na]), axis=2)
   vtx = np.reshape(v, (nx*ny,2))
   elt = np.reshape(p, (2*(nx-1)*(ny-1),3))
   return vtx, elt 




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

def Bj_matrix(nx, ny, j, J, belt_artf):
    ny_loc = (ny - 1) // J + 1
    nv_loc = nx * ny_loc
    nbelt = belt_artf.shape[0]
    rows = np.arange(0, nbelt, 1, dtype=np.int64)
    cols = rows + j * (ny_loc - 1) * nx
    data = np.ones(nbelt, dtype=np.float64)
    Bj = csr_matrix((data, (rows, cols)), shape=(nbelt, nv_loc), dtype=np.float64)
    return Bj

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
    if j == 0:
        cols = np.arange(0, nx, 1, dtype=np.int64)
    elif j == J - 1:
        cols = np.arange((ny_loc - 1) * nx, ny_loc * nx, 1, dtype=np.int64)
    else:
        cols[:nx] = np.arange(0, nx, 1, dtype=np.int64)
        cols[nx:] = np.arange((ny_loc - 1) * nx, ny_loc * nx, 1, dtype=np.int64)
    
    data = np.ones(sizeCrow, dtype=np.float64)
    Cj = csr_matrix((data, (rows, cols)), shape=(sizeCrow, sizeCcol), dtype=np.float64)
    return Cj

def boundary(nx, ny):
    bottom = np.hstack((np.arange(0,nx-1,1)[:,na],
                        np.arange(1,nx,1)[:,na]))
    top    = np.hstack((np.arange(nx*(ny-1),nx*ny-1,1)[:,na],
                        np.arange(nx*(ny-1)+1,nx*ny,1)[:,na]))
    left   = np.hstack((np.arange(0,nx*(ny-1),nx)[:,na],
                        np.arange(nx,nx*ny,nx)[:,na]))
    right  = np.hstack((np.arange(nx-1,nx*(ny-1),nx)[:,na],
                        np.arange(2*nx-1,nx*ny,nx)[:,na]))
    return np.vstack((bottom, top, left, right))

def get_area(vtx, elt):
    d = np.size(elt, 1)
    if d == 2:
        e = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        areas = la.norm(e, axis=1)
    else:
        e1 = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        e2 = vtx[elt[:, 2], :] - vtx[elt[:, 0], :]
        areas = 0.5 * np.abs(e1[:,0] * e2[:,1] - e1[:,1] * e2[:,0])
    return areas

def mass(vtx, elt):
    nv = np.size(vtx, 0)
    d = np.size(elt, 1)
    areas = get_area(vtx, elt)
    M = csr_matrix((nv, nv), dtype=np.float64)
    for j in range(d):
        for k in range(d):
           row = elt[:,j]
           col = elt[:,k]
           val = areas * (1 + (j == k)) / (d*(d+1))
           M += csr_matrix((val, (row, col)), shape=(nv, nv))
    return M

def stiffness(vtx, elt):
    nv = np.size(vtx, 0)
    d = np.size(elt, 1)
    areas = get_area(vtx, elt)
    ne, d = np.shape(elt)
    E = np.empty((ne, d, d-1), dtype=np.float64)
    E[:,0,:] = 0.5 * (vtx[elt[:,1],0:2] - vtx[elt[:,2],0:2])
    E[:,1,:] = 0.5 * (vtx[elt[:,2],0:2] - vtx[elt[:,0],0:2])
    E[:,2,:] = 0.5 * (vtx[elt[:,0],0:2] - vtx[elt[:,1],0:2])
    K = csr_matrix((nv, nv), dtype=np.float64)
    for j in range(d):
        for k in range(d):
           row = elt[:,j]
           col = elt[:,k]
           val = np.sum(E[:,j,:] * E[:,k,:], axis=1) / areas
           K += csr_matrix((val, (row, col)), shape=(nv, nv))
    return K

def point_source(sp, k):    
    def ps(x):
        v = np.zeros(np.size(x,0), float)
        for s in sp:
            v += s[2]*np.exp(-10*(k/(2.0*pi))**2 * la.norm(x - s[na,0:2], axis=1)**2)
        return v
    return ps 

def plot_mesh(vtx, elt, val=None, **kwargs):
    trig = mtri.Triangulation(vtx[:, 0], vtx[:, 1], elt)
    if val is None:
        plt.triplot(trig, **kwargs)
    else:
        plt.tripcolor(
            trig, val,
            shading='gouraud',
            cmap=cm.jet,
            **kwargs
        )
    plt.axis('equal')







if __name__ == "__main__":
    np.random.seed(1)
    Lx = 1
    Ly = 2
    nx = int(1 + Lx * 32)
    ny = int(1 + Ly * 32)

    vtx_loc, elt_loc = local_mesh(Lx, Ly, nx, ny)
    belt_phys, belt_artf = local_boundary(nx, ny, rank, size)
    Rj = Rj_matrix(nx, ny, rank, size)
    belt = belt_phys

    print(f"[Rank {rank}] " f"Vertices: {vtx_loc.shape[0]}, "f"Elements: {elt_loc.shape[0]}")

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
