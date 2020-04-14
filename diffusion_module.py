import numpy as np
import time
from multiprocessing import Pool
from functools import partial

def diffusion(D, dt, dx, u):
    """ compute one time step of diffusion
    D: diffusion constant
    dt: step size in time
    dx: step size in space
    u: concentration on lattice points of diffusing quantity
    return: u after one time step """
    
    tmp = np.empty((u.shape[0] - 2, u.shape[1] - 2))
    for x in np.arange(1, u.shape[0] - 1):
        for y in np.arange(1, u.shape[1] - 1):
            tmp[x - 1, y - 1] = u[x, y] + dt * D * (u[x - 1, y] + u[x, y - 1] - 4 * u[x, y] + 
                                                    u[x + 1, y] + u[x, y + 1]) / dx**2
    return tmp


def parallel(grid, workers, gridsize, param):
    """ compute one time step of diffusion
    grid: grid at time t=0
    workers: number of processes, typically to be matched with CPU cores to disposition
    gridsize: two element list containing x and y gridsize in number of lattice sites
    param: several parameters defining the diffusion process 
      - "dt": integration step in time
      - "dx": integration step in space
      - "T": integration time
      - "D": diffusion constant
    return: grid after time T """
    
    # define integration step sizes
    dx = param["dx"]
    dt = param["dt"]

    T = param["T"]   # integration time
    D = param["D"]   # diffusion constant
    
    # define number of processes
    p = Pool(workers)

    # define how many partitions of grid in x and y direction and their length
    (nx, ny) = (int(workers / 2), 2)
    lx = int(gridsize[0] / nx)
    ly = int(gridsize[1] / ny)

    # this makes sure that D, dt, dx are the same when distributed over processes
    # for integration, so the only interface parameter that changes is the grid
    func = partial(diffusion, D, dt, dx)
    ts = time.time()  # measure computation time
    for t in np.arange(T/dt):  # note numpy.arange is rounding up floating points
        data = []
        # prepare data to be distributed among workers
        # 1. insert boundary conditions and partition data
        grid = np.insert(grid, 0, grid[0, :], axis=0)       # top
        grid = np.vstack((grid, grid[-1, :]))               # bottom
        grid = np.insert(grid, 0, grid[:, 0], axis=1)       # left
        grid = np.hstack((grid, np.array([grid[:, -1]]).T))   # right
        # partition into subgrids
        for i in range(nx):
            for j in range(ny):
                # subgrid
                subg = grid[i * lx + 1:(i+1) * lx + 1, j * ly + 1:(j+1) * ly + 1]
                subg = np.insert(subg, 0, grid[i * lx, j * ly + 1:(j+1) * ly + 1], axis=0)  # upper subgrid boundary
                subg = np.vstack((subg, grid[(i+1) * lx + 1, j * ly + 1:(j+1) * ly + 1]))  # lower subgrid boundary
                subg = np.insert(subg, 0, grid[i * lx:(i+1) * lx + 2, j * ly], axis=1)  # left subgrid boundary
                subg = np.hstack((subg, np.array([grid[i * lx:(i+1) * lx + 2, (j+1) * ly + 1]]).T))  # right subgrid boundary
                # collect subgrids in list to be distributed over processes
                data.append(subg)
        # 2. divide among workers
        results = p.map(func, data)
        grid = np.vstack([np.hstack((results[i * ny:(i+1) * ny])) for i in range(nx)])
    print('Parallel processing took {}s'.format(time.time() - ts)) # alternative to write variable to string as used above
    return grid


def sequential(grid, param):
    """ compute one time step of diffusion
    grid: grid at time t=0
    param: several parameters defining the diffusion process 
      - "dt": integration step in time
      - "dx": integration step in space
      - "T": integration time
      - "D": diffusion constant
    return: grid after time T """
    
    # define integration step sizes
    dx = param["dx"]
    dt = param["dt"]

    T = param["T"]   # integration time
    D = param["D"]   # diffusion constant
    
    ts = time.time()  # measure computation time
    for t in np.arange(T/dt):
        # insert upper and lower boundary: reflecting boundary
        tmp = np.insert(grid, 0, grid[0, :], axis=0)
        tmp = np.vstack((tmp, grid[-1, :]))
        # insert left and right boundary: reflecting boundary
        tmp = np.insert(tmp, 0, tmp[:, 0], axis=1)
        tmp = np.hstack((tmp, np.array([tmp[:, -1]]).T)) # note: slicing gives a row vector therefore transpose to get column vector
        grid = diffusion(D, dt, dx, tmp)
    print('Sequential processing took {}s'.format(time.time() - ts))
    return grid