from typing import Any
import jax.numpy as jnp
import numpy as np
import jax.random as random
import os
from utils import get_random_points_on_sphere
import h5py
# from G import G


class InitialConditions():
    def __init__(self, G : dict):
        self.type = type
        self.N = G["N_cells"]

        self.scale = G["IC_scale"]

        if ":" in G["IC_type"]:
            t, c = G["IC_type"].split(":")
            assert t == "continue", "Only continue is allowed to have ':continue_from'"
            self.type, self.continue_from = G["IC_type"].split(":")
            assert os.path.isfile("runs/" + self.continue_from + ".hdf5"), f"File '{self.continue_from}' does not exist"

        else:
            self.type = G["IC_type"]

        self.egg_shape = jnp.array([1., 1./3., 1./3.])

        self.scaled_egg_shape = self.egg_shape * self.scale

        self.sphere_shape = jnp.array([1., 1., 1.])

        self.scaled_sphere_shape = self.sphere_shape * self.scale

    def initialize(self,) -> Any:
        return self.get_ic(self.type)(self.N)
    
    def continue_IC(self, *args):
        assert self.continue_from is not None, "No file to continue from"
        assert os.path.isfile("runs/" + self.continue_from + ".hdf5"), "File does not exist"

        with h5py.File("runs/" + self.continue_from + ".hdf5", 'r') as f:
            cells = f['cells'][-1]
            properties = f['properties'][:]
        # cells = np.load(path)[-1].reshape(-1, 3,3)

        # if os.path.isfile(path[:-4] + '_properties.npy'):
        #     properties = np.load(path[:-4] + '_properties.npy')
        # else:
        #     properties = np.zeros(cells.shape[0])


        return cells, properties

    def make_better_egg_pos(self, cells):
        cells = cells.at[2].set(cells[2] + jnp.square(jnp.abs(cells[0]/self.scaled_egg_shape[0]))*self.scaled_egg_shape[2]/2)
        return cells

    def inv_make_better_egg_pos(self, cells):
        cells = cells.at[2].set(cells[2] - jnp.square(jnp.abs(cells[0]/self.scaled_egg_shape[0]))*self.scaled_egg_shape[2]/2)
        return cells

    def make_better_egg(self, cells):
        cells = cells.at[:,0,2].set(cells[:,0,2] + jnp.square(jnp.abs(cells[:,0,0]/self.scaled_egg_shape[0]))*self.scaled_egg_shape[2]/2)
        return cells

    def egg_IC(self, N : int) -> jnp.ndarray:
        pos = get_random_points_on_sphere(N, 42)

        p = pos #/ jnp.linalg.norm(pos, axis=1, keepdims=True)

        # make them all point in the same direction, whirl around the x axis
        q = pos.copy()
        q = q.at[:,0].set(0)
        q = q / jnp.linalg.norm(q, axis=1, keepdims=True)
        _y = q[:,1]
        _z = q[:,2]
        q = q.at[:,1].set(-_z)
        q = q.at[:,2].set(_y)

        pos = pos * self.scaled_egg_shape

        cells = jnp.stack([pos, p, q], axis=1)

        cell_properties = jnp.ones(N)

        return cells, cell_properties

    def egg_half_IC(self, N : int) -> jnp.ndarray:
        cells, _ = self.egg_IC(N)
        cell_properties = jnp.where(cells[:,0,:][:,2] < 0, 0., 1.)

        return cells, cell_properties

    def egg_genius(self, N:int) -> jnp.ndarray:
        cells, _ = self.egg_IC(N)

        cell_properties = jnp.where(cells[:,0,:][:,2] < self.scaled_egg_shape[2]/3, 0., 1.)

        cell_properties = jnp.where(cells[:,0,:][:,0] < -self.scaled_egg_shape[0]/2, 1., cell_properties)

        return cells, cell_properties

    # def egg_genius_(self, N:int) -> jnp.ndarray:
    #     cells, _ = egg_IC(N)

    #     cell_properties = jnp.where(cells[:,0,:][:,2] < egg_shape[2]/3, 0., 1.)

    #     cell_properties = jnp.where(cells[:,0,:][:,0] < -egg_shape[0]/2, 1., cell_properties)

    #     return cells, cell_properties

    def better_egg(self, N:int) -> jnp.ndarray:
        cells, cell_properties = self.egg_IC(N)

        cells = self.make_better_egg(cells)
        return cells, cell_properties

    def better_egg_genius(self, N:int) -> jnp.ndarray:
        cells, cell_properties = self.egg_IC(N)

        cells = self.make_better_egg(cells)
        cell_properties = jnp.where(cells[:,0,:][:,2] < self.egg_shape[2]/2, 0., 1.)

        cell_properties = jnp.where(cells[:,0,:][:,0] < -self.egg_shape[0]/2, 1., cell_properties)

        return cells, cell_properties


    def plane_IC(self, N : int) -> jnp.ndarray:
        # pos = get_random_points_on_sphere(N, 42)
        pos = random.normal(random.PRNGKey(42), (N, 3))*jnp.sqrt(N)/2.

        pos = pos.at[:,2].set(jnp.zeros(N))
        # pinting in the z direction
        p = jnp.array([0, 0, 1.0])[None, :].repeat(N, axis=0)

        q = jnp.array([0, 1.0, 0])[None, :].repeat(N, axis=0)


        cells = jnp.stack([pos, p, q], axis=1)

        cell_properties = jnp.zeros(N)


        return cells, cell_properties


    def get_ic(self, type:str) -> jnp.ndarray:
        if type == "egg":
            return self.egg_IC
        elif type == "plane":
            return self.plane_IC
        elif type == "egg_half":
            return self.egg_half_IC
        elif type == "egg_genius":
            return self.egg_genius
        elif type == "better_egg":
            return self.better_egg
        elif type == "continue":
            return self.continue_IC
        elif type == "better_egg_genius":
            return self.better_egg_genius
        
        else:
            raise ValueError("type must be egg, plane or egg_half")
