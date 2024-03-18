### Imports ###
import numpy as np
import torch
from scipy.spatial import cKDTree
import os
import itertools
import gc
import pickle
from time import time
import json
import enum


def custom_progress(f : float, a : str = "X", b : str = "x", c : str = "X") -> str:
    return "".join([a] * int(f*10) + [c] + [b] * int((1-f)*10))


class InteractionType(enum.IntEnum):
    ONLY_AB = 0
    STANDARD = 1
    ANGLE = 2
    ANGLE_ISOTROPIC = 3
    NON_INTERACTING = 4


def printfull(x):
    torch.set_printoptions(profile="full")
    print(x) 
    torch.set_printoptions(profile="default") # reset


class Simulation:
    def __init__(self, sim_dict):
        self.device = sim_dict['device']
        self.dtype = sim_dict['dtype']
        self.k = sim_dict['init_k'] 
        self.true_neighbour_max = sim_dict['init_k']//2
        self.dt = sim_dict['dt']
        self.sqrt_dt = np.sqrt(self.dt)
        self.eta = sim_dict['eta']
        self.lambdas = sim_dict['lambdas']
        self.gamma = sim_dict['gamma']
        self.seethru = sim_dict['seethru']
        self.alpha = sim_dict['alpha']
        self.yield_every  = sim_dict['yield_every']
        self.d = None
        self.idx = None

        self.scaled_egg_shape = np.array([60., 60./3., 60./3.]) #TODO: make this a parameter

    @staticmethod
    def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf, workers=-1):
        tree = cKDTree(x)
        d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=workers)
        return d[:, 1:], idx[:, 1:]

    def find_true_neighbours(self, d, dx):
        with torch.no_grad():
            z_masks = []
            i0 = 0
            batch_size = 250
            i1 = batch_size
            while True:
                if i0 >= dx.shape[0]:
                    break

                n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
                n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]

                z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= self.seethru
                z_masks.append(z_mask)

                if i1 > dx.shape[0]:
                    break
                i0 = i1
                i1 += batch_size
        z_mask = torch.cat(z_masks, dim=0)
        return z_mask
    

    
    
    def calculate_interaction(self, dx, p, q, p_mask, idx):
        # Making interaction mask
        interaction_mask = p_mask[:,None].expand(p_mask.shape[0], idx.shape[1]) + p_mask[idx]

        # Calculate S
        l00, l01, l1, l2, l3 = self.lambdas

        pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
        pj = p[idx]
        qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
        qj = q[idx]

        lam = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1], 4),
                        device=self.device)                                                            # Initializing an empty array for our lambdas
        
        # lam[interaction_mask == 0] = torch.tensor([l00,0,0,0], device=self.device)                     # Setting lambdas for non polar interaction
        # lam[interaction_mask == 1] = torch.tensor([l01,0,0,0], device=self.device)                     # Setting lambdas for polar-nonpolar interaction
        # lam[interaction_mask == 2] = torch.tensor([0,l1,l2,l3], device=self.device)                    # Setting lambdas for pure polar interaction
        lam[:,:] = torch.tensor([0,l1,l2,l3], device=self.device)  

        alphas = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1]), 
                        device=self.device)   # Initializing an empty array for our alphas
        # alphas[interaction_mask == 0] = 0.5                                                             # Setting alphas for non polar interaction
        # alphas[interaction_mask == 1] = 0                                                               # Setting alphas for polar-nonpolar interaction

        # printfull(interaction_mask)

        pi_tilde = pi - self.alpha*dx
        pj_tilde = pj + self.alpha*dx

        pi_tilde = pi_tilde/torch.sqrt(torch.sum(pi_tilde ** 2, dim=2))[:, :, None]                           # The p-tildes are normalized
        pj_tilde = pj_tilde/torch.sqrt(torch.sum(pj_tilde ** 2, dim=2))[:, :, None]

        S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2)            # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
        S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)                        # Calculating S2 (The ABP-PCP part of S).
        S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                        # Calculating S3 (The PCP-position part of S)

        S = lam[:,:,1] * S1 + lam[:,:,2] * S2 + lam[:,:,3] * S3

        return S
    

    def inv_make_even_better_egg_pos(self, pos):
        # with torch.no_grad():

        return_pos = pos.clone()
        xx, yy, zz = pos[:,0], pos[:,1], pos[:,2]

        z_add = torch.square(torch.abs(xx/self.scaled_egg_shape[0]))*self.scaled_egg_shape[2]/2

        z_add = torch.where(xx < 0, z_add, z_add/2.)

        return_pos[:,2] -=  z_add
        
        x_sub = torch.where(xx < 0, 0, self.scaled_egg_shape[0]/3.*xx/self.scaled_egg_shape[0])
        
        return_pos[:,0] += x_sub

        return return_pos
    
    def egg_BC(self, pos):
        # with torch.no_grad():
        corrected_pos_vec = self.inv_make_even_better_egg_pos(pos)
        corrected_pos_vec = corrected_pos_vec / self.scaled_egg_shape[None, :]
        mag = torch.sum(corrected_pos_vec**2, dim=1)
            
        v_add = torch.where(mag > 1.0, (torch.exp(mag*mag - 1) - 1), 0.)

        # make sure the number is not too large

        # print(v_add.cpu().detach().numpy())
        
        
        v_add = torch.where(v_add > 10., 10., v_add)

        
        v_add = torch.sum(v_add)
        
        return v_add 
    
    def potential(self, x, p, q, p_mask, idx, d):
        # Find neighbours
        full_n_list = x[idx]
        dx = x[:, None, :] - full_n_list
        z_mask = self.find_true_neighbours(d, dx)

        # Minimize size of z_mask and reorder idx and dx
        sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)
        z_mask = torch.gather(z_mask, 1, sort_idx)
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
        idx = torch.gather(idx, 1, sort_idx)
        m = torch.max(torch.sum(z_mask, dim=1)) + 1
        z_mask = z_mask[:, :m]
        dx = dx[:, :m]
        idx = idx[:, :m]

        # Normalise dx
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        dx = dx / d[:, :, None]


        # Calculate potential
        S = self.calculate_interaction(dx, p, q, p_mask, idx)

        # bc_contrib = torch.zeros_like(S)


        # # bc_contrib[:, 0] = egg_bc


        Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d/5))

        
        bc = self.egg_BC(x)
        
        Vij_sum = torch.sum(Vij) + bc 

        return Vij_sum, int(m) # TODO print m
        
    def init_simulation(self, x, p, q, p_mask):
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(p_mask) == len(x)

        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
        p_mask = torch.tensor(p_mask, dtype=torch.int, device=self.device)
        self.beta   = torch.zeros_like(p_mask)

        self.scaled_egg_shape = torch.tensor(self.scaled_egg_shape, dtype=self.dtype, device=self.device)


        return x, p, q, p_mask

    def update_k(self, true_neighbour_max):
        return 25
        k = self.k
        fraction = true_neighbour_max / k                                                         # Fraction between the maximimal number of nearest neighbors and the initial nunber of nearest neighbors we look for.
        if fraction < 0.25:                                                                       # If fraction is small our k is too large and we make k smaller
            k = int(0.75 * k)
        elif fraction > 0.75:                                                                     # Vice versa
            k = int(1.5 * k)
        self.k = k                                                                                # We update k
        print(k)
        return k # TODO: maybe set to set number
    
    def should_update_neighbors_bool(self, tstep): ## TODO: maybe remove
        if self.idx is None:
            return True


        return (tstep % 50 == 0)

    def time_step(self, x, p, q, p_mask, tstep):
        # Start with cell division
        gc.collect()
        

        # Idea: only update _potential_ neighbours every x steps late in simulation
        # For now we do this on CPU, so transfer will be expensive
        k = self.update_k(self.true_neighbour_max)
        k = min(k, len(x) - 1)

        if self.should_update_neighbors_bool(tstep):
            d, idx = self.find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
            self.idx = torch.tensor(idx, dtype=torch.long, device=self.device)
            self.d = torch.tensor(d, dtype=self.dtype, device=self.device)
        idx = self.idx
        d = self.d

        # Normalise p, q
        with torch.no_grad():
            p /= torch.sqrt(torch.sum(p ** 2, dim=1))[:, None]          # Normalizing p. Only the non-zero polarities are considered.
            q /= torch.sqrt(torch.sum(q ** 2, dim=1))[:, None]          # Normalizing q. Only the non-zero polarities are considered.


        # Calculate potential
        V, self.true_neighbour_max = self.potential(x, p, q, p_mask, idx, d)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            x += -x.grad * self.dt + self.eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            p += -p.grad * self.dt + self.eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            q += -q.grad * self.dt + self.eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

            p.grad.zero_()
            q.grad.zero_()

        x.grad.zero_()


        return x, p, q, p_mask

    # def simulation(self, x, p, q, lam, beta, eta, potential=potential, yield_every=1, dt=0.1):
    def simulation(self, x, p, q, p_mask):
        
        x, p, q, p_mask = self.init_simulation(x, p, q, p_mask)

        tstep = 0
        while True:
            tstep += 1
            x, p, q, p_mask = self.time_step(x, p, q, p_mask, tstep)

            large_mask = x > 10000
            # if large_mask.any():
            #     printfull(x)
            #     print("LARGE")
            #     # break
            
            nan_mask = torch.isnan(x)
            if nan_mask.any():
                printfull(x)
                print("NAN")
                break
                
            if tstep % self.yield_every == 0:
                xx = x.detach().to("cpu").numpy().copy()
                pp = p.detach().to("cpu").numpy().copy()
                qq = q.detach().to("cpu").numpy().copy()
                pp_mask = p_mask.detach().to("cpu").numpy().copy()
                yield xx, pp, qq, pp_mask

import h5py

def save(data_tuple, name, sim_dict):
    # with open(f'runs/{name}.npy', 'wb') as f:
    #     pickle.dump(data_tuple, f)
    p_mask_lst, x_lst, p_lst,  q_lst = data_tuple

    def append_or_create_h5py(name, data, file):
        if not name in file:
            file.create_dataset(name, data=data)
        else:
            del file[name]
            file.create_dataset(name, data=data)

    with h5py.File("runs/"+name+".hdf5", "a") as f:
            append_or_create = lambda name, data: append_or_create_h5py(name, data, f)

            append_or_create("x", np.array(x_lst))

            append_or_create("properties", np.array(p_mask_lst))

            append_or_create("p", np.array(p_lst))

            append_or_create("q", np.array(q_lst))

            # f.attrs.update(sim_dict)

def run_simulation(sim_dict):
    # Make the simulation runner object:
    continue_from = sim_dict.pop('continue')
    yield_steps, yield_every = sim_dict['yield_steps'], sim_dict['yield_every']


    name = sim_dict.pop('name')

    # assert len(data_tuple) == 4 or len(data_tuple) == 2, 'data must be tuple of either len 2 (for data generation) or 4 (for data input)'

    # if len(data_tuple) == 4:
    #     p_mask, x, p, q = data_tuple
    # else:
    #     data_gen = data_tuple[0]
    #     p_mask, x, p, q = data_gen(*data_tuple[1])

    assert continue_from + ".hdf5" in os.listdir('runs'), 'continue_from must be a valid run in /runs/'
    
    with h5py.File("runs/"+continue_from+".hdf5", "r") as f:
        x = f['x'][-1]
        p = f['p'][-1]
        q = f['q'][-1]
        p_mask = np.array(f['properties'])
        


    sim = Simulation(sim_dict)
    runner = sim.simulation(x, p, q, p_mask)

    x_lst = [x]
    p_lst = [p]
    q_lst = [q]
    p_mask_lst = [p_mask]


    save((p_mask_lst, x_lst, p_lst,  q_lst), name=name, sim_dict=sim_dict)


    notes = sim_dict['notes']
    print('Starting simulation with notes:')
    print(notes)

    i = 0
    t1 = time()
    
    for xx, pp, qq, pp_mask in itertools.islice(runner, yield_steps):
        i += 1
        ss = custom_progress(i/yield_steps, "😊  ", "😔  ", "😮  ")
        print(ss + f'  Running {i} of {yield_steps}   ({yield_every * i} of {yield_every * yield_steps})   ({len(xx)} cells)', end = "\r")

        x_lst.append(xx)
        p_lst.append(pp)
        q_lst.append(qq)
        # p_mask_lst.append(pp_mask)
        

        if i % 100 == 0:
            save((p_mask_lst, x_lst, p_lst,  q_lst), name=name, sim_dict=sim_dict)

    print(f'Simulation done, saved {yield_steps} datapoints')
    print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst), name=name, sim_dict=sim_dict)
