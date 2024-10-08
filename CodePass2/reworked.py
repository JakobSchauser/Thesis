### Imports ###
import numpy as np
import torch
from scipy.spatial import cKDTree, Voronoi
import os
import itertools
import gc
import pickle
from time import time
import json
import enum


def custom_progress(f : float, a : str = "X", b : str = "x", c : str = None) -> str:
    return "".join([a] * int(np.floor(f*10)) + [c if c is not None else a] + [b]*(10 - int(np.floor((f*10)))))


# class InteractionType(enum.IntEnum):
#     ONLY_AB = 0
#     STANDARD = 1
#     ANGLE = 2
#     ANGLE_ISOTROPIC = 3
#     NON_INTERACTING = 4

def lerp(v, d):
    return v[0] * (1 - d) + v[1] * d
    
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
        self.interaction_data = sim_dict['interaction_data']
        self.gamma = sim_dict['gamma']
        self.seethru = sim_dict['seethru']
        self.yield_every  = sim_dict['yield_every']
        self.d = None
        self.idx = None

        self.alpha_scale = 1.
        self.alpha_scale_speed = None
        
        if "alpha_scale_speed" in sim_dict:
            self.alpha_scale_speed = sim_dict.pop('alpha_scale_speed')
            if not self.alpha_scale_speed is None:
                self.alpha_scale = 0.

        self.start_xs = None

        # self.scaled_egg_shape = np.array([60., 60./3., 60./3.]) #TODO: make this a parameter
        self.scaled_egg_shape = np.array([80., 80./3., 80./3.])

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
    

    def get_lambds_from_interaction_mask(self, interaction_mask, interaction_mask_b, lam, alphas):
        # ll = 0.5*self.lambdas[0] # used to be 0.5
        # ll[2] = 0
        
        # lam[(interaction_mask == 0) * (interaction_mask_b == 1)] = ll
        # lam[(interaction_mask == 1) * (interaction_mask_b == 0)] = ll
        nnnn = 0.8
        
        lam[(interaction_mask == 1) * (interaction_mask_b == 2)] = nnnn*self.lambdas[1]
        lam[(interaction_mask == 2) * (interaction_mask_b == 1)] = nnnn*self.lambdas[2]

        lam[(interaction_mask == 0) * (interaction_mask_b == 2)] = nnnn*self.lambdas[0]
        lam[(interaction_mask == 2) * (interaction_mask_b == 0)] = nnnn*self.lambdas[2]

        lam[(interaction_mask == 0) * (interaction_mask_b == 5)] = 0.5*self.lambdas[0]
        lam[(interaction_mask == 5) * (interaction_mask_b == 0)] = 0.5*self.lambdas[5]

        lam[(interaction_mask == 1) * (interaction_mask_b == 5)] = 0.5*self.lambdas[1]
        lam[(interaction_mask == 5) * (interaction_mask_b == 1)] = 0.5*self.lambdas[5]

        lam[(interaction_mask == 2) * (interaction_mask_b == 5)] = 0.5*self.lambdas[2]
        lam[(interaction_mask == 5) * (interaction_mask_b == 2)] = 0.5*self.lambdas[5]
        
        lam[(interaction_mask == 0) * (interaction_mask_b == 4)] = 0.9*self.lambdas[0]
        lam[(interaction_mask == 4) * (interaction_mask_b == 0)] = 0.9*self.lambdas[4]
        
        lam[(interaction_mask == 1) * (interaction_mask_b == 4)] = 0.9*self.lambdas[1]
        lam[(interaction_mask == 4) * (interaction_mask_b == 1)] = 0.9*self.lambdas[4]

        lam[(interaction_mask == 0) * (interaction_mask_b == 3)] = 0.6*self.lambdas[0]
        lam[(interaction_mask == 3) * (interaction_mask_b == 0)] = 0.6*self.lambdas[0]

        lam[(interaction_mask == 1) * (interaction_mask_b == 3)] = 0.8*self.lambdas[1]
        lam[(interaction_mask == 3) * (interaction_mask_b == 1)] = 0.8*self.lambdas[1]

        lam[(interaction_mask == 4) * (interaction_mask_b == 3)] = 0.9*self.lambdas[4]
        lam[(interaction_mask == 3) * (interaction_mask_b == 4)] = 0.4*self.lambdas[3]
        
        lam[(interaction_mask == 2) * (interaction_mask_b == 4)] = 0.5*self.lambdas[2]
        lam[(interaction_mask == 4) * (interaction_mask_b == 2)] = 0.5*self.lambdas[4]


        alphas[(interaction_mask == 0)*(interaction_mask_b == 4)] = -self.alphas[4]
        alphas[(interaction_mask == 0)*(interaction_mask_b == 4)] = -self.alphas[4]

        return lam, alphas
    
    def calculate_interaction(self, dx, p, q, p_mask, idx, d, x):
        # Making interaction mask
        interaction_mask = p_mask[:,None].expand(p_mask.shape[0], idx.shape[1])#*5 + p_mask[idx]
        interaction_mask_b = p_mask[idx]
        # Calculate S
        # l00, l01, l1, l2, l3 = self.lambdas

        pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
        pj = p[idx]
        qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
        qj = q[idx]

        lam = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1], 4),
                        device=self.device)                                                            # Initializing an empty array for our lambdas
        original_lams = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1], 4),
                device=self.device)                                                            # Initializing an empty array for our lambdas
        alphas = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1], 3), 
                        device=self.device)   # Initializing an empty array for our alphas

        for k in range(len(self.interaction_data)):
            a_s = 1.
            lk_s = self.lambdas[k].clone()
            
            with torch.no_grad():
                if k in (2, 999):
                    sss = np.maximum((self.alpha_scale - 0.25), 0.)
                    a_s = sss + 1.
                    # lk_s[1] *= 1.-sss
                    # lk_s[3] = sss
                    pass

            lam[interaction_mask == k] = lk_s
            alphas[interaction_mask == k] = self.alphas[k]*a_s
            original_lams[interaction_mask == k] = self.lambdas[k]


        lam, alphas = self.get_lambds_from_interaction_mask(interaction_mask, interaction_mask_b, lam, alphas)


        if not self.alpha_scale_speed is None and self.alpha_scale < 0.9901:
            self.alpha_scale = lerp((self.alpha_scale, 1.), self.alpha_scale_speed)
            
            if self.alpha_scale >= 0.99:
                print("alpha scale = 1!")
        

        with torch.no_grad():
            nl3 = (x[:,2] + 27)/(2*27)
            # nl3 = (self.start_xs[:,2] + 27)/(2*27)
            # nl3 = torch.sqrt(torch.clamp(1.-nl3*2.,0., 1.))*0.1
            nl3 = (1-nl3)*(1-nl3)*0.15

            # nl3 = torch.where(torch.logical_and(self.start_xs[:,0] > -60, self.start_xs[:,0] < 50), nl3, 0.)
            lam[:,:,3] = 0.
            lam[:,:,3] = torch.where(original_lams[:,:,3] == -1, nl3[:,None], 0.)
            lam[:,:,1] = torch.where(original_lams[:,:,3] == -1, lam[:,:,1] - 1.*nl3[:,None], lam[:,:,1])



        # print(lam[:,:,3].max()) 
        # print(lam[:,:,3].min()) 
        
        # lam[:,:,3] = nl3[:,None]
        # lam[:,:,0] = lam[:,:,0] - nl3[:,None]

        # lam[interaction_mask == 2][:,3] = 0.
        # lam[interaction_mask_b == 2][:,3] = 0.
        # lam[interaction_mask == 3][:,3] = 0.
        # lam[interaction_mask == 4][:,3] = 0.

        # areas = self.get_projectioned_areas(x, p, q, interaction_mask, dx)

        # print(areas)

        
        # angle_dx = dx
        avg_q = (qi + qj)*0.5
        ts = (avg_q*dx).sum(axis = 2)

        avg_p = (pi+pj)*0.5
        perps = torch.cross(avg_q, avg_p, dim=2)
        ts2 = (perps*dx).sum(axis = 2)

        angle_dx = torch.where(interaction_mask[:,:,None] == 2, avg_q*ts[:,:,None], dx)
        angle_dx = torch.where(interaction_mask[:,:,None] == 5, perps*ts2[:,:,None], angle_dx)
        # angle_dx = torch.where(interaction_mask[:,:,None] == 6, perps*ts2[:,:,None], angle_dx)

        # angle_dx = angle_dx*d[:,:,None]
        
        pi_tilde = pi + alphas*angle_dx
        pj_tilde = pj - alphas*angle_dx

        pi_tilde = pi_tilde/torch.sqrt(torch.sum(pi_tilde ** 2, dim=2))[:, :, None]                           # The p-tildes are normalized
        pj_tilde = pj_tilde/torch.sqrt(torch.sum(pj_tilde ** 2, dim=2))[:, :, None]

        # S0 = (d - 2.)/10.
        S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2)            # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
        S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)                        # Calculating S2 (The ABP-PCP part of S).
        S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                        # Calculating S3 (The PCP-position part of S)

        # msk = torch.sum(pj_tilde * pi_tilde, dim = 2) < -0.0
        # S1[msk] = 0.
        
        
        S = lam[:,:,1] * S1 + lam[:,:,2] * torch.abs(S2) + lam[:,:,3] * torch.abs(S3)

        return S, ts, ts2
    

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
            
        v_add = torch.where(mag > 1.0, (torch.exp(mag*mag - 1) - 1)*10., 0.)

        # make sure the number is not too large

        # print(v_add.cpu().detach().numpy())
        
        
        v_add = torch.where(v_add > 50., 50., v_add)

        
        v_add = torch.sum(v_add)
        
        return v_add 
    
        

    # def get_projectioned_areas(self, x, p, q, z_mask, dx):
    #     # for each neighbour in z_mask 
    #     areas = []
    #     for point in range(x.shape[0]):
    #         nbs = z_mask[point]

    #         x_vec = q[point]
    #         y_vec = torch.cross(p[point], q[point])

    #         x_vec = x_vec / torch.sqrt(torch.sum(x_vec ** 2))
    #         y_vec = y_vec / torch.sqrt(torch.sum(y_vec ** 2))

    #         projected_positions = []
    #         # project their positions onto the plane defined by the p vector
    #         for nb in range(len(nbs)):
    #             if dx[point][nb] > 10:
    #                 continue
    #             # pp_3d = nbs[nb] - torch.dot(nbs[nb] - x[point], p[point])*p[point]

    #             # project the position onto the plane defined by the q vector
    #             pp_2d = torch.tensor([torch.dot(nbs, x_vec), torch.dot(nbs, y_vec)])

    #             projected_positions.append(pp_2d)

    #         vor_vertices = Voronoi(projected_positions).vertices

    #         # use shoe-lace formula to calculate the area of the voronoi cell
    #         area = 0
    #         for i in range(len(vor_vertices)):
    #             area += vor_vertices[i][0]*vor_vertices[(i+1)%len(vor_vertices)][1] - vor_vertices[(i+1)%len(vor_vertices)][0]*vor_vertices[i][1]
    #         areas.append(area/2)

    #     return torch.tensor(areas, device=self.device, dtype=self.dtype)

            
    
    def potential(self, x, p, q, p_mask, idx, d):
        # Find neighbours
        full_n_list = x[idx]

        # define the effective centrum
        o = torch.cross(p, q)
        o = o / torch.sqrt(torch.sum(o ** 2, dim=1))[:, None]

        nb_o = o[idx]

        # define the pairwise effective centrum
        dx_normal = x[:, None, :] - full_n_list
        dx_normal_lengths = torch.sqrt(torch.sum(dx_normal ** 2, dim=2))
        dx_normal = dx_normal / dx_normal_lengths[:, :, None]


        ec_add = torch.cross(o[:,None,:], dx_normal, dim=2) * torch.cross(nb_o, dx_normal, dim=2) 

        ec_add = ec_add / torch.sqrt(torch.sum(ec_add ** 2, dim=2))[:, :, None]

        ec = x[:, None, :] + ec_add 

        dx_new = ec - full_n_list
        dx_new_lengths = torch.sqrt(torch.sum(dx_new ** 2, dim=2))
        dx_new = dx_new / dx_new_lengths[:, :, None]
        
        two = torch.cat([dx_normal_lengths[None,:,:], dx_new_lengths[None, :, :]], 0)
        _, indices = torch.min(two, dim=0)

        dx = torch.where(indices[:, :, None] == 0, dx_normal, dx_new)

        d = torch.where(indices == 0, dx_normal_lengths, dx_new_lengths)
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
        # d = torch.sqrt(torch.sum(dx**2, dim=2))
        # dx = dx / d[:, :, None]

        # Calculate potential
        S, parallels, perpendicularities = self.calculate_interaction(dx, p, q, p_mask, idx, d, x)

        # # bc_contrib = torch.zeros_like(S)
        # with torch.no_grad():
        #     aas = np.maximum((self.alpha_scale-0.1), 0.)
        #     anisotropy_add = torch.where(p_mask[:,None] == 2, 5.*torch.abs(perpendicularities)*aas, 0.)
        
        # anisotropy = torch.where(p_mask[:,None] == 1, -0.05*(torch.abs(perpendicularities)), anisotropy)
        anisotropy = 1.
        # anisotropy = anisotropy_add + 1.
        print(z_mask.shape)
        print(S.shape)
        print(d.shape)
        Vij = z_mask.float() * (torch.exp(-d/anisotropy) - S * torch.exp(-d/5))

        # stress_strain = torch.abs(Vij)*torch.abs(dx)

        # stress_strain_sum = torch.sum(stress_strain, dim=1)

        bc = self.egg_BC(x)
        
        Vij_sum = torch.sum(Vij) + bc 

        return Vij_sum, int(m) # TODO print m
        
    def init_simulation(self, x, p, q, p_mask):
        print(p.shape, x.shape)
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(p_mask) == len(x)

            
        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
        p_mask = torch.tensor(p_mask, dtype=torch.int, device=self.device)
        self.beta   = torch.zeros_like(p_mask)

        self.scaled_egg_shape = torch.tensor(self.scaled_egg_shape, dtype=self.dtype, device=self.device)

        
        self.lambdas = []

        for type in self.interaction_data:
            self.lambdas.append(torch.tensor(type[:4], device=self.device))

        self.alphas = [torch.tensor(type[4], device=self.device) for type in self.interaction_data]

        self.start_xs = x.clone().detach()
        
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

def genetic_step(sim_dict, genetics):
    continue_from = sim_dict.pop('continue')
    yield_steps, yield_every = sim_dict['yield_steps'], sim_dict['yield_every']

    name = sim_dict.pop('name')

    assert continue_from + ".hdf5" in os.listdir('runs'), 'continue_from must be a valid run in /runs/'
    
    with h5py.File("runs/"+continue_from+".hdf5", "r") as f:
        if len(f['x'][:].shape) > 2:
            x = f['x'][-1]
            p = f['p'][-1]
            q = f['q'][-1]
        else:
            x = f['x'][:]
            p = f['p'][:]
            q = f['q'][:]
            

    start_x = x.copy()

    p_mask = genetics

    sim = Simulation(sim_dict)
    runner = sim.simulation(x, p, q, p_mask)


    i = 0
    t1 = time()
    for xx, pp, qq, pp_mask in itertools.islice(runner, yield_steps):
        i += 1
        ss = custom_progress(i/yield_steps, "🌻 ", "🌰 ", "🌱 ")
        print(ss + f'  Running {i} of {yield_steps}   ({yield_every * i} of {yield_every * yield_steps})   ({len(xx)} cells) ', end = "\r")

    print(f'Simulation done')
    print('Took', time() - t1, 'seconds')

    return start_x, x

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
        if len(f['x'][:].shape) > 2:
            x = f['x'][-1]
            p = f['p'][-1]
            q = f['q'][-1]
        else:
            x = f['x'][:]
            p = f['p'][:]
            q = f['q'][:]
            
        if len(f['properties']) == 1:
            p_mask = np.array(f['properties'][0])
        else:
            p_mask = np.array(f['properties'])
        


    sim = Simulation(sim_dict)
    runner = sim.simulation(x, p, q, p_mask)

    x_lst = [x]
    p_lst = [p]
    q_lst = [q]
    p_mask_lst = [p_mask]

    print("running "+ name)
    with h5py.File("runs/"+name+".hdf5", "w") as f:
        if "x" in f:
            del f["x"]
        if "p" in f:
            del f["p"]
        if "q" in f:
            del f["q"]
        if "properties" in f:
            del f["properties"]
    
    save((p_mask_lst, x_lst, p_lst,  q_lst), name=name, sim_dict=sim_dict)


    notes = sim_dict['notes']
    print('Starting simulation with notes:')
    print(notes)

    i = 0
    t1 = time()
    
    for xx, pp, qq, pp_mask in itertools.islice(runner, yield_steps):
        i += 1
        ss = custom_progress(i/yield_steps, "✅ ", "🟩 ")
        print(ss + f'  Running {i} of {yield_steps}   ({yield_every * i} of {yield_every * yield_steps})   ({len(xx)} cells) - scale : {sim.alpha_scale:.3}         ', end = "\r")

        x_lst.append(xx)
        p_lst.append(pp)
        q_lst.append(qq)
        # p_mask_lst.append(pp_mask)
        

        if i % 100 == 0:
            print("Saved!")
            save((p_mask_lst, x_lst, p_lst,  q_lst), name=name, sim_dict=sim_dict)

    print(f'Simulation done, saved {yield_steps} datapoints')
    print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst), name=name, sim_dict=sim_dict)
