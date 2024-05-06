from reworked import run_simulation
from torch import float as torch_float

l0 = 0.
l1, l2, l3 = 0.55, 0.1, 0.095
alpha = 0.2



                            
interaction_data = [[l0, l1*1., l2, 0., 0.],        # green
                    [l0, l1*0.55, l2, l3, 0.],        # red
                    [0.2, l1, l2, 0.3*l3, alpha],     # blue  
                    [l0, l1*1.7, l2, 0., alpha*0.0],        # yellow 
                    [0., l1*1.2, l2, 0., alpha*0.8],     # baby blue from 0.8
                    [0., l1*1.5, l2, 0., alpha*0.8], # pink
                    # [l0, l1*1., l2, 0., -alpha*0.9] # black
                    ]


# l1, l2, l3 = 0.7, 0.1, 0.2

# interaction_data = [[l0, l1, l2, 0., 0.],        # green
#                     [l0, l1, l2, l3, 0.],
#                     [l0, l1, l2, 0., 0.],
#                     [l0, l1, l2, 0., 0.],
#                     [l0, l1, l2, 0., 0.],
#                     [l0, l1, l2, 0., 0.]]


# interaction_data = [[0., 0.7, 0., 0., 0.],        # green
#                     [l0, l1, l2, l3, 0.],        # red
#                     [0.05, l1, l2, 0., alpha],     # blue
#                     [0., 0., 0., 0., 0.],        # yellow
#                     [0.05, 2*l1, l2, 0., 0.5*alpha],     # baby blue  
#                     ]



if __name__ == '__main__':
    sim_dict = {
        # 'data'              : (make_random_sphere, [size, conc, 35]), #if tuple 2 long then data generation, if 4 long then data
        # 'continue'          : "random_baby_2",
        'continue'          : "stas_one_stripe",
        "name"              : "inverse_anisotropy",
        'dt'                : 0.1,
        'eta'               : 0.002,
        'init_k'            : 100,
        'alpha'             : 0,
        'yield_steps'       : 2000,
        'yield_every'       : 50,
        'interaction_data'  : interaction_data,
        'gamma'             : 5.0,
        'seethru'           : 1,
        'device'            : 'cuda',
        'dtype'             : torch_float,
        'notes'             : f"trying stripes",
    }

    run_simulation(sim_dict)