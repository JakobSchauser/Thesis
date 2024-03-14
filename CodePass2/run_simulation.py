from reworked import run_simulation
from torch import float as torch_float

if __name__ == '__main__':


    sim_dict = {
        'output_folder'     : f"size{size}_conc{conc}_{i}",
        'data'              : (make_random_sphere, [size, conc, 35]), #if tuple 2 long then data generation, if 4 long then data
        'dt'                : 0.2,
        'eta'               : 0.2,
        'yield_steps'       : 2,
        'yield_every'       : 10_000,
        'lambdas'           : [0.3 ,0.3, 0.45, 0.43, 0.12],
        'gamma'             : 5.0,
        'seethru'           : 1,
        'alpha'             : 0,
        'device'            : 'cuda',
        'dtype'             : torch_float,
        'notes'             : f"size{size}_conc{conc}_{i}",
    }

    run_simulation()