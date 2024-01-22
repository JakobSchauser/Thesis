import h5py
import numpy as np
import sys
import os 
from pprint import pprint

if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please provide a run name"
    assert os.path.isfile(f'runs/{sys.argv[1]}'+".hdf5"), "Please provide a valid run name"    
    
    with h5py.File(f'{"runs/"+sys.argv[1]}.hdf5', 'r') as f:
        print("\nDatasets:")
        pprint(list(f.keys()))

        if "properties" in f:
            print("\nNumber of different cell types:")
            pprint(len(np.unique(f['properties'][:])))
            types, counts = np.unique(f['properties'][:], return_counts=True)
            print("\nCell type counts:")
            pprint(dict(zip(types, counts)))
            
        print("\nSimulation attributes:")
        pprint(dict(f.attrs.items()))