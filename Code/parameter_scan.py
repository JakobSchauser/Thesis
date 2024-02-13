import sys
import importlib
import threading

# import itertools
from itertools import product

if __name__ == "__main__":

    arguments = sys.argv
    if len(arguments) < 4:
        print("Usage: python parameter_scan.py <name> <parameter> <steps> [<parameter> <steps>]")
        sys.exit(1)
    
    if len(arguments) >= 4:
        assert len(arguments) in [4,6,8]
    # assert arguments[2] in G, f"Parameter '{arguments[2]}' not found"

    N_params = (len(arguments)-2)//2


    for t in threads:
        t.join()

    print("Done with all steps")


