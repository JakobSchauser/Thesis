import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt



def get_rosettes(positions, properties, types = [-1]):

    def get_nbs(xs) -> set:
        tri = Delaunay(xs)
        indsx, all_neighbors = tri.vertex_neighbor_vertices

        neighbors = []
        for i in range(len(indsx)-1):
            neighbors.append(set(all_neighbors[indsx[i]:indsx[i+1]]))

        return neighbors

    
    N_time_steps = int(len(positions)//scl)
        
    counts = np.zeros((N_time_steps, positions.shape[1]))


    scl = 100
    for iii in range(N_time_steps):

        # vecs = []
        xx0 = positions[iii*scl]
        xx1 = positions[iii*scl+scl]


        # xx1 = xx1[xx0[:,2] < 0]
        # xx0 = xx0[xx0[:,2] < 0]

        nbs_0 : set = get_nbs(xx0)
        nbs_1 : set = get_nbs(xx1)

        pairs = []

        for i in range(len(nbs_0)):
            if properties[i] not in types:
                continue

            if xx0[i,2] > 0:
                continue
            
            shouldbreak = False
            new_nbs = nbs_1[i] - nbs_0[i]
            for new_nb in new_nbs:
                if set([i, new_nb]) in pairs:
                    continue
                pairs.append(set([i, new_nb]))


                dist = np.array([np.linalg.norm(positions[iii*scl+ scl,i] - positions[iii*scl+scl,new_nb])])

                if dist > 6:
                    continue

                nb_nbs = nbs_1[new_nb]
                overlap = nb_nbs & nbs_1[i]
                
                for common_nb in overlap:
                    common_ns_prev_nbs = nbs_0[common_nb]
                    common_nb_new_nbs = nbs_1[common_nb]

                    overlapx2 = (common_ns_prev_nbs - common_nb_new_nbs) & overlap 

                    if len(overlapx2) > 0:
                        counts[iii,i] += 1
                        vec_between = positions[iii*scl+scl,common_nb] - positions[iii*scl+scl,list(overlapx2)[0]]
                        # vecs.append(vec_between)

                        shouldbreak = True
                        break

                if shouldbreak:
                    break
    return counts


def make_rosette_images(positions, properties, types = [-1]):
    rosettes = get_rosettes(positions, properties, types)
    
    counts = rosettes.sum(axis=0)
    xx, yy, zz = positions[1,:,0], positions[1,:,1], positions[1,:,2]

    fig = plt.figure(figsize=(13,3))
    plt.scatter(xx[yy<0], zz[yy<0], c = counts[yy<0], s=4)
    plt.scatter(xx[yy>0]+ 1.1*(xx.max() - xx.min()), zz[yy>0], c = counts[yy>0], s=4)
    plt.colorbar()
    # remove the yticks
    plt.yticks([])
    # remove the xticks
    plt.xticks([])

    fig.tight_layout()
    plt.show()

    sumcount = np.sum(counts, axis=1)
    plt.xlabel("Time")
    plt.ylabel("Sum of Rosettes")
    plt.plot(sumcount)




def germ_band_length(position, properties):
    finalgb = position[-1][properties[0] == 1]

    finalgb = (finalgb - np.min(finalgb, axis=0)) / (np.max(finalgb, axis=0) - np.min(finalgb, axis=0))*100


    xx, yy, zz = finalgb[:,0], finalgb[:,1], finalgb[:,2]

    dst = 2
    final_furrow_mask = (yy > 50-dst) * (yy < 50+dst)
    final_furrow = finalgb[final_furrow_mask]

    print(final_furrow.shape[0],  "found")

    dists_from_cephallic = np.linalg.norm(final_furrow - [0, 50, 0], axis=1)

    # np.sort(dists_from_cephallic)
    return (final_furrow.shape[0], dists_from_cephallic.max() - dists_from_cephallic.min())