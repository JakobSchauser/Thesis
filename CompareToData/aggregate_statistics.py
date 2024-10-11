import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import kdtree
from tqdm import tqdm

seethru = 0

# def find_neighbors_3d_with_threshold(coords, max_distance=15, seethru=1.0):
#     # Step 1: Compute pairwise distances, but only for points within the max distance
#     diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # Broadcasting for pairwise differences
#     distances = np.linalg.norm(diff, axis=2)  # Pairwise distance matrix
    
#     # Step 2: Apply the max distance threshold (ignores points beyond 15 units)
#     distances[distances > max_distance] = np.inf  # Mark distances greater than the threshold as inf
#     np.fill_diagonal(distances, np.inf)  # Set diagonal to inf to avoid self-neighbors

#     # Step 3: Define the threshold distance as half of the valid distances
#     thresholds = distances / 2

#     # Step 4: Find neighbors using the threshold, skipping distant points
#     neighbors = []
#     for i in range(coords.shape[0]):
#         print("looking @", i)
#         # Apply the threshold and get neighbors within 15 units
#         z_mask = find_true_neighbours_np_with_threshold(thresholds[i], distances, seethru)
        
#         # Exclude self and get valid neighbor indices
#         neighbor_indices = np.where(z_mask)[0]
#         neighbors.append(neighbor_indices[neighbor_indices != i])
    
#     return neighbors

# def find_true_neighbours_np_with_threshold(threshold, distances, seethru):
#     # Directly check for neighbors using the threshold
#     z_mask = (distances < threshold) & (distances < 15)  # Boolean mask for neighbors below threshold and max distance
#     return z_mask

def get_rosettes(positions, properties, types = [-1], scale = 100, max_dist = 5):

    def get_nbs(xs) -> set:
        tri = Delaunay(xs, furthest_site = 1)
        indsx, all_neighbors = tri.vertex_neighbor_vertices

        neighbors = []
        for i in range(len(indsx)-1):
            neighbors.append(set(all_neighbors[indsx[i]:indsx[i+1]]))


        # check the distances between the neighbors
        actual_neighbors = []
        for i in range(len(xs)):
            toappend = []
            for j in neighbors[i]:
                if np.linalg.norm(xs[i] - xs[j]) < max_dist:
                    toappend.append(j)
            actual_neighbors.append(set(toappend))

        return actual_neighbors

    # def get_nbs(xs):
    #     return find_neighbors_3d_with_threshold(xs)

    if types == [-1]:
        types = np.unique(properties)
    
    scl = scale
    N_time_steps = int(len(positions)//scl)
        
    counts = np.zeros((N_time_steps, positions.shape[1]))
    vecs_between = []


    potential_rosettes = []

    for iii in tqdm(range(N_time_steps - 1)):
        vecs_between_this_time_step = []

        # vecs = []
        xx0 = positions[iii*scl]
        xx1 = positions[iii*scl+scl]


        # xx1 = xx1[xx0[:,2] < 0]
        # xx0 = xx0[xx0[:,2] < 0]

        if iii == 0:
            nbs_0 : set = get_nbs(xx0)
            nbs_1 : set = get_nbs(xx1)
        else:
            nbs_0 = nbs_1
            nbs_1 = get_nbs(xx1)
        pairs = []

        potenial_to_look_at = [p for p in potential_rosettes if p[0] == iii-2]

        for p in potenial_to_look_at:
            p_iii = p[0]
            p_i = p[1]
            p_new_nb = p[2]
            p_common_nb = p[3]
            p_lost_nb = p[4]

            # has_been_seen = False
            # for pair in pairs:
            #     if p_i in pair and p_new_nb in pair:
            #         has_been_seen = True
            #         break
            
            # if has_been_seen and iii > 20:
            #     continue

            # check if the cells are still neighbors
            if p_i not in nbs_0[p_new_nb] or p_new_nb not in nbs_0[p_i]:
                continue
            # check if the common neighbor is still a neighbor of both
            if p_common_nb not in nbs_0[p_i] or p_common_nb not in nbs_0[p_new_nb]:
                continue
            # check if the common neighbor has lost the connection to the new neighbor
            if p_lost_nb in nbs_1[p_common_nb]:
                continue

            counts[p_iii, p_i] += 1
            vec_between = xx0[p_new_nb] - xx0[p_i]
            vecs_between_this_time_step.append(vec_between)
            pairs.append(set([new_nb, common_nb, p_lost_nb]))
            # print(p)
            # print(iii*scl)

        vecs_between.append(vecs_between_this_time_step)



        # for each cell
        for i in range(len(nbs_0)):
            if properties[i] not in types:
                continue

            # if xx0[i,2] > 0:   wtf dude?
            #     continue
            
            shouldbreak = False
            new_nbs = nbs_1[i] - nbs_0[i]

            # for each newly aquired neighbor
            for new_nb in new_nbs:
                if set([i, new_nb]) in pairs:     # I think removing this will allow for N>2 rosettes to be counted but will also double count t1-transitions
                    continue
                if properties[new_nb] not in types:
                    continue



                dist = np.array([np.linalg.norm(positions[iii*scl + scl,i] - positions[iii*scl+scl,new_nb])])

                
                overlap = nbs_1[new_nb] & nbs_1[i]
                overlap_old = nbs_0[new_nb] & nbs_0[i]

                # for each cell that is a neighbor of both of the original cell and the new neighbor 
                for common_nb in overlap:
                    common_ns_prev_nbs = nbs_0[common_nb]
                    common_nb_new_nbs = nbs_1[common_nb]

                    old_nb_nbs = (common_nb_new_nbs & (overlap | overlap_old) )
                    new_nb_nbs =  (common_ns_prev_nbs & (overlap | overlap_old))
                    # if the common neighbor has lost a neighbor that they both are connected to
                    overlapx2 = old_nb_nbs - new_nb_nbs

                    if len(overlapx2) == 1:
                        # print(len(overlapx2))
                        # print(old_nb_nbs)
                        # print(new_nb_nbs)
                        potential_rosettes.append([iii, i, new_nb, common_nb, list(overlapx2)[0]])

                        shouldbreak = True
                        break

                if shouldbreak:
                    break
    return counts, vecs_between


def make_rosette_images(positions, properties, types = [-1], scale = 100):
    rosettes, vecs_between = get_rosettes(positions, properties, types, scale)
    
    # counts = rosettes.sum(axis=0)
    counts = rosettes
    per_cell_counts = rosettes.sum(axis=0) 
    per_time_counts = rosettes.sum(axis=1)
    print("counts")
    print(counts.shape)
    print("per cell")
    print(per_cell_counts.shape)
    print("per time step")
    print(per_time_counts.shape)

    # plt.hist(counts, bins=20)
    # plt.show()
    fig = plt.figure(figsize=(13,3))
    xx, yy, zz = positions[1,:,0], positions[1,:,1], positions[1,:,2]
    # xx, yy, zz = positions[1,:,0], positions[1,:,1], positions[1,:,2]
    plt.scatter(xx[yy<0], zz[yy<0], c = per_cell_counts[yy<0], s=4)
    plt.scatter(xx[yy>0]+ 1.1*(xx.max() - xx.min()), zz[yy>0], c = per_cell_counts[yy>0], s=4)
    plt.colorbar()
    # remove the yticks
    plt.yticks([])
    # remove the xticks
    plt.xticks([])

    fig.tight_layout()
    plt.show()

    # for kkk in range(5):
    #     fig = plt.figure(figsize=(13,3))
    #     xx, yy, zz = positions[kkk*scale,:,0], positions[kkk*scale,:,1], positions[kkk*scale,:,2]
    #     # xx, yy, zz = positions[1,:,0], positions[1,:,1], positions[1,:,2]
    #     print(xx.shape, counts.shape, counts[kkk].shape)
    #     plt.scatter(xx[yy<0], zz[yy<0], c = counts[kkk][yy<0], s=4)
    #     plt.scatter(xx[yy>0]+ 1.1*(xx.max() - xx.min()), zz[yy>0], c = counts[kkk][yy>0], s=4)
    #     plt.colorbar()
    #     # remove the yticks
    #     plt.yticks([])
    #     # remove the xticks
    #     plt.xticks([])

    #     fig.tight_layout()
    #     plt.title("Sum of events this time step: "+ str(sum(counts[kkk])))

    #     plt.show()


    plt.xlabel("Time")
    plt.ylabel("Sum of Rosettes")
    plt.plot(per_time_counts)

    return rosettes, per_time_counts, vecs_between




def germ_band_length(position, properties, sensitivity = 5, n_timesteps = 5, scaled = False):
    gb_types = np.logical_or(properties == 1, properties == 2)
    non_gb_types = np.logical_not(gb_types)

    intersections = np.zeros((n_timesteps, 360//2))



    for kkk in range(n_timesteps):
        timeindex = int(len(position)//n_timesteps*kkk)
        finalgb = position[timeindex]

        max, min = np.max(finalgb, axis=0), np.min(finalgb, axis=0)



        if not scaled:
            # using max and min transform center into center of shape
            center = (max + min)/2.
            print(center)
        else:
            center = np.array([50,50,50])
            finalgb = (finalgb-min)/(max - min)*100

        print(finalgb.shape[0], "points in germ band")
        print(np.max(finalgb, axis=0), "max")
        print(np.min(finalgb, axis=0), "min")

        for iii, degangle in enumerate(range(0, 360, 2)):
            angle = np.deg2rad(degangle)
            # raycast from the center
            ray = np.array([-np.cos(angle), 0, np.sin(angle)])
            
            shouldbe = False
            # find the intersection with the germ band
            nsteps = 100
            for step in range(nsteps):
                point = center - step * ray / nsteps * 50. + ray*100
                if np.min(np.linalg.norm(finalgb[non_gb_types] - point, axis=1)) < sensitivity:
                    shouldbe = False
                    break
                if np.min(np.linalg.norm(finalgb[gb_types] - point, axis=1)) < sensitivity:
                    shouldbe = True
                    break
                
            intersections[kkk, iii] = shouldbe

    # xx, yy, zz = finalgb[:,0], finalgb[:,1], finalgb[:,2]

    # dst = 2
    # final_furrow_mask = (yy > 50-dst) * (yy < 50+dst)
    # final_furrow = finalgb[final_furrow_mask]

    # print(final_furrow.shape[0],  "found")

    # dists_from_cephallic = np.linalg.norm(final_furrow - [0, 50, 0], axis=1)

    # # np.sort(dists_from_cephallic)
    # return (final_furrow.shape[0], dists_from_cephallic.max() - dists_from_cephallic.min())
    return intersections


