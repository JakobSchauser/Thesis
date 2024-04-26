import numpy as np
import vispy.scene
from vispy.scene import visuals
import scipy
import sys
from vispy import app, scene
from vispy.visuals.transforms import STTransform
import os
import h5py


iterator = 0

cuts =   [[[63, 60],
  [59,  0],
  [18 , 9],],

 [[43, 52],
  [71, 47],
  [61, 40],],]
cuts = np.array(cuts)

def from_pos_to_perc(poss):
    xx = poss[-1][:,0]
    yy = poss[-1][:,1]
    zz = poss[-1][:,2]

    xx = (xx - xx.min())/(xx.max() - xx.min())*100
    yy = (yy - yy.min())/(yy.max() - yy.min())*100
    zz = (zz - zz.min())/(zz.max() - zz.min())*100

    return np.array([xx, yy, zz]).T


def get_genetics(cuts, percs):
    genetics = np.zeros(5000)
    xx,yy,zz = percs.T

    types = [1,2,4]

    for i in range(len(cuts)):
        xlow, xhigh =  cuts[i,0,0],  cuts[i,0,0] + cuts[i,0,1]
        ylow, yhigh =  cuts[i,1,0],  cuts[i,1,0] + cuts[i,1,1]
        zlow, zhigh =  cuts[i,2,0],  cuts[i,2,0] + cuts[i,2,1]
        
        cut = (xx > xlow) * (xx < xhigh) * (yy > ylow) * (yy < yhigh) * (zz >zlow) * (zz < zhigh)
        genetics[cut] = types[i]
    
    return genetics



def interactive_animate(positions, ps, qs, loaded_properties = None, alpha=10,
                     frame_every = 1, interval=1/30, save=True ):
    print('Animating')

    # Load the data
    xs = positions

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()                               

    size = 2.0

    # Create scatter object and fill in the data
    colors = [(1,0,0) for i in range(len(xs[0]))]

    different_colors = [(0,1,0), (1,0,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (0,0,0)]

    loaded_properties = get_genetics(cuts, from_pos_to_perc(xs))

    if loaded_properties is not None:
        colors = [different_colors[int(l)] for l in loaded_properties]

    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter1.set_data(xs[0], edge_width=0, face_color=(1, 1, 1, .5), size=size)

    # color so we can see the direction of the particles
    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter2.set_data(xs[0] + ps[0]/5, edge_width=0, face_color=colors, size=size)

        # color so we can see the direction of the particles
    scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter3.set_data(xs[0] + qs[0]/5, edge_width=0, face_color=(0.9,0.9,0), size=size)


    view.add(scatter1)
    view.add(scatter2)
    view.add(scatter3)


    def update(ev):
        global iterator
        scatter1.set_data(xs[int(iterator%len(xs))], edge_width=0, face_color=(1, 1, 1, .5), size=size)
        scatter2.set_data(xs[int(iterator%len(xs))] + ps[int(iterator%len(xs))]/10, edge_width=0, face_color=colors, size=size)
        scatter3.set_data(xs[int(iterator%len(xs))] + qs[int(iterator%len(xs))]/9, edge_width=0, face_color=(0.9,0.9,0), size=size)
        iterator += 1
        if iterator >= len(xs):
            iterator = len(xs)-1
    timer = app.Timer(interval=interval)
    timer.connect(update)
    timer.start()

    # We want to fly around
    view.camera = 'fly'

    if sys.flags.interactive != 1:
        vispy.app.run()

    vispy.app.quit()



if __name__ == '__main__':
    # get command line argument

    with h5py.File("stas_one_stripe.hdf5", 'r') as f:
        positions = f['x'][::1]
        loaded_properties = None
        if 'properties' in f:
            loaded_properties = f['properties'][:]

        ps = f['p'][::1]
        qs = f['q'][::1]
        
    print(loaded_properties)
    
    interactive_animate(positions, ps, qs, loaded_properties[0])