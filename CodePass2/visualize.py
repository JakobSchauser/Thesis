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
        iterator += 10
        if iterator >= len(xs):
            iterator = len(xs)-1
    timer = app.Timer(interval=interval)
    timer.connect(update)
    timer.start()

    # We want to fly around
    view.camera = 'fly'

    # move the camera a bit back
    #view.camera.center = (100,0,0)

    # rotate the camera
    #view.camera.azimuth = 90
    #view.camera.elevation = 0


    if sys.flags.interactive != 1:
        vispy.app.run()

    vispy.app.quit()



if __name__ == '__main__':
    # get command line argument

    if len(sys.argv) > 1:
        # loaded_cells = np.load(sys.argv[1])
        # loaded_properties = None

        with h5py.File("runs/" + sys.argv[1] + ".hdf5", 'r') as f:
            positions = f['x'][::1]
            loaded_properties = None
            if 'properties' in f:
                loaded_properties = f['properties'][:]

            ps = f['p'][::1]
            qs = f['q'][::1]
            
        print(loaded_properties)
        
        interactive_animate(positions, ps, qs, loaded_properties[0])
    else:
        print('No file specified')