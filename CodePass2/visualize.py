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

def interactive_animate(positions, ps, qs, loaded_properties = None, alpha=10, interval=1/30, speed = 1, tosave = None):
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
    view.add(scatter1)

    if tosave is None:
        # color so we can see the direction of the particles
        scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
        scatter2.set_data(xs[0] + ps[0]/5, edge_width=0, face_color=colors, size=size)
        view.add(scatter2)

        # color so we can see the direction of the particles
        scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
        scatter3.set_data(xs[0] + qs[0]/5, edge_width=0, face_color=(0.9,0.9,0), size=size)
        view.add(scatter3)


    daniel_stripe = (xs[0][:,0] > 35)*(xs[0][:,0] < 45)

    def update(ev):
        global iterator



        if tosave is not None:
            tosave_c = tosave[int(iterator%len(xs))]
            # # normalize colors
            min2, max2 = np.min(tosave_c[loaded_properties == 4]), np.max(tosave_c[loaded_properties == 4]) 

            # nonz = np.maximum((max2 - min2), 1e-6)
            tosave_c = (tosave_c - min2)/(max2 - min2)

            # tosave_c[loaded_properties != 4] = 1.
            tosave_c = [(c, c, c) for c in tosave_c]

            # tosave_c = [(0,0,0,1) if d else (1,1,1,1) for d in daniel_stripe] 
        else:
            tosave_c = (1, 1, 1, .5)

        scatter1.set_data(xs[int(iterator%len(xs))], edge_width=0, face_color=tosave_c, size=size)

        if tosave is None:
            scatter2.set_data(xs[int(iterator%len(xs))] + ps[int(iterator%len(xs))]/10, edge_width=0, face_color=colors, size=size)
            scatter3.set_data(xs[int(iterator%len(xs))] + qs[int(iterator%len(xs))]/9, edge_width=0, face_color=(0.9,0.9,0), size=size)

        iterator += speed
        if iterator >= len(xs):
            iterator = len(xs)-1

            
    timer = app.Timer(interval=interval)
    timer.connect(update)
    timer.start()


    @canvas.connect
    def on_key_press(event):
        global iterator
        if event.text == ' ':
            if timer.running:
                timer.stop()
            else:
                timer.start()

        if event.text == 'r':
            iterator = 0
            update(1)
        elif event.text == 'q':
            iterator -= 51
            update(1)
        elif event.text == 'e':
            iterator += 49
            update(1)
        elif event.text == ',':
            iterator -= 2
            update(1)
        elif event.text == '.':
            update(1)


    # We want to fly around
    view.camera = 'fly'

    km = view.camera._keymap 
    km.pop('E')
    km.pop('Q')
    view.camera._keymap = km

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

    plot_tosave = 0




    if len(sys.argv) > 2:
        speed = int(sys.argv[2])
    else:
        speed = 1
        # loaded_cells = np.load(sys.argv[1])
        # loaded_properties = None
    if len(sys.argv) > 1:
        path = "runs/"
        # path = "D:/"
        with h5py.File(path + sys.argv[1] + ".hdf5", 'r') as f:
            positions = f['x'][::1]
            loaded_properties = None
            if 'properties' in f:
                loaded_properties = f['properties'][:]

            if "tosave" in f:
                tosave = f['tosave'][:]
                print(tosave.shape)
                print(positions.shape)
            else:
                tosave = None

            ps = f['p'][::1]
            qs = f['q'][::1]
            
        print(loaded_properties)
        
        interactive_animate(positions, ps, qs, loaded_properties[0], speed = speed, tosave = tosave if plot_tosave else None)
    else:
        print('No file specified')