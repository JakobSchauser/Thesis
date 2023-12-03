import vispy
import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys

def interactive_plot(cells, alpha=10):

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    
    # Load the data
    loaded_cells = np.load(cells)
    x = loaded_cells[:,0,:]
    ps = loaded_cells[:,1,:]
    print(ps)
    # Create scatter object and fill in the data
    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter1.set_data(x, edge_width=0, face_color=(1, 1, 1, .5), size=2.5)

    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter2.set_data(x + ps*2.5, edge_width=0, face_color=(1, 0, 0, .5), size=0.5)
    # Add the scatter object to the view
    view.add(scatter1)
    view.add(scatter2)

    # We want to fly around
    view.camera = 'fly'

    print(sys.flags.interactive )
    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()


if __name__ == '__main__':
    # get command line argument
    if len(sys.argv) > 1:
        interactive_plot(sys.argv[1])
    else:
        print('No file specified')