from psychopy import visual, event, misc
import numpy as np
import math


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

win = visual.Window(units='pix')

num_check = 360
check_size = [10, 10]

location = [0, 0]
# generate loc array
loc = np.array(location) + np.array(check_size) // 2

# array of rgbs for each element (2D)

colors = np.random.random((num_check ** 2, 3))

# array of coordinates for each element
xys = []
# populate xys
low, high = num_check // -2, num_check // 2

for y in range(low, high):
    for x in range(low, high):
        xys.append((check_size[0] * math.cos(x),
                    check_size[1] * math.sin(y)))

loop_radius = 10
n_circ = 300
interval_dur = 1                    # duration (in sec) of target time interval
angle_ratio = 360/float(interval_dur)
tolerances = 0.125                   # allowed error in sec on EACH side

xys = []
loop_radius = 8
n_circ =360

hsv = np.ones([n_circ,1,3], dtype=float)
hsv[:,:,0] = np.linspace(0,360,n_circ, endpoint=False)[:,np.newaxis]
rgb = misc.hsv2rgb(hsv)
for theta in np.linspace(0,360,n_circ, endpoint = False):
    xys.append((loop_radius * np.cos(theta*np.pi/180),
                loop_radius * np.sin(theta*np.pi/180)))

stim = visual.ElementArrayStim(win, nElements=n_circ,sizes=0.9,xys = xys,
                           elementTex = None, elementMask = "circle",
                           colors=rgb.reshape(n_circ,3),colorSpace='rgb', interpolate = True)


stim.draw()
win.flip()
keys = event.waitKeys(keyList = ['space', 'escape'])
