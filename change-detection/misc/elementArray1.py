from psychopy import visual, event
import numpy as np
import math

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

# Parameters
loop_radius = 10
n_circ = 30
interval_dur = 1                    # duration (in sec) of target time interval
angle_ratio = 360/float(interval_dur)
tolerances = 0.125                   # allowed error in sec on EACH side

# Stimuli
win = visual.Window(size=(1920, 1200), fullscr=True, color=(0,0,0),
                    monitor='testMonitor',
                    allowGUI=False, units='cm')
target_zone = visual.RadialStim(win, tex='sqrXsqr', color='green', size=(loop_radius*2) + 1.5,  # size here = diameter
    visibleWedge=[0, angle_ratio * (tolerances*2)], radialCycles=1, angularCycles=0, interpolate=False,
    autoLog=False, units='cm')
target_zone.ori = 180 - (tolerances * angle_ratio)   # zero starts at 12 oclock for radial stim.
target_zone_cover = visual.Circle(win, radius = loop_radius - 1.5/2, edges=100,
    lineColor=None, fillColor=[0, 0, 0]) # Covers center of target zone wedge

circ_angles = np.linspace(-90,270,n_circ)
circ_X, circ_Y = pol2cart(circ_angles,[loop_radius] * n_circ)
circles = visual.ElementArrayStim(win, nElements=n_circ,sizes=.3,xys = zip(circ_X, circ_Y),
                           elementTex = None, elementMask = "circle",
                           colors=[(-1,-1,-1)] * n_circ)


# Plot first time
target_zone.draw()
target_zone_cover.draw()
#circles.draw()
win.flip()

# Assume incorrect response, update target zone
tolerances+= 0.012    # add 0.012 sec to tolerance
target_zone.visibleWedge = [0,  2*tolerances*angle_ratio]
target_zone.ori = 180 - (tolerances * angle_ratio)
