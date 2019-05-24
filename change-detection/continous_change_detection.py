# -*- coding: utf-8 -*-

# change_detection.py
# Visual Cognitive Neuroscience Lab
# Brock University
#
# Created by Thomas Nelson <tn90ca@gmail.com>
#
# Modified by Aditya Upadhyayula <supadhy6@jhu.edu>
#
# This script was developed for use by the Visual Cognitive Neuroscience Lab
# at Brock University.


"""This script runs a standard color change detection task (modeled after Luck & Vogel, 1997, Nature). The script
was written by Thomas Nelson for the Visual Cognitive Neuroscience Lab at Brock University. Feel free to use this
as a starting point for creating your own experiments.

"""


# Import required third party modules
import os
import csv
import math
import random
import numpy as np
from psychopy import visual, monitors, core, event, os, data, gui, misc, logging
from psychopy.tools.monitorunittools import (cm2pix, deg2pix, pix2cm,
                                             pix2deg, convertToPix)
from sklearn.metrics.pairwise import euclidean_distances

try:
	import matplotlib
	if matplotlib.__version__ > '1.2':
		from matplotlib.path import Path as mplPath
	else:
		from matplotlib import nxutils
	haveMatplotlib = True
except Exception:
	haveMatplotlib = False


########################################################################################################################
#                                                  Program Constants                                                   #
########################################################################################################################

# I have these settings configured for my macbook you will need to change them to suit your monitor
MONITOR_WIDTH = 30.48  # The width of the display on your monitor
MONITOR_DISTANCE = 65  # The veiwing distance from the user to the monitor
MONITOR_RESOLUTION = [1440, 900]  # The resolution of your monitor display

EXP_NAME = "continuous_change_detection"  # The name of the experiment for save files
VERBOSE  = False  # Set to True to print out a trial timing log for testing

# Set file paths for required directories
EXP_PATH   = os.path.dirname(os.path.realpath(__file__))  # The path to this script
HOME_PATH  = os.path.realpath(os.path.expanduser("~"))  # The path to the home folder
SAVE_PATH  = os.path.join(HOME_PATH, 'Desktop', 'experiment_data', EXP_NAME)  # Path to save experiment results
IMAGE_PATH = os.path.join(EXP_PATH, 'images')  # Path to store any required experiment images

# Note that these RGB values are converted from (0 and 255) to (-1 and 1)
BG_color     = [0, 0, 0]  # Set a background color, currently grey
FIX_color    = [-1, -1, -1]  # Set the fixation color, currently black
TEXT_color   = [-1, -1, -1]  # The text color, currently white

textureRes = 64
n_circ = 360

hsv = np.ones([n_circ,1,3], dtype=float)
hsv[:,:,0] = np.linspace(0,360,n_circ, endpoint=False)[:,np.newaxis]
rgb = misc.hsv2rgb(hsv)
TRIAL_colorS = [[-1, -1, -1],  # Black
                 [-1, -1, 1],   # Blue
                 [-1, 1, -1],   # Green
                 [-1, 1, 1],    # Cyan
                 [1, -1, -1],   # Red
                 [1, -1, 1],    # Purple
                 [1, 1, -1],    # Yellow
                 [1, 0.296, -1]]     # Orange


NUM_TYPE = 3  # Number of different trial types
NUM_REPS = 50  # Number of repetitions for each different trial type

# Note that all sizing is in visual degrees
FIXATION_SIZE   = 0.2  # Size of the fixation at the center of the screen in visual degree
STIM_POS_RADIUS = 4  # Number of visual degrees between the center and stimuli
STIM_SIZE       = 1  # Size of the stimuli in visual degrees, length and width
STIM_THICKNESS  = 1  # The thickness of the outline of the stimuli

TEXT_HEIGHT = 0.75   # The height in visual degrees of instruction text
TEXT_WRAP   = 40  # The character limit of each line of text before word wrap

# Note that all timing is in seconds
ITI_TIME   = 0.5  # The time in seconds between trials
STIM_TIME  = 0.5  # The time in seconds to display the stimuli
DELAY_TIME = 1.0  # The time in seconds between stimuli and probe
BREAK_TIME = 0.75  # The time in seconds between break end and trial start

INS_MSG   = "You will be presented with colored squares, try to remember their colors.\n\n"
INS_MSG  += "For each, trial there will follow a second set of squares in the same locations.\n\n"
INS_MSG  += "If there was any change in color to the second set of squares from the first, press "
INS_MSG  += "the z key,\n\nIf they have not changed color, press the m key\n\nPress any key when "
INS_MSG  += "you are ready to begin."
BREAK_MSG = "Take a quick break. When you are ready to continue, press any key."
THANK_MSG = "Thank you for your participation. Please go find the experimenter."

# This is a list of the column headers for the output file
HEADER_LIST = ['Subject_Number', 'Trial_Number', 'Number_of_Stim', 'Stim_color_1', 'Stim_color_2', 'Stim_color_3',
               'Stim_color_4', 'Stim_color_5', 'Stim_color_6', 'Probe_color', 'Subject_Response_color', 'Error (l2 norm)',
               'Response_Time']


########################################################################################################################
#                                                  Class Declaration                                                   #
########################################################################################################################

class Trial(object):
    """The standard trial class represents a single trial in a standard change detection task. This class is used to
    set up and run a trial.

    """

    def __init__(self, trial_num, rep_num):
        """Class constructor function initializes which trial format to follow from parameter input, also calls the
        functions to set the memory trial olour and location.

        trial_num: Integer
            The trial number used to determine the trial format.
        rep_num: Integer
            The rep number of this trial, used to determine color change.

        """

        self.rep_num     = rep_num
        self.trial_num   = trial_num
        self.num_stimuli = 0

        self.stim_positions = []

        self.change        = False
        self.stim_colors  = []
        self.probe_colors = []

        # Determine the load number for this trial based on the trial type number
        if trial_num == 0:
            self.num_stimuli = 2
        elif trial_num == 1:
            self.num_stimuli = 4
        elif trial_num == 2:
            self.num_stimuli = 6
    # end def __init__

    def set_positions(self):
        """This function will determine the location (left or right) for the memory sample and distraction sample. Uses
        even and odd numbers to ensure an even distribution of left and right positioning. Also generates the
        coordinates for the gui and results print out.

        """

        # Generate a list of 12 positions around the center
        for pos in range(12):
            angle = math.radians(360 / 12 * pos)
            self.stim_positions.append([math.cos(angle)*STIM_POS_RADIUS, math.sin(angle)*STIM_POS_RADIUS])
        random.shuffle(self.stim_positions)  # Shuffle the list of positions

        # Cut the position list to the correct number of stimuli required
        self.stim_positions = self.stim_positions[:self.num_stimuli]
    # end def set_positions

    def set_colors(self):
        """This function is used to randomly generate memory stimuli colors and memory probe colors based on a color
        match or not.

        """

        # Shuffle the list of available colors
        random.shuffle(TRIAL_colorS)

        # Create the lists of stimuli and probe colors
        for color in TRIAL_colorS:
            self.stim_colors.append(color)
            self.probe_colors.append(color)

        # If a change is present replace on of the colors in the probe list with a new color
        if (self.rep_num % 2) == 0:
            self.change = True
            rand_1 = random.randint(0, self.num_stimuli-1)
            rand_2 = random.randint(self.num_stimuli, 7)
            self.probe_colors[rand_1] = self.probe_colors[rand_2]

        # Cut the color lists to the correct number of stimuli required
        self.stim_colors  = self.stim_colors[:self.num_stimuli]
        self.probe_colors = self.probe_colors[:self.num_stimuli]
    # end def set_colors

# end class Trial


########################################################################################################################
#                                                 Function Declaration                                                 #
########################################################################################################################

def setup_subject():
    """The purpose of this function is to present a dialog box to the experimenter so they can assign a subject number
    to each subject. This number will be used to create an output file and then used a random seed.

    """

    global NUM_REPS

    num_error = gui.Dlg(title="Error!")
    num_error.addText("This is not a valid subject!")

    subj_error = gui.Dlg(title="Error!")
    subj_error.addText("This subject number has already been used!")

    while True:
        subj_info = {'Subject Number': ''}
        subj_dlg  = gui.DlgFromDict(dictionary=subj_info, title=EXP_NAME)

        # If user hits cancel then safely close program
        if not subj_dlg.OK:
            core.quit()

        if subj_info['Subject Number'].isdigit():
            file_name = subj_info['Subject Number'] + '.csv'
            file_path = os.path.normpath(os.path.join(SAVE_PATH, file_name))

            # If we are using the test subject number make the experiment shorter
            if int(subj_info['Subject Number']) == 999:
                NUM_REPS = 2
                break

            if not os.path.isfile(file_path):
                break
            else:
                subj_error.show()
        else:
            num_error.show()

    return subj_info['Subject Number'], file_path
# end def setup_subject


def set_psychopy():
    """

    """

    # Build the monitor with correct sizing for psychopy to calculate visual degrees
    mon = monitors.Monitor('testMonitor')
    mon.setDistance(MONITOR_DISTANCE)  # Measure first to ensure this is correct
    mon.setWidth(MONITOR_WIDTH)  # Measure first to ensure this is correct
    mon.setSizePix(MONITOR_RESOLUTION)

    # Build the window for psychopy to run the experiment in
    win = visual.Window(fullscr=True, screen=0, allowGUI=False, allowStencil=False, monitor=mon, color=BG_color,
                        colorSpace='rgb', units='deg')

    # Set up an event clock for timing in trials
    event_clock = core.Clock()

    # Set up an event catcher to collect keyboard and mouse responses
    mouse    = event.Mouse(win=win)
    key_resp = event.BuilderKeyResponse()

    return win, mon, event_clock, key_resp, mouse
# End def set_psychopy


def set_trials():
    """

    """

    # Build all trials before we start experiment
    test_set = []

    for rep in range(NUM_REPS):
        for trial in range(NUM_TYPE):
            set_trial = Trial(trial, rep)  # Initialize the Trial
            set_trial.set_positions()  # Set the stimuli positions for the Trial
            set_trial.set_colors()  # Set the stimuli colors for the Trial
            test_set.append(set_trial)

    # Randomize our trial order
    random.shuffle(test_set)

    return test_set
# end def set_trials


def display_message(win, fix, txt, msg):
    """A function to display text to the experiment window.

    win: psychopy.visual.Window
        The window to write the message to.

    fix: psychopy.visual.Circle
        The fixation point to be removed from the screen.

    txt: psychopy.visual.TextStim
        The text object to present to the screen.

    msg: String
        The contents for the text object.

    """

    txt.setText(msg)
    fix.setAutoDraw(False)
    txt.setAutoDraw(True)
    win.flip()

    event.waitKeys()

    txt.setAutoDraw(False)
    fix.setAutoDraw(True)
    win.flip()
# end def display_message


def display_fixation(win, clk, fix, dur):
    """A function to display a fixation to the screen for a duration of time. This us to be used to either an
    ITI or ISI.

    win: psychopy.visual.Window
        The window to write the message to.

    fix: psychopy.visual.Circle
        The fixation point to be removed from the screen.

    dur: Float
        The duration of time to display the fixation on screen.

    """

    event_clock.reset()

    while True:
        if clk.getTime() >= dur:
            break
        fix.draw(True)
        win.flip()
        if clk.getTime() >= dur:
            break

    FT = clk.getTime()

    if VERBOSE:
        print("FIXATION SCREEN:", FT)
# end def display_fixation

## Porting psychopy's contains function to elementArrayStim
def contains(thisElementArrayStim, x, y=None, units=None):
	"""Returns True if a point x,y is inside the stimulus' border.

	Can accept variety of input options:
		+ two separate args, x and y
		+ one arg (list, tuple or array) containing two vals (x,y)
		+ an object with a getPos() method that returns x,y, such
			as a :class:`~psychopy.event.Mouse`.

	Returns `True` if the point is within the area defined either by its
	`border` attribute (if one defined), or its `vertices` attribute if
	there is no .border. This method handles
	complex shapes, including concavities and self-crossings.

	Note that, if your stimulus uses a mask (such as a Gaussian) then
	this is not accounted for by the `contains` method; the extent of the
	stimulus is determined purely by the size, position (pos), and
	orientation (ori) settings (and by the vertices for shape stimuli).

	See Coder demos: shapeContains.py
	"""
	# get the object in pixels
	if hasattr(x, 'border'):
		xy = x._borderPix  # access only once - this is a property
		units = 'pix'  # we can forget about the units
	elif hasattr(x, 'verticesPix'):
		# access only once - this is a property (slower to access)
		xy = x.verticesPix
		units = 'pix'  # we can forget about the units
	elif hasattr(x, 'getPos'):
		xy = x.getPos()
		units = x.units
	elif type(x) in [list, tuple, np.ndarray]:
		xy = np.array(x)
	else:
		xy = np.array((x, y))
	# try to work out what units x,y has
	if units is None:
		if hasattr(xy, 'units'):
			units = xy.units
		else:
			units = thisElementArrayStim.units
	if units != 'pix':
		xy = convertToPix(xy, pos=(0, 0), units=units, win=thisElementArrayStim.win)
	# ourself in pixels
	if hasattr(thisElementArrayStim, 'border'):
		poly = thisElementArrayStim._borderPix  # e.g., outline vertices
	else:
		poly = thisElementArrayStim.verticesPix[:, :, 0:2]  # e.g., tesselated vertices

	return any( np.fromiter( ( visual.helpers.pointInPolygon(xy[0], xy[1], thisPoly) for thisPoly in poly), np.bool ) )

def get_keypress():
    keys = event.getKeys()
    if keys:
        return keys[0]
    else:
        return None

def shutdown():
    win.close()
    core.quit()
########################################################################################################################
#                                                   Experiment Setup                                                   #
########################################################################################################################

# Kill the explorer if we are on a windows machine, if not kill EEG use
if os.name == 'nt':
    os.system("taskkill /im explorer.exe")

# Collect the subject number and create the subject output file
subj_num, subj_file = setup_subject()

# Create save directory if it does not already exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Seed random with the subject number so we can recreate the experiment
random.seed(int(subj_num))

# Write output headers to subject save file
with open(subj_file, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(HEADER_LIST)

# Set up psychopy
win, mon, event_clock, key_resp, mouse = set_psychopy()

# Set the experiment trials
test_set = set_trials()

# Build all experiment stimuli, *Note this needs to be done before experiment runtime to ensure proper timing
display_text = visual.TextStim(win=win, ori=0, name='text', text="", font='Arial', pos=[0, 0],
                               wrapWidth=TEXT_WRAP, color=TEXT_color, colorSpace='rgb', opacity=1, depth=-1.0)

fixation = visual.Circle(win, pos=[0, 0], radius=FIXATION_SIZE, lineColor=FIX_color, fillColor=FIX_color)

stimuli = []
for target in range(6):
    stimuli.append(visual.Rect(win, width=STIM_SIZE, height=STIM_SIZE, fillColorSpace='rgb', lineColorSpace=''))

# Present instructions for the experiment
display_message(win, fixation, display_text, INS_MSG)

# Open the output file reader for writing
csv_file = open(subj_file, 'a')
writer   = csv.writer(csv_file)

# Set required run time variables
current_trial = 0

########################################################################################################################
#                                                  Experiment Run-time                                                 #
########################################################################################################################

for trial in test_set:
    key = get_keypress()
    if key == 'escape':
        shutdown()
    else:
        pass
    current_trial += 1

    # Present a break message every 25 trials
    if current_trial % 25 == 0 and current_trial != 0:
        display_message(win, fixation, display_text, BREAK_MSG)

    # Set up ITI screen
    fixation.setAutoDraw(True)

    # Run ITI screen
    event_clock.reset()
    win.flip()

    # Set up presentation stimuli screen
    for target in range(trial.num_stimuli):
        stimuli[target].setPos((trial.stim_positions[target][0], trial.stim_positions[target][1]))

        if trial.stim_colors[target]==[-1,-1,-1]:
            ix = np.random.choice(np.arange(0,n_circ),1,replace=False)[0]
            print (ix,len(rgb))
            stimuli[target].setFillColor(rgb[ix,:,:][0], colorSpace = 'rgb')
        else:
            stimuli[target].setFillColor(trial.stim_colors[target], colorSpace = 'rgb')
        stimuli[target].setLineColor('Black')
        stimuli[target].setAutoDraw(True)

    # Wait until ITI screen is done
    while event_clock.getTime() < ITI_TIME:
        pass

    # Run presentation stimuli screen
    event_clock.reset()
    win.flip()

    probe_stim = stimuli
    # Set up ISI screen
    for target in range(trial.num_stimuli):
        probe_stim[target].setFillColor(None)
        probe_stim[target].setLineColor('Black')
        probe_stim[target].setAutoDraw(True)

    # Wait until presentation stimuli screen is done
    while event_clock.getTime() < STIM_TIME:
        pass

    # Run ISI screen
    event_clock.reset()
    win.flip()

    # Set up memory probe screen
    xys = []
    loop_radius = 8
    n_circ =360
    for theta in np.linspace(0,360,n_circ, endpoint = False):
        xys.append((loop_radius * np.cos(theta*np.pi/180),
                    loop_radius * np.sin(theta*np.pi/180)))

    stim = visual.ElementArrayStim(win, nElements=n_circ,sizes=0.9,xys = xys,
                           elementTex = None, elementMask = "circle",
                           colors=rgb.reshape(n_circ,3),colorSpace='rgb', interpolate = True)

    mask1 = visual.Circle(win,radius = loop_radius-0.5, units = 'deg')
    mask1.units = 'deg'
    mask2 = visual.Circle(win,radius = loop_radius+0.5, units = 'deg')
    mask2.units = 'deg'

    while event_clock.getTime() < 0.1:
        pass
    event_clock.reset()
    win.flip()
    probe = random.sample(np.arange(0,trial.num_stimuli),1)[0]
    for target in range(trial.num_stimuli):
        if target==probe:
            probe_Color = trial.stim_colors[target]
            probe_stim[target].setFillColor(None)
            probe_stim[target].setLineColor('White')
            probe_stim[target].setPos((trial.stim_positions[target][0], trial.stim_positions[target][1]))
            probe_stim[target].setAutoDraw(True)
        else:
            probe_stim[target].setFillColor(None)
            probe_stim[target].setLineColor('Black')
            probe_stim[target].setPos((trial.stim_positions[target][0], trial.stim_positions[target][1]))
            probe_stim[target].setAutoDraw(True)
    win.flip()
    while event_clock.getTime() < 0.1:
        pass
    event_clock.reset()
    stim.setAutoDraw(True)
    event.clearEvents()

    # Wait until ISI screen is done
    while event_clock.getTime() < DELAY_TIME:
        pass

    # Run memory probe screen, wait for key response, and record
    key_resp.clock.reset()
    win.flip()
    buttons = mouse.getPressed()
    pressed = 1
    tic = event_clock.getTime()
    while pressed:
        for target in range(trial.num_stimuli):
            mousePos = tuple(mouse.getPos()) # Get mouse coordinates
            loc = np.argmin(euclidean_distances(stim.xys,[mousePos]))
            min_dist = np.min(euclidean_distances(stim.xys,[mousePos]))
            if target==probe:
                if not mask1.contains(mouse) and mask2.contains(mouse):
                    probe_stim[target].setFillColor(rgb[loc,:,:][0], colorSpace= 'rgb')
                else:
                    probe_stim[target].setFillColor(None)

                probe_stim[target].setLineColor('White')
                probe_stim[target].setPos((trial.stim_positions[target][0], trial.stim_positions[target][1]))
                probe_stim[target].setAutoDraw(True)
            else:
                probe_stim[target].setFillColor(None)
                probe_stim[target].setLineColor('Black')
                probe_stim[target].setPos((trial.stim_positions[target][0], trial.stim_positions[target][1]))
                probe_stim[target].setAutoDraw(True)

        win.flip()
        buttons = mouse.getPressed()
        if any(buttons):
            sub_resp_Color = rgb[loc,:,:][0]
            judgement_Error = euclidean_distances([sub_resp_Color],[probe_Color]).flatten()
            pressed = 0
            toc = event_clock.getTime()
            response = toc-tic
            stim.setAutoDraw(False)
            #win.close()

    # Output trial results to file
    output = [subj_num, current_trial, trial.num_stimuli]

    for target in range(6):
        try:
            output.append(color_NAMES[str(trial.stim_colors[target])])
        except:
            output.append('NaN')

    output.extend([probe_Color,sub_resp_Color,judgement_Error, response])

    writer.writerow(output)
    csv_file.flush()

    for target in range(trial.num_stimuli):
        stimuli[target].setAutoDraw(False)
# end of experiment

# Close the csv file
csv_file.close()

# Thank subject
display_message(win, fixation, display_text, THANK_MSG)

# Close the experiment
win.close()
core.quit()
