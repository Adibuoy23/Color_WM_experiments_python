# -*- coding: utf-8 -*-

# change_detection.py
# Visual Cognitive Neuroscience Lab
# Brock University
#
# Created by Thomas Nelson <tn90ca@gmail.com>
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
from psychopy import visual, core, event, gui, monitors


########################################################################################################################
#                                                  Program Constants                                                   #
########################################################################################################################

# I have these settings configured for my macbook you will need to change them to suit your monitor
MONITOR_WIDTH = 28.5  # The width of the display on your monitor
MONITOR_DISTANCE = 52  # The veiwing distance from the user to the monitor
MONITOR_RESOLUTION = [1440, 900]  # The resolution of your monitor display

EXP_NAME = "change_detection"  # The name of the experiment for save files
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
TRIAL_colorS1 = [[-1, -1, -1],  # Black
                 [-1, -1, 1],   # Blue
                 [-1, 1, -1],   # Green
                 [-1, 1, 1],    # Cyan
                 [1, -1, -1],   # Red
                 [1, -1, 1],    # Purple
                 [1, 1, -1],    # Yellow
                 [1, 1, 1]]     # White
TRIAL_colorS2 = [[-1, -1, -1],  # Black
                 [-1, -1, 1],   # Blue
                 [-1, 1, -1],   # Green
                 [-1, 1, 1],    # Cyan
                 [1, -1, -1],   # Red
                 [1, -1, 1],    # Purple
                 [1, 1, -1],    # Yellow
                 [1, 1, 1]]     # White


color_NAMES  = {str([-1,-1,-1]) : 'Black',
                 str([-1,-1,1])  : 'Blue',
                 str([-1,1,-1])  : 'Green',
                 str([-1,1,1])   : 'Cyan',
                 str([1,-1,-1])  : 'Red',
                 str([1,-1,1])   : 'Purple',
                 str([1,1,-1])   : 'Yellow',
                 str([1,1,1])    : 'White'}

NUM_TYPE = 6  # Number of different trial types
NUM_REPS = 50  # Number of repetitions for each different trial type

# Note that all sizing is in visual degrees
FIXATION_SIZE   = 0.1  # Size of the fixation at the center of the screen in visual degree
STIM_POS_RADIUS = 4  # Number of visual degrees between the center and stimuli
STIM_SIZE       = 0.65  # Size of the stimuli in visual degrees, length and width
STIM_THICKNESS  = 1  # The thickness of the outline of the stimuli

TEXT_HEIGHT = 1   # The height in visual degrees of instruction text
TEXT_WRAP   = 50  # The character limit of each line of text before word wrap

# Note that all timing is in seconds
ITI_TIME   = 0.5  # The time in seconds between trials
STIM_TIME  = 0.5  # The time in seconds to display the stimuli
DELAY_TIME = 0.9  # The time in seconds between stimuli and probe
BREAK_TIME = 0.75  # The time in seconds between break end and trial start

INS_MSG   = "You will be presented with colored squares, try to remember their colors.\n\n"
INS_MSG  += "For each trial, there will follow a second set of squares in the same locations.\n\n"
INS_MSG  += "If there was any change in color to the second set of squares from the first, press "
INS_MSG  += "the z key,\n\nIf they have not changed color, press the m key\n\nPress any key when "
INS_MSG  += "you are ready to begin."
BREAK_MSG = "Take a quick break. When you are ready to continue, press any key."
THANK_MSG = "Thank you for your participation. Please go find the experimenter."

# This is a list of the column headers for the output file
HEADER_LIST = ['Subject_Number', 'Trial_Number', 'Number_of_Stim',
                'Stim_color_1', 'Stim_color_2', 'Stim_color_3', 'Stim_color_4',
                'Stim_color_5', 'Stim_color_6', 'Stim_color_7', 'Stim_color_8',
                'Stim_color_9', 'Stim_color_10', 'Stim_color_11', 'Stim_color_12',
                'Probe_color_1', 'Probe_color_2', 'Probe_color_3', 'Probe_color_4',
                'Probe_color_5', 'Probe_color_6', 'Probe_color_7', 'Probe_color_8',
                'Probe_color_9', 'Probe_color_10', 'Probe_color_11', 'Probe_color_12',
                'Change_Present', 'Subject_Response', 'Response_Time']


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
            self.num_stimuli = 1
        elif trial_num == 1:
            self.num_stimuli = 2
        elif trial_num == 2:
            self.num_stimuli = 3
        elif trial_num == 3:
            self.num_stimuli = 4
        elif trial_num == 4:
            self.num_stimuli = 8
        elif trial_num == 5:
            self.num_stimuli = 12


    # end def __init__

    def set_positions(self):
        """This function will determine the location (left or right) for the memory sample and distraction sample. Uses
        even and odd numbers to ensure an even distribution of left and right positioning. Also generates the
        coordinates for the gui and results print out.
        """

        # Generate a list of 12 positions around the center
        for pos in xrange(12):
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
        random.shuffle(TRIAL_colorS1)
        random.shuffle(TRIAL_colorS2)
        TRIAL_colorS = TRIAL_colorS1 + TRIAL_colorS2
        # Create the lists of stimuli and probe colors
        for color in TRIAL_colorS:
            self.stim_colors.append(color)
            self.probe_colors.append(color)

        # If a change is present replace on of the colors in the probe list with a new color
        if (self.rep_num % 2) == 0:
            self.change = True
            rand_1 = random.randint(0, self.num_stimuli-1)
            rand_2 = random.randint(self.num_stimuli, 13)
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

    for rep in xrange(NUM_REPS):
        for trial in xrange(NUM_TYPE):
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
        print "FIXATION SCREEN:", FT
# end def display_fixation


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
display_text = visual.TextStim(win=win, ori=0, name='text', text="", font='Arial', pos=[0, 0], height=TEXT_HEIGHT,
                               wrapWidth=TEXT_WRAP, color=TEXT_color, colorSpace='rgb', opacity=1, depth=-1.0)

fixation = visual.Circle(win, pos=[0, 0], radius=FIXATION_SIZE, lineColor=FIX_color, fillColor=FIX_color)

stimuli = []
for target in xrange(12):
    stimuli.append(visual.Rect(win, width=STIM_SIZE, height=STIM_SIZE, fillColorSpace='rgb', lineColorSpace='rgb'))

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
    for target in xrange(trial.num_stimuli):
        stimuli[target].setPos((trial.stim_positions[target][0], trial.stim_positions[target][1]))
        stimuli[target].setFillColor(trial.stim_colors[target])
        stimuli[target].setLineColor(trial.stim_colors[target])
        stimuli[target].setAutoDraw(True)

    # Wait until ITI screen is done
    while event_clock.getTime() < ITI_TIME:
        pass

    # Run presentation stimuli screen
    event_clock.reset()
    win.flip()

    # Set up ISI screen
    for target in xrange(trial.num_stimuli):
        stimuli[target].setAutoDraw(False)

    # Wait until presentation stimuli screen is done
    while event_clock.getTime() < STIM_TIME:
        pass

    # Run ISI screen
    event_clock.reset()
    win.flip()

    # Set up memory probe screen
    for target in xrange(trial.num_stimuli):
        stimuli[target].setFillColor(trial.probe_colors[target])
        stimuli[target].setLineColor(trial.probe_colors[target])
        stimuli[target].setPos((trial.stim_positions[target][0], trial.stim_positions[target][1]))
        stimuli[target].setAutoDraw(True)
    event.clearEvents()

    # Wait until ISI screen is done
    while event_clock.getTime() < DELAY_TIME:
        pass

    # Run memory probe screen, wait for key response, and record
    key_resp.clock.reset()
    win.flip()

    while True:
        # Check for response keys or quit
        if event.getKeys(["z"]):
            response  = True
            resp_time = key_resp.clock.getTime()
            break
        elif event.getKeys(["m"]):
            response  = False
            resp_time = key_resp.clock.getTime()
            break
        elif event.getKeys(["escape"]):
            core.quit() # If escape key is hit then safely close program

    # Output trial results to file
    output = [subj_num, current_trial, trial.num_stimuli]

    for target in xrange(12):
        try:
            output.append(color_NAMES[str(trial.stim_colors[target])])
        except:
            output.append('NaN')

    for target in xrange(12):
        try:
            output.append(color_NAMES[str(trial.probe_colors[target])])
        except:
            output.append('NaN')

    output.extend([trial.change, response, resp_time])

    writer.writerow(output)
    csv_file.flush()

    for target in xrange(trial.num_stimuli):
        stimuli[target].setAutoDraw(False)
# end of experiment

# Close the csv file
csv_file.close()

# Thank subject
display_message(win, fixation, display_text, THANK_MSG)

# Close the experiment
win.close()
core.quit()
