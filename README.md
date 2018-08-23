# Change Detection Task

This script runs a standard color change detection task (modeled after Luck & Vogel, 1997, Nature). The script was written by Thomas Nelson for the Visual Cognitive Neuroscience Lab at Brock University. Feel free to use this as a starting point for creating your own change detection experiments.

Modified for the continuous color WM task by Aditya Upadhyayula (supadhy6@jhu.edu)

## Dependencies

- [PsychoPy](http://www.psychopy.org/)
- [Python](http://www.python.org/)
- [scikit learn] (http://scikit-learn.org/stable/)
## Documentation

All documentation can be found [here]().

## Version Log

### V-1.3 (Master)
- Adapted for the continuous color WM task (Wilken & Ma, 2004)
- Some differences between the task in the original paper, and the code implemented here
	* Colorspace used is HSV colorpalette. Psychopy 3.0 does not support CIELAB colorspace yet

### V-1.2 (Master)
- Path and file creation now work cross platform

### V-1.1
- Code has been cleaned up
- Improved runtime efficiency

### V-1.0
- Initial version
