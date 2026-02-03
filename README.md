# PTC Pipeline
This is a Bachelors thesis in astronomy for classification of a CMOS for the RAMSES mission.

The code can be used for general PTC generation for CCD or CMOS sensors.


## Install:
To get started, make sure to install the required packages first by executing the following command after opening the project folder.
>pip install -r requirements.txt
It is recomended, to create a virtual environment first.

## Usage:
To make a PTC, create a new folder in **src/data/** with the name of your camera.
In there, place the corresponding **.fit** files with the raming as follows:
>bias_"*custom name*".fit
>dark_"*custom name*".fit
>light_"*custom name*".fit

the dark frames are used for generating a dark-current analysis, and not for the PTC itself.

To run the code, open the project in the terminal, and run the main.py there.
### Parameters:
>-l, --list: lists all available Cameras
>-c *CAMERA*, --camera *CAMERA*: set the camera to analyze

The camera parameter corresponds to the folders name, in which the data is stored.



### TODO:
Future plans might include to generalize the code further, to lupport more formats, such os general RAW images.

