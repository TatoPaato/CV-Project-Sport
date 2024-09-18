#   _____ _        _    ____
#  |  ___| |      / \  / ___|___
#  | |_  | |     / _ \| |  _/ __|
#  |  _| | |___ / ___ \ |_| \__ \
#  |_|   |_____/_/   \_\____|___/


SAVE_PARAM = False     # To save camera parameters to file after calibration
SHOW_TEST_FRAME = True  # To show the before/after intrinsic calibration of the camera
GET_SAMPLE = False      # True: scan videos and get sample chessboard frames, False: get chessboard from existing frames
READ_PARAM = True       # Read parameters from pickle file
CROP = True             # To crop undistorted images

#   ____  _____ _____ _____ ___ _   _  ____ ____
#  / ___|| ____|_   _|_   _|_ _| \ | |/ ___/ ___|
#  \___ \|  _|   | |   | |  | ||  \| | |  _\___ \
#   ___) | |___  | |   | |  | || |\  | |_| |___) |
#  |____/|_____| |_|   |_| |___|_| \_|\____|____/

NUM_OF_SAMPLES = 8
FRAME_SKIP = 30
SQUARE_SIZE = 1  # 0.028 (m) # 28 (mm)
# CAMS basic info, position in (m)
CAMS = {
    "CAM1": {"number": 1, "WIDE": False, "position": [-15.1, -17.9, 6.20]},
    "CAM2": {"number": 2, "WIDE": True, "position": [0.0, -17.9, 6.2]},
    "CAM3": {"number": 3, "WIDE": False, "position": [-22.3, -10.2, 6.60]},
    "CAM4": {"number": 4, "WIDE": False, "position": [14.8, -18.1, 6.20]},
    "CAM5": {"number": 5, "WIDE": False, "position": [-22.0, 10.2, 6.80]},
    "CAM6": {"number": 6, "WIDE": True, "position": [0.0, 10.2, 6.35]},
    "CAM7": {"number": 7, "WIDE": False, "position": [24.8, 0.0, 6.40]},
    "CAM8": {"number": 8, "WIDE": False, "position": [22, 10, 6.35]},
    "CAM12": {"number": 12, "WIDE": True, "position": [24.8, -10, 6.90]},
    "CAM13": {"number": 13, "WIDE": False, "position": [-22.0, 0.0, 7.05]},
}

SCALE = 0.2 # Reduce plot resolution for faster plotting 0 < Scale < 1
# List of Camera to handle/calibrate/plot
TO_CAL = [
    "CAM1",
    "CAM2",
    "CAM3",
    "CAM4",
    "CAM5",
    "CAM6",
    "CAM7",
    "CAM8",
    "CAM12",
    "CAM13"
]
# Sizes of calibration chessboards, edit just if using different calibration videos!
CHESSBOARD_SIZES = {
    1: (5, 7),
    2: (5, 7),
    3: (5, 7),
    4: (5, 7),
    5: (6, 9),
    6: (6, 9),
    7: (5, 7),
    8: (6, 9),
    12: (5, 7),
    13: (5, 7),
}

#   ____       _   _         
#  |  _ \ __ _| |_| |__  ___ 
#  | |_) / _` | __| '_ \/ __|
#  |  __/ (_| | |_| | | \__ \
#  |_|   \__,_|\__|_| |_|___/
                           

# Game videos
VIDEO_FOLDER = "./Video"
# Video format
FORMAT = ".mp4"

# Video of calibration with chessboards
CALIBRATION_FOLDER = "./Video/Calibration"
# Destination of sampled frames from calibration videos
SAMPLE_FOLDER = "./Cameras/ChessBoardSamples"
SAMPLE_PATH = "./Video/GAME/Samples/"
UNDISTORTED_SAMPLES = "./Cameras/UndistortedSamples"

# Folder to store camera parameters
PARAMETER_FOLDER = "./Cameras/Parameters"
ERROR_FOLDER = "./Cameras/Errors"
PLOT_FOLDER = "./Plots"
# PLOT settings
SHOW_COURT = True
COURT_IMG_LR = "./MISC/Court_low_res.png"
COURT_IMG_MR = "./MISC/Court_mid_res.png"

COURT_IMG_QUALITY = 1  # integer.  1: highest -> slower rendering, 10: terrible-> faster
COURT_IMG_XL = "./MISC/Court_stands.png"

#   ______        __  ____       _       _
#  |  _ \ \      / / |  _ \ ___ (_)_ __ | |_ ___
#  | |_) \ \ /\ / /  | |_) / _ \| | '_ \| __/ __|
#  |  _ < \ V  V /   |  __/ (_) | | | | | |_\__ \
#  |_| \_\ \_/\_/    |_|   \___/|_|_| |_|\__|___/

# KEY point on the court/sports hall to use in correspondences
import numpy as np

COURT_POINTS = np.array(
    [
        [-9, 4.5, 0.0], # START VolleyBall court points
        [-3, 4.5, 0.0],
        [0, 4.5, 0.0],
        [3, 4.5, 0.0],
        [9, 4.5, 0.0],
        [-9, -4.5, 0.0],
        [-3, -4.5, 0.0],
        [0, -4.5, 0.0],
        [3, -4.5, 0.0],
        [9, -4.5, 0.0],  # END VolleyBall court
        [0.0,1.8,0.0],   # START BasketBall court points
        [0.0,-1.8,0.0],
        [-8.2, 1.8, 0.0],
        [8.2, 1.8, 0.0],
        [-8.2, -1.8, 0.0],
        [8.2, -1.8, 0.0],
        [-8.2, 2.45, 0.0],
        [8.2, 2.45, 0.0],
        [-8.2, -2.45, 0.0],
        [8.2, -2.45, 0.0],
        [-5, -7.5, 0.0],
        [5, -7.5, 0.0],
        [-5.675, 7.5, 0.0],
        [5.675, 7.5, 0.0] # END BasketBall court points
        
    ]
)


#   ____   ___    _   _  ___ _____   _____ ____ ___ _____ 
#  |  _ \ / _ \  | \ | |/ _ \_   _| | ____|  _ \_ _|_   _|
#  | | | | | | | |  \| | | | || |   |  _| | | | | |  | |  
#  | |_| | |_| | | |\  | |_| || |   | |___| |_| | |  | |  
#  |____/ \___/  |_| \_|\___/ |_|   |_____|____/___| |_|  
                                                    
MIN_POINTS = 6 # Minimum points for Ext Calibration 
"""
cv2.solvePnP
------------
    Initial solution for non-planar "objectPoints" needs at least 6 points and uses the DLT algorithm. Initial solution for planar "objectPoints" needs at least 4 points and uses pose from homography decomposition.
    In this code planar 3D points are used, actually are coplanar, however this guarantee compatibility for the Calibration with points in 3D space.
    
cv2.findHomography
------------------
    Requires at least 4 points, however the minimum of 6 is used, since correspondences are used for both calibration and plane to plane mapping. 

"""