#   ___ _   _ _____ ____  ___ _   _ ____ ___ ____    ____    _    _     ___ ____  ____      _  _____ ___ ___  _   _ 
#  |_ _| \ | |_   _|  _ \|_ _| \ | / ___|_ _/ ___|  / ___|  / \  | |   |_ _| __ )|  _ \    / \|_   _|_ _/ _ \| \ | |
#   | ||  \| | | | | |_) || ||  \| \___ \| | |     | |     / _ \ | |    | ||  _ \| |_) |  / _ \ | |  | | | | |  \| |
#   | || |\  | | | |  _ < | || |\  |___) | | |___  | |___ / ___ \| |___ | || |_) |  _ <  / ___ \| |  | | |_| | |\  |
#  |___|_| \_| |_| |_| \_\___|_| \_|____/___\____|  \____/_/   \_\_____|___|____/|_| \_\/_/   \_\_| |___\___/|_| \_|
                                                                                                                  

import CameraUtils as cu
import configuration as conf
import termcolor
import os
import time

# create a dictionary with the selected CAMS
cam_to_calibrate = {CAM: conf.CAMS[CAM] for CAM in conf.TO_CAL if CAM in conf.CAMS }
print(f"Cameras to calibrate: {list(cam_to_calibrate.keys())}")
camera_list = []

# Create instances for each camera and set chessboard size
for key, values in cam_to_calibrate.items():
    camera_list.append(cu.Camera(camera_number=values["number"],approximate_position=values["position"],WIDE_LENS=values["WIDE"]))
    camera_list[-1].chessboard_size = conf.CHESSBOARD_SIZES[values["number"]]

    
    
if conf.GET_SAMPLE:
    start_time = time.time()
    for camera in camera_list:
        camera.GetSampleFrames(
            video_dir=conf.CALIBRATION_FOLDER, frame_skip=15, out_dir=conf.SAMPLE_FOLDER
        )

    stop_time = time.time()
    termcolor.cprint(f"Sample frames retrieved in {stop_time-start_time} seconds", "blue")


elif os.path.exists(conf.SAMPLE_FOLDER):
    for camera in camera_list:
        camera.GetCornersFromSamples(conf.SAMPLE_FOLDER)
else:
    termcolor.cprint("Cannot find sample folder", "red")


# Calibrate cameras
for camera in camera_list:
    
    camera.CalibrateCamera()
    camera.NewOptimalCameraMatrix()
    camera.RepError()

    if conf.SAVE_PARAM:
        camera.SaveParameters(conf.PARAMETER_FOLDER)


# Show undistorsion results 
if conf.SHOW_TEST_FRAME:
    for camera in camera_list:
        camera.TestUndistorsion()
