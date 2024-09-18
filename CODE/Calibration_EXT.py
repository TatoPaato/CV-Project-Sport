#   _______  _______ ____  ___ _   _ ____ ___ ____    ____    _    _     ___ ____  ____      _  _____ ___ ___  _   _
#  | ____\ \/ /_   _|  _ \|_ _| \ | / ___|_ _/ ___|  / ___|  / \  | |   |_ _| __ )|  _ \    / \|_   _|_ _/ _ \| \ | |
#  |  _|  \  /  | | | |_) || ||  \| \___ \| | |     | |     / _ \ | |    | ||  _ \| |_) |  / _ \ | |  | | | | |  \| |
#  | |___ /  \  | | |  _ < | || |\  |___) | | |___  | |___ / ___ \| |___ | || |_) |  _ <  / ___ \| |  | | |_| | |\  |
#  |_____/_/\_\ |_| |_| \_\___|_| \_|____/___\____|  \____/_/   \_\_____|___|____/|_| \_\/_/   \_\_| |___\___/|_| \_|

# Imports
import CameraUtils as cu
import configuration as conf
import termcolor
import sys
import numpy as np

court_points = conf.COURT_POINTS

# create a dictionary with the selected CAMS
cam_to_calibrate = {}
for CAM in conf.TO_CAL:
    if CAM in conf.CAMS:
        cam_to_calibrate[CAM] = conf.CAMS[CAM]
"""
OR with list comprehension:
cam_to_calibrate = {CAM: conf.CAMS[CAM] for CAM in conf.TO_CAL if CAM in conf.CAMS}
"""
print(f"Cameras to calibrate: {list(cam_to_calibrate.keys())}")
camera_list = []

# Load attributes
for key in cam_to_calibrate:
    camera_list.append(
        cu.Camera.LoadCamera(param_dir=conf.PARAMETER_FOLDER, camera=key)
    )

for cam in camera_list:
    if cam.INT_CAL:
        print(f"CAM {cam.camera_number} is intrinsic-calibrated")
        cam.EXT_CAL = False
    else:
        termcolor.cprint(
            f"Warning: CAM {cam.camera_number} is not intrinsic-calibrated, calibrate it first",
            "yellow",
            attrs=["bold"],
        )
        sys.exit(1)


for cam in camera_list:
    if not cam.EXT_CAL:
        path = conf.SAMPLE_PATH + f"Cam{cam.camera_number}.jpg"
        # IMAGE = cv2.imread(path)
        while True:
            point = cam.GetCorrespondences(court_points, path)
            if point == []:
                termcolor.cprint("No points selected!", "yellow")
                break

            img_points = np.array(point)

            nan_mask = np.isnan(img_points).any(axis=1)
            acquired_points = len(img_points)
            valid_points = img_points[~nan_mask].shape[0]

            if valid_points >= conf.MIN_POINTS:

                # keeping the first points in case acquisition is early-terminated by user
                cam.corr_world_points = court_points[:acquired_points][~nan_mask]
                cam.corr_image_points = img_points[~nan_mask]

                print(
                    f"Corresponding point for camera {cam.camera_number} "
                    "successfully retrieved"
                )

                termcolor.cprint(
                    f"Successfully retrieved corresponding points for the camera",
                    "green",
                    attrs=["bold"],
                )
                # Perform extrinsic calibration
                # print(f"Performing extrinsic calibration of camera {cam.camera_number}")
                cam.ExtCalibration()

                # Compute Positioning error with reference position
                cam.PosError()
                if conf.SAVE_PARAM:
                    cam.SaveParameters(conf.PARAMETER_FOLDER)

                break

            elif 0 <= valid_points < conf.MIN_POINTS:
                termcolor.cprint(
                    f"The number of valid selected point is {valid_points}, you need at least a number of point >= {conf.MIN_POINTS} in order to calibrate the camera",
                    "yellow",
                )
                sys.exit("Exiting...")

            else:
                termcolor.cprint(
                    f"{valid_points} is an invalid num of point. Repeat correspondences retrieve process",
                    "blue",
                )
                sys.exit("Exiting...")

    elif cam.EXT_CAL:
        print(f"Camera {cam.camera_number} is already extrinsic-calibrated")


termcolor.cprint("All done!", "cyan")
