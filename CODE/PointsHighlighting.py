#   _   _ ___ ____ _   _ _     ___ ____ _   _ _____   ____   ___ ___ _   _ _____ ____  
#  | | | |_ _/ ___| | | | |   |_ _/ ___| | | |_   _| |  _ \ / _ \_ _| \ | |_   _/ ___| 
#  | |_| || | |  _| |_| | |    | | |  _| |_| | | |   | |_) | | | | ||  \| | | | \___ \ 
#  |  _  || | |_| |  _  | |___ | | |_| |  _  | | |   |  __/| |_| | || |\  | | |  ___) |
#  |_| |_|___\____|_| |_|_____|___\____|_| |_| |_|   |_|    \___/___|_| \_| |_| |____/ 
                                                                                     
import CameraUtils as cu
import configuration as conf
import sys

# create a dictionary with the selected CAMS
cam_to_calibrate = {CAM: conf.CAMS[CAM] for CAM in conf.TO_CAL if CAM in conf.CAMS}
print(f"Cameras to calibrate: {list(cam_to_calibrate.keys())}")
camera_list = []

# Load attributes
for key in cam_to_calibrate:
    camera_list.append(
        cu.Camera.LoadCamera(param_dir=conf.PARAMETER_FOLDER, camera=key)
    )
    
for camera in camera_list:
    # check if all camera are calibrated
    if not (camera.INT_CAL and camera.EXT_CAL):
        print(80*"*")
        print("Camera must be both internally and externally calibrated in order to display plots!")
        print(f"CAM:{camera.camera_number}. \n INT_CAL-> {camera.INT_CAL} \n EXT_CAL-> {camera.EXT_CAL}")
        sys.exit()


# Instante plots and GUI
pc= cu.PlotCameras()

for cam in camera_list:
    pc.AddView(cam)
    
pc.InitPlot()
pc.PlotImages()
# Launch the GUI
pc.ShowViews()