import CameraUtils as cu
import configuration as conf
import numpy as np
import matplotlib.pyplot as plt
# import cv2
import sys
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


court_points = conf.COURT_POINTS

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

#  save errors to file
for cam in camera_list:
    cam.PosError()
    cam.RepError()
    cam.FindHomography()
    
cu.Camera.SaveErrors(camera_list,conf.ERROR_FOLDER)


# # Save Undistorsion samples
# for camera in camera_list:
#     camera.TestUndistorsion()

plot3d = cu.Camera.PlotMultipleCameras(camera_list)
plot3d.savefig(conf.PLOT_FOLDER+"/plot3d.png")
plot2d = cu.Camera.PlotMultipleCameras2D(camera_list)
plot2d.savefig(conf.PLOT_FOLDER+"/plot2d.png")

# plot scatter of errors
re= np.array([cam.rep_err for cam in camera_list])
pe= np.array([cam.pos_err for cam in camera_list])
ne= [cam.camera_number for cam in camera_list]

fig = plt.figure() #facecolor="#2e2e2e")
ax = fig.add_subplot()
ax.set_title("Rep. Error vs Loc. Error")#,color = "white")

for i in range (len(ne)):
    ax.scatter(re[i],pe[i],marker="o",label=f"CAM:{ne[i]}")
ax.set_xlabel("Reprojection error (pixels)")#,color = "white")
ax.set_ylabel("Position error (m)")#,color = "white")

# ax.tick_params(axis='both', colors='white')
ax.grid(True)
ax.legend(loc=0)
fig.tight_layout()
fig.savefig("./Plots/EvE.png")
fig.savefig("./Report/Media/plots/EvE.png")
# plt.show()

# pc= cu.PlotCameras()

# for cam in camera_list:
#     pc.AddView(cam)
    
# pc.InitPlot()
# pc.PlotImages()
# pc.ShowViews()