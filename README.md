# CV-Project: 3D camera calibration (geometry and 3D reconstruction)
Author:  Bonan Marco 
## Contents
- [CV-Project: 3D camera calibration (geometry and 3D reconstruction)](#cv-project-3d-camera-calibration-geometry-and-3d-reconstruction)
  - [Contents](#contents)
  - [Project goal](#project-goal)
  - [Project delivery](#project-delivery)
  - [CODE](#code)
  - [Materials](#materials)
  - [Useful links](#useful-links)
    - [Coding](#coding)
    - [Game rules (court lines)](#game-rules-court-lines)

## Project goal
1. Create a 3D reconstruction of the camera positions relative to the field
   1. Intrinsic camera calibration
   2. Extrinsic camera calibration
2. Develop a tool such that by clicking on a point on the field surface, the point is highlighted in all camera views.
3. **BONUS** track ball motion in 3D (not necessary since project done individually)


## Project delivery
Code (zip folder + git) (this repository) containing:
- link to the Dataset ([the videos](#materials)) (files are too large to include in the archive)
- Output images, videos, `.pkl` files of calibrated cameras.
- Documentation [HTML](doc/html/index.html), [PDF](doc/latex/cv-project-2024.pdf)
- Short written [report](CV-Project-Bonan.pdf) with results
- 5' video including:
  - Motivations and background
  - Methodology
  - Results

## CODE
- Make sure to **[setup](Setting_up.md)** the environment  :gear: 
- Get an overview of the code of this repository reading the documentation ([HTML](doc/html/index.html), [PDF](doc/latex/cv-project-2024.pdf))

## Materials
- [Starting point github](https://github.com/Elia-Tomaselli/CV-CameraCalibration)
- [Starting point drive](https://drive.google.com/drive/folders/1P6Bs7bx_CGXWCbx_5wyAnqc8fPY2SGxO?usp=sharing)
- [Videos](https://drive.google.com/drive/folders/11RhLrWwb_tH9uLBCGraR55N0_Lnnaww-?usp=sharing)
- [Calibration videos](https://drive.google.com/drive/folders/15_CCC2mGQZmn3WqdCiWEGJuTSz584Ch0?usp=sharing)


## Useful links

### Coding
- :camera:  [Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- :camera:  [OpenCV 4.10 documentation ](https://docs.opencv.org/4.10.0/)
- :fish:  [OpenCV Fisheye](https://docs.opencv.org/4.10.0/db/d58/group__calib3d__fisheye.html)
### Game rules (court lines)  
- :basketball:  [FIBA Rules](https://www.fiba.basketball/documents/official-basketball-rules/current.pdf)
- :volleyball:  [FIVB Rules](https://www.fivb.com/wp-content/uploads/2024/03/FIVB-Volleyball_Rules_2021_2024_pe.pdf)

