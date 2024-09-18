import numpy as np
import cv2
import os
import termcolor
import random
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as path_effects
import sys
import configuration as conf
import pickle

random.seed(123)


class Camera:
    """
    A class used to represent the calibration process of a camera using chessboard images.

    Attributes
    ----------
    camera_number : int
        The identifier number for the camera being calibrated.
    FISHEYE : bool
        A flag indicating if the camera is a WideLens.
    chessboard_size : tuple or None
        The number of inner corners per chessboard row and column (e.g., (7, 6)).
    obj_points : list
        The 3D points in the real-world space.
    img_points : list
        The 2D points in the image plane.
    mtx : numpy.ndarray or None
        The camera matrix.
    dist : numpy.ndarray or None
        The distortion coefficients.
    rvecs : list or None
        The rotation vectors estimated for each pattern view.
    tvecs : list or None
        The translation vectors estimated for each pattern view.
    new_mtx : numpy.ndarray or None
        The refined camera matrix used for undistortion.
    INT_CAL : bool
        A flag indicating whether the camera has intrinsic calibration.
    EXT_CAL : bool
        A flag indicating whether the camera has extrinsic calibration.
    img_size : list
        The size of the images used for calibration (h, w, channels).
    roi : tuple or None
        The region of interest for the undistorted images.
    corr_world_points : numpy.ndarray or None
        A numpy array of points in 3D space, in the court reference frame.
    corr_image_points : numpy.ndarray or None
        A numpy array of points in the 2D image coordinates corresponding to
        the real-world points (corr_world_points).
    ext_mtx : numpy.ndarray or None
        The matrix representing the camera-to-world transformation.
    H_mtx : numpy.ndarray
        The homography matrix between the court and image plane.
    rep_err : float or None
        The overall reprojection error after intrinsic calibration, representing the quality of the calibration.
    pos_err : numpy.ndarray
        A vector representing the positional error in 3D space after extrinsic calibration, initialized to zeros.

    Methods
    -------
    __init__(camera_number, approximate_position, WIDE_LENS=False)
        Initializes the Camera object with a specific camera number, an approximate position in the real world,
        and whether or not it is WideLens. Sets all other attributes to default values.

    GetSampleFrames(video_dir, frame_skip, out_dir)
        Extracts sample frames from a video to use for camera calibration.

    GetCornersFromSamples(sample_folder)
        Reads existing images from a folder of existing samples to use for camera calibration.

    CalibrateCamera()
        Performs intrinsic calibration of the camera and sets the status to calibrated.

    SaveParameters(save_dir)
        Saves camera parameters to a .yaml file.

    ReadParameters(param_dir)
        Reads camera parameters from a .yaml file.

    NewOptimalCameraMatrix()
        Refines the camera matrix for the undistorsion.

    TestUndistorsion()
        Applies undistorsion to the image and shows the effect on an example.

    GetCorrespondences(world_points, image_path)
        Obtain correspondences between world points and image points through
        a manual user interface.

    AddManualCorrespondences(world_points, image_points)
        Manually add corresponding points to camera attributes.

    ExtCalibration()
        Perform external camera calibration to compute the transformation matrices
        between the world and camera coordinates, also compute the homography 
        matrix that map the image and court planes.

    PosError()
        Calculate and display the positional error between the approximate
        and estimated positions

    SaveErrors(`camera_list`, path)
        *This is a class method.*
        Save the positional and re-projection errors of a list of cameras to a CSV file

    PlotCamera(ax)
        Plot the camera's position and direction on a 3D axis. *This method is intended to be called by `PlotMultipleCameras()` class method

    PlotMultipleCameras(camera_list)
        *This is a class method*
        Plot the positions and orientations of multiple cameras on a 3D plot.

    PlotCamera2D(ax)
        Plot the camera's position and direction on an axis. *This method is intended to be called by `PlotMultipleCameras2D()` class method.

    PlotMultipleCameras2D(camera_list)
        *This is a class method*
        Plot the positions and orientations of multiple cameras on a 2D court image.

    PrintAttributes(skip_attributes)
        Prints all the attributes of the CameraInfo instance, except for those in the skip_attributes list.
        Method useful for quick attributes checks/debugging

    FindHomography()
        Compute the homography matrix from world coordinates (court plane) to image coordinates.

    Court2Image(coords)
        Convert court coordinates to image coordinates using the homography matrix

    Image2Court(coords)
        Convert image coordinates to court coordinates using the homography matrix.

    """

    def __init__(
        self,
        camera_number,
        approximate_position,
        WIDE_LENS=False,
    ):
        """
        Initializes the CameraCalibration object with a specific camera number
        and sets default values for other attributes.

        Parameters
        ----------
        camera_number : int
            The identifier number for the camera being calibrated.
        """

        self.camera_number = camera_number
        self.FISHEYE = WIDE_LENS
        self.chessboard_size = None
        self.obj_points = []
        self.img_points = []
        self.mtx = np.zeros((3, 3), np.float64)
        self.dist = np.zeros((4, 1), np.float64)
        self.rvecs = []
        self.tvecs = []
        self.new_mtx = None
        self.INT_CAL = False
        self.EXT_CAL = False
        self.img_size = []
        self.roi = None
        self.corr_world_points = None
        self.corr_image_points = None
        self.C2W_mtx = None
        self.W2C_mtx = None
        self.camera_position = None
        self.approximate_position = approximate_position
        self.H_mtx = None
        self.rep_err = None
        self.pos_err = np.zeros(3, np.float64)

    def GetSampleFrames(self, video_dir, frame_skip, out_dir):
        """
        Extracts sample frames from a video to use for camera calibration.

        This method reads a video file, samples frames at specified intervals,
        and detects chessboard corners in these frames. It then saves the frames
        and the detected points to be used for camera calibration.

        Parameters
        ----------
        video_dir : str
            The directory where the video file is located.
        frame_skip : int
            The number of frames to skip between each sampled frame.
        out_dir : str
            The directory where the sampled frames will be saved.

        Raises
        ------
        FileNotFoundError
            If the specified video file does not exist.
        RuntimeError
            If the video file cannot be opened or read.

        Notes
        -----
        - The video file is expected to be named in a specific format:
            "out{camera_number}F{format}",
            where 'camera_number' is the identifier
            of the camera and 'format' is defined in the configuration
            (example 'out3F.mp4' for camera number 3)
        - The method uses a chessboard pattern to detect corners in the frames.
        - It divides the frame into quadrants and saves a specified number of
            sample frames from each quadrant:
            - TL: Top-Left
            - TR: Top-Right
            - BL: Bottom-Left
            - BR: Bottom-Right

        - The method stops when the required number of samples from all
            quadrants have been collected. The required number of steps can be
            set in the setting script NUM_OF_SAMPLES.
        """
        termcolor.cprint(
            f"Initiating frame sampling for camera number {self.camera_number}",
            "yellow",
        )

        # create string with filename and add to the file path
        VIDEO_NAME = "/out" + str(self.camera_number) + "F"
        filepath = video_dir + VIDEO_NAME + conf.FORMAT
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)

        # open video
        video_capture = cv2.VideoCapture(filepath)
        # set frame counter
        frame_counter = 0
        # Print the number of frames in the video

        print(
            f"Number of frames in the video: {video_capture.get(cv2.CAP_PROP_FRAME_COUNT)}"
        )

        # Save sampled frames to folder
        sample_dir = out_dir + VIDEO_NAME
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        # clear folder from previous samples
        for file in os.listdir(sample_dir):
            os.remove(os.path.join(sample_dir, file))
        # Dictionary of quadrant
        quadrant_frame_counter = {
            "TL": 0,
            "BL": 0,
            "TR": 0,
            "BR": 0,
        }  # is a dictionary the best??
        if conf.NUM_OF_SAMPLES:
            num_of_samples = conf.NUM_OF_SAMPLES
        else:
            print("Number of samples not specified, using default value of 1")
            num_of_samples = 1

        # Start loop
        while True:
            ret, frame = video_capture.read()
            # save image size to camera info
            if frame_counter == 0:
                self.img_size = frame.shape
                self.roi = (0, 0, self.img_size[1] - 1, self.img_size[0] - 1)

            if not ret:
                termcolor.cprint("Could not open video successfully", "red")
                break

            # skip frame
            if frame_counter % frame_skip == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # find corner in checkerboard
                found_corners, corners = cv2.findChessboardCorners(
                    gray_frame, self.chessboard_size, None
                )

                if found_corners:
                    # Determine in which quadrant is the checkerboard
                    w_half = int(frame.shape[1] / 2)
                    h_half = int(frame.shape[0] / 2)
                    quadrant = "TL"
                    if corners[0][0][0] < w_half:
                        if corners[0][0][1] < h_half:
                            quadrant = "TL"
                        else:
                            quadrant = "BL"
                    else:
                        if corners[0][0][1] < h_half:
                            quadrant = "TR"
                        else:
                            quadrant = "BR"
                    if quadrant_frame_counter[quadrant] < num_of_samples:
                        # frame_skip= frame_skip*0.5
                        frame_filename = os.path.join(
                            sample_dir,
                            f"{quadrant}_frame_{quadrant_frame_counter[quadrant]}.jpg",
                        )
                        cv2.imwrite(frame_filename, frame)
                        quadrant_frame_counter[quadrant] += 1
                        self.obj_points.append(objp)
                        self.img_points.append(corners)
                        print(
                            f"CAM{self.camera_number}> Quadrant {quadrant}: saved frame {frame_counter}"
                        )
                    else:
                        # frame_skip= 50
                        print(
                            f"CAM{self.camera_number}> Quadrant {quadrant}: already saved {num_of_samples} samples"
                        )
                # else:
                # print(f"No corners found in frame {frame_counter}")
            frame_counter += 1
            # Check if enough samples are saved
            if all(
                count >= num_of_samples for count in quadrant_frame_counter.values()
            ):
                print(
                    f"CAM{self.camera_number}> Done, saved enough samples for all quadrants"
                )
                break
        video_capture.release()
        cv2.destroyAllWindows()

        # DONE
        termcolor.cprint(f"Camera {self.camera_number} Done", "green", attrs=["bold"])

    def GetCornersFromSamples(self, sample_folder):
        """
        Reads existing images from a folder of existing samples to use for camera calibration.

        This method reads sample images and detects chessboard corners
        in these frames. It then saves the frames and the detected points
        to be used for camera calibration.

        Parameters
        ----------
        sample_folder : str
            The directory where the sampled frames are retrieved.

        Raises
        ------
        FileNotFoundError
            If the specified sample folder does not exist.

        Notes
        -----
        - The images are expected to be named in a specific format:
            "out{camera_number}F{format}",
            where 'camera_number' is the identifier
            of the camera and 'format' is defined in the configuration
            (example `out3F.jpg` for camera number 3).
        - The method uses a chessboard pattern to detect corners in the frames.
        - It divides the frame into quadrants and saves a specified number of
            sample frames from each quadrant:
            - TL: Top-Left
            - TR: Top-Right
            - BL: Bottom-Left
            - BR: Bottom-Right
        - The method stops when the required number of samples from all
            quadrants have been collected. The required number of steps can be
            set in the setting script NUM_OF_SAMPLES.
        """
        if not os.path.exists(sample_folder):
            termcolor.cprint(f"Folder {sample_folder} does not exist", "red")
            sys.exit("Exiting... cannot open folder.")

        VIDEO_NAME = "/out" + str(self.camera_number) + "F"
        # prepare objects points
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)

        if not self.FISHEYE:
            objp *= conf.SQUARE_SIZE

        path = sample_folder + VIDEO_NAME

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if not os.path.exists(path):
            termcolor.cprint(f"Cannot open {path}, the folder does not exist", "red")
            sys.exit("Exiting... cannot open folder.")

        for frame in os.listdir(path):
            img = cv2.imread(os.path.join(path, frame))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.img_size = img.shape
            self.roi = (0, 0, self.img_size[1] - 1, self.img_size[0] - 1)
            found_corners, corners = cv2.findChessboardCorners(
                gray_img, self.chessboard_size, None
            )
            if found_corners:
                self.obj_points.append(objp)
                print(f"Found corners in sample frame: {frame}")
                corners2 = cv2.cornerSubPix(
                    gray_img, corners, (11, 11), (-1, -1), criteria
                )
                self.img_points.append(corners2)

        termcolor.cprint(f"Camera {self.camera_number} DONE", "green")

    def CalibrateCamera(self):
        """
        Calibrate the camera and set the status to calibrated.

        This method performs the camera calibration process using the collected
        object points and image points. It computes the camera matrix and
        distortion coefficients. The calibration process is different for
        fisheye lenses and standard lenses.

        Raises
        ------
        RuntimeError
            If there is an error during the calibration process.

        Notes
        -----
        - For fisheye lenses, it uses `cv2.fisheye.calibrate` and sets the fisheye
        calibration flags.
        - For standard lenses, it uses `cv2.calibrateCamera`.
        - Sets `self.INT_CAL` to True if the calibration is successful.
        - Prints the calibration status.

        """
        if self.img_size is []:
            img = cv2.imread(
                SampleFile(conf.SAMPLE_FOLDER + f"/out{self.camera_number}F")
            )
            self.img_size = img.shape

        termcolor.cprint(
            f"Calibrating camera {self.camera_number}", "yellow", attrs=["blink"]
        )
        if self.FISHEYE:
            N_OK = len(self.obj_points)
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]

            calibration_flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                # + cv2.fisheye.CALIB_CHECK_COND
                + cv2.fisheye.CALIB_FIX_SKEW
            )
            img_size = self.img_size[:2]

            obj = np.expand_dims(np.asarray(self.obj_points), -2)
            # img = np.expand_dims(np.asarray(self.obj_points), -2)

            rms, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.fisheye.calibrate(
                obj,
                self.img_points,
                img_size,
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            )
            if rms:
                self.INT_CAL = True
                print("Successful calibration")
            else:
                termcolor.cprint(f"Error occurred during calibration", "red")

        else:

            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points, self.img_size[:2], None, None
            )
            if ret:
                self.INT_CAL = True
                print("Successful calibration")
            else:
                termcolor.cprint(f"Error occurred during calibration", "red")

    def RepError(self):
        """
        Calculate the re-projection error given the parameters found in calibration.

        Notes
        -----
        The new refined camera matrix is used.

        If the camera is not calibrated (`INT_CAL` is False),
        the method will print a message indicating that the re-projection error
        cannot be calculated and return without performing any calculations.

        Returns
        -------
        None

        """

        if self.INT_CAL:
            error = 0

            if self.FISHEYE:

                for i in range(len(self.obj_points)):
                    # Ensure object points are numpy array of type float64 and correctly shaped
                    obj = np.array(self.obj_points[i], dtype=np.float64).reshape(
                        -1, 1, 3
                    )

                    # Convert rvec and tvec to numpy arrays of type float64
                    rvec = np.array(self.rvecs[i], dtype=np.float64)
                    tvec = np.array(self.tvecs[i], dtype=np.float64)

                    # project points
                    img_pnt_rep, _ = cv2.fisheye.projectPoints(
                        objectPoints=obj,
                        rvec=rvec,
                        tvec=tvec,
                        K=self.mtx,
                        D=self.dist,
                    )
                    # Reshape img_pnt_rep to match the shape of self.img_points[i]
                    img_pnt_rep = img_pnt_rep.reshape(-1, 2).astype(np.float64)
                    img_points_reshaped = np.array(
                        self.img_points[i], dtype=np.float64
                    ).reshape(-1, 2)

                    # Calculate error
                    error += cv2.norm(
                        img_points_reshaped, img_pnt_rep, cv2.NORM_L2
                    ) / len(img_pnt_rep)

                self.rep_err = error / len(self.obj_points)
                print(
                    f"Re-projection error of CAM{self.camera_number}: {self.rep_err:05.3f}"
                )

            else:

                for i in range(len(self.obj_points)):
                    img_pnt_rep, _ = cv2.projectPoints(
                        objectPoints=self.obj_points[i],
                        rvec=self.rvecs[i],
                        tvec=self.tvecs[i],
                        cameraMatrix=self.mtx,
                        distCoeffs=self.dist,
                    )
                    # Calculate error
                    error += cv2.norm(
                        self.img_points[i], img_pnt_rep, cv2.NORM_L2
                    ) / len(img_pnt_rep)

                self.rep_err = error / len(self.obj_points)
                print(
                    f"Re-projection error of CAM{self.camera_number}: {self.rep_err:05.3f}"
                )

            return
        else:
            print(
                f"Cannot calculate re-projection error. CAM{self.camera_number} is not calibrated"
            )
            return

    def SaveParameters(self, save_dir):
        """
        Save camera parameters to a .pkl file.

        This method serializes the Camera object and saves it to a file in the
        specified directory.

        Parameters
        ----------
        save_dir : str
            The directory where the parameters will be saved.

        Notes
        -----
        - If the folder does not exist, it is created.
        - The saved file is named "Camera_{camera_number}.pkl" where
        `{camera_number}` is the identifier of the camera.
        
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(save_dir + f"/Camera_{self.camera_number}.pkl", "wb") as file:
            pickle.dump(self, file)

        print("Camera parameters saved to file")

    @classmethod
    def LoadCamera(cls, param_dir, camera):
        """
        Load camera parameters from a .pkl file.

        This method deserializes the Camera object from a file in the specified
        directory.

        Parameters
        ----------
        param_dir : str
            The directory where the parameters are stored.
        camera : str
            The identifier for the camera to be loaded.

        Returns
        -------
        Camera
            The Camera object with the loaded parameters.

        Raises
        ------
        FileNotFoundError
            If the specified parameter directory does not exist.

        Notes
        -----
        - The method expects the file to be named "Camera_{camera_number}.pkl"
          where `{camera_number}` is the identifier of the camera.
        """
        if not os.path.exists(param_dir):
            termcolor.cprint(
                f"Cannot open {param_dir}, the folder does not exist", "red"
            )
            sys.exit("Existing... cannot find parameter folder")

        cam_number = conf.CAMS[camera]["number"]

        with open(os.path.join(param_dir, f"Camera_{cam_number}.pkl"), "rb") as file:
            camera = pickle.load(file)
            print(
                f"Camera parameters read from file and set to camera {camera.camera_number}"
            )
            return camera

    def NewOptimalCameraMatrix(self):
        """
        Refine the camera matrix for the undistortion process.

        This method computes a new optimal camera matrix based on the current
        camera matrix and distortion coefficients. It adjusts the camera matrix
        to improve the undistortion of images.

        Depending on whether a fisheye lens is used or a standard lens, the
        method applies different algorithms to compute the new camera matrix:

        - For fisheye lenses, it uses
          :func:`cv2.fisheye.estimateNewCameraMatrixForUndistortRectify`.
        - For standard lenses, it uses :func:`cv2.getOptimalNewCameraMatrix`.

        The computed camera matrix is stored in `self.new_mtx`. In the case of
        standard lenses, the method also updates `self.roi` with the region of
        interest.

        Notes
        -----
        - Ensure that the `self.img_size`, `self.mtx`, and `self.dist` are
          correctly set before calling this method.
        - The `self.FISHEYE` flag determines whether to use the fisheye or
          standard lens processing approach.
        """
        h, w = self.img_size[:2]
        if self.FISHEYE:
            new_camera_mtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.mtx, self.dist, (w, h), np.eye(3), new_size=(w, h)
            )
            self.new_mtx = new_camera_mtx
            pass
        else:
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), 0, (w, h)
            )
            self.new_mtx = new_camera_mtx
            self.roi = roi

    def TestUndistorsion(self):
        """
        Test the undistorsion of the camera images.

        This method reads a sample image, applies the undistorsion process based on
        the calibration parameters, and visualizes the results. It also saves the
        undistorted image to a specified directory.

        Notes
        -----
        - The sample image is read from a predefined folder.
        - The method checks if the camera is calibrated before proceeding.
        - If the camera uses a fisheye lens, `cv2.fisheye.undistortImage` is used for undistorsion.
        - For standard lenses, `cv2.undistort` is used.
        - If cropping is enabled, the undistorted image is cropped to the region of interest (ROI).
        - The original and undistorted images are displayed side by side for comparison.

        Raises
        ------
        RuntimeError
            If the camera is not calibrated.
        FileNotFoundError
            If the sample image is not found.

        Saves
        -----
        - The undistorted image is saved to a predefined directory with the filename "Cam{camera_number}.jpg".

        """

        print(f"Testing undistorsion of camera {self.camera_number}")
        folder = conf.SAMPLE_FOLDER + f"/out{self.camera_number}F"
        img = cv2.imread(SampleFile(folder))

        if self.INT_CAL is not True:
            termcolor.cprint(
                f"Camera {self.camera_number} is not calibrated, calibrate it first",
                "yellow",
            )
            sys.exit("Exiting... camera not calibrated")

        if self.FISHEYE:
            undst = cv2.fisheye.undistortImage(
                img, self.mtx, self.dist, Knew=self.new_mtx
            )

        else:

            undst = cv2.undistort(img, self.mtx, self.dist, None, self.new_mtx)

            if conf.CROP:
                if self.roi != ((0, 0, 0, 0) or None):
                    # crop image
                    x, y, w, h = self.roi
                    undst = undst[y : y + h, x : x + w]
                else:
                    print(f"Cannot crop the image ROI -> {self.roi}")

        out_file = conf.UNDISTORTED_SAMPLES + f"/Cam{self.camera_number}.jpg"
        cv2.imwrite(out_file, undst)

        # Draw the corners
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(g_img, self.chessboard_size, None)
        cv2.drawChessboardCorners(img, self.chessboard_size, corners, ret)

        # Visualize un-distortion
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Camera view #{self.camera_number}. Wide-Lense : {self.FISHEYE}")
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image")
        ax2.imshow(cv2.cvtColor(undst, cv2.COLOR_BGR2RGB))
        ax2.set_title("Undistorted Image")
        ax1.set_axis_off()
        ax2.set_axis_off()
        fig.tight_layout()
        fig.savefig(conf.UNDISTORTED_SAMPLES + f"/Confront/Cam{self.camera_number}.png")
        if conf.SHOW_TEST_FRAME:
            fig.show()

    def GetCorrespondences(self, world_points, image_path):
        """
        Obtain correspondences between world points and image points through a manual user interface.

        Parameters:
        -----------
        world_points : list of tuples
            List of world points (x, y, z) that need to be mapped to the image points (They must lie on the court plane, so z=0!).
        image_path : str
            Path to the image file in which correspondences need to be identified.

        Returns:
        --------
        coords : list of lists
            List of image coordinates corresponding to the world points. If acquisition is skipped or terminated, returns an empty list.
            Each coordinate is either a list of two floats [x, y] or [NaN, NaN] if the point is skipped.

        Notes:
        ------
        This function allows the user to manually select corresponding points in the image using a GUI. The user can interact with the GUI to select points, skip points, or terminate the acquisition process.

        If the camera is not calibrated, the function will print a message and return without acquiring any correspondences.

        The user can interact with the GUI as follows:
        - Right-click on the image to select a point.
        - Press the space bar to skip the current point.
        - Press 'n' to skip the calibration for the current camera.
        - Press 't' to terminate the point acquisition process.

        This method uses OpenCV for image processing and Matplotlib for the GUI.

        Internal Processing:
        --------------------
        1. Reads the court plan image and the input image.
        2. Undistort the input image based on the camera calibration parameters.
        3. Optionally crops the image based on the region of interest (ROI).
        4. Transforms world points to court plan coordinates.
        5. Initiates a GUI for the user to select corresponding points.
        6. Handles user interactions via mouse clicks and keyboard events.
        7. Returns the list of acquired coordinates or an empty list if acquisition is skipped/terminated.
        """

        COURT = "./MISC/Court.png"
        courtplan = cv2.imread(COURT)
        SKIP_CAL = False
        TERM_ACQ = False

        point_counter = 0

        # image_path =
        if not self.INT_CAL:
            print("Camera is not calibrated, cannot un-distort the image")

        image = cv2.imread(image_path)
        if self.FISHEYE:
            undst = cv2.fisheye.undistortImage(
                image, self.mtx, self.dist, Knew=self.mtx
            )
        else:
            undst = cv2.undistort(image, self.mtx, self.dist, None, self.new_mtx)

        if conf.CROP:
            if self.roi == (0, 0, 0, 0) or (self.roi == None):
                print(
                    f"Cam {self.camera_number} roi is {self.roi}, not cropping the image"
                )

            else:
                # crop image
                x, y, w, h = self.roi
                undst = undst[y : y + h, x : x + w]

        n = len(world_points)
        # Get translation vector form RW to Img transformation
        tx, ty = courtplan.shape[:2][::-1]
        t_vector = np.array([tx / 2 - 1, ty / 2 - 1, 0])
        new_points = RW2courtIMG(world_points, 100, t_vector)
        (x_p, y_p, z_p) = zip(*new_points)

        def onClick(event):

            nonlocal point_counter

            if event.button == 3:
                coords.append([event.xdata, event.ydata])
                if (event.xdata and event.ydata) != None:
                    point_counter += 1
                plt.close(fig)
                print(f"Coordinates: {event.xdata}, {event.ydata}")

        def onKey(event):
            nonlocal SKIP_CAL, TERM_ACQ, POP

            if event.key == " ":
                coords.append([np.nan, np.nan])
                print("Skipped point (space-bar keystroke detected)")
                plt.close(fig)

            elif event.key == "r":
                if not coords:
                    print("coords list is empty, cannot remove point")

                    plt.close(fig)
                else:
                    coords.pop(-1)
                    print('Removed last point ("r" keystroke detected)')
                    POP = True
                    plt.close(fig)

            elif event.key == "t":
                print('Point acquisition terminated by user  ("t" keystroke detected)')
                TERM_ACQ = True
                plt.close(fig)

            elif event.key == "n":
                print('Skipped calibration of the camera ("n" keystroke detected)')
                SKIP_CAL = True
                plt.close(fig)

        coords = []

        termcolor.cprint(
            'Press "n" to skip entirely the calibration of '
            f"this camera, (e.g. if you cannot acquire >= {conf.MIN_POINTS} corresponding point, occlusions, ...",
            "green",
            attrs=["bold"],
        )
        pnt = 0

        while pnt < n:

            POP = False
            if not SKIP_CAL:
                if not TERM_ACQ:

                    fig, (ax1, ax2) = plt.subplots(
                        nrows=1, ncols=2, figsize=(12, 5), facecolor="#2e2e2e"
                    )
                    cid = fig.canvas.mpl_connect("button_press_event", onClick)
                    cid_key = fig.canvas.mpl_connect("key_press_event", onKey)
                    ax1.imshow(cv2.cvtColor(courtplan, cv2.COLOR_BGR2RGB))
                    ax1.scatter(
                        x_p[pnt],
                        y_p[pnt],
                        color="yellow",
                        marker="+",
                        label=f"z= {z_p[pnt]}",
                    )
                    ax1.legend(loc=8, fontsize="x-small")

                    ax2.imshow(cv2.cvtColor(undst, cv2.COLOR_BGR2RGB))
                    fig.suptitle(
                        f"Get correspondences for camera #{self.camera_number}",
                        color="white",
                    )
                    text = (
                        "On the right image, find on the point corresponding to the one highlighted in the court plan,"
                        " you can use the GUI tools, e.g. the zoom, to help yourself. Right click the point"
                        " to add the coordinates or press the space-bar to skip point acquisition if it is not visible.\n"
                        f'Press "n" to skip camera. Press "t" to terminate point acquisition and go to next camera (>={conf.MIN_POINTS} points are needed!)\n'
                        f"Acquired Points: {point_counter}"
                    )
                    box = dict(
                        boxstyle="round,pad=1",
                        edgecolor="royalblue",
                        facecolor="lightsteelblue",
                    )
                    txt = fig.text(
                        0.5,
                        0.78,
                        text,
                        family="sans-serif",
                        style="normal",
                        ha="center",
                        wrap=True,
                        linespacing=1.2,
                        bbox=box,
                        in_layout=True,
                    )
                    fig_w = fig.get_figwidth() * fig.dpi
                    txt._get_wrap_line_width = lambda: 0.8 * fig_w
                    ax1.set_title("Point on the court plan", color="white")
                    ax2.set_title(
                        f"Undistorted camera view {self.camera_number}", color="white"
                    )
                    ax1.set_axis_off()
                    ax2.set_axis_off()
                    fig.tight_layout(pad=1.5, rect=(0, 0.05, 1, 0.8))
                    # fig.legend()
                    plt.show()
                # repeat point acquisition if last acquisition is removed
                elif TERM_ACQ:
                    break

                if POP:
                    pnt -= 1
                else:
                    if not coords:
                        pnt = 0
                    else:
                        pnt += 1

            elif SKIP_CAL:
                return []

        # Replace None with np.nan
        for i in range(len(coords)):
            coords[i] = [np.nan if x is None else x for x in coords[i]]

        return coords

    def AddManualCorrespondences(self, world_points, image_points):
        """
        Manually add corresponding points to camera attributes after performing type and NaN checks.

        :param world_points: Point coordinates (x, y, z) expressed in a numpy array with shape (n, 3),
                             where `n` is the number of points. Example: `[[x, y, z], ...]`
        :type world_points: numpy.ndarray

        :param image_points: Point coordinates (u, v) expressed in a numpy array with shape (n, 2),
                             where `n` is the number of points. Example: `[[u, v], ...]`
        :type image_points: numpy.ndarray

        :returns: None
        :rtype: None

        :notes:
            - This method verifies that both input arrays are of type `numpy.ndarray`.
            - The input arrays are checked to have the correct shapes: `(n, 3)` for `world_points` and `(n, 2)` for `image_points`.
            - NaN values within the input arrays are filtered out before the points are stored.
            - If any of these checks fail, appropriate error messages will be printed.
        """

        # Check if inputs are numpy arrays
        if not isinstance(world_points, np.ndarray) or not isinstance(
            image_points, np.ndarray
        ):
            raise TypeError("Both world_points and image_points must be numpy arrays.")

        if world_points.shape[1] != 3:
            raise ValueError("Invalid input: world_points must have shape (n, 3).")
        elif image_points.shape[1] != 2:
            raise ValueError("Invalid input: image_points must have shape (n, 2).")
        elif image_points.shape[0] != world_points.shape[0]:
            raise ValueError(
                f"arrays should contains the same number of points, world points: {image_points.shape[0]}, image points: {world_points.shape[0]}"
            )
        # Filter outs eventual NaN
        valid_indices = ~np.isnan(np.hstack((world_points, image_points))).any(axis=1)

        self.corr_world_points = world_points[valid_indices]
        self.corr_image_points = image_points[valid_indices]

    def ExtCalibration(self):
        """
        Perform external camera calibration to compute the transformation matrices between the world and camera coordinates.

        This method calculates the rotation and translation vectors using the `cv2.solvePnP` function, which solves the Perspective-n-Point problem to find the position and orientation of the camera relative to the world coordinate system. It then computes the world-to-camera and camera-to-world transformation matrices and stores them as instance attributes. Additionally, it computes the homography matrix for the plane to plane mapping of the court plane to the image plane.

        Returns:
        --------
        None

        Raises:
        -------
        SystemExit
            If the PnP solving process fails, an error message is printed, and the program exits.

        Notes:
        ------
        - The method sets the following instance attributes:
            - `self.W2C_mtx` : numpy.ndarray
                4x4 transformation matrix from world coordinates to camera coordinates.
            - `self.C2W_mtx` : numpy.ndarray
                4x4 transformation matrix from camera coordinates to world coordinates.
            - `self.EXT_CAL` : bool
                Flag indicating that the external calibration has been successfully completed.
            - `self.H_mtx` : numpy.ndarray
                Homography matrix for the court plane to the image plane.

        - This method relies on the instance attributes:
            - `self.corr_world_points` : list of tuples
                List of corresponding points in the world coordinates.
            - `self.corr_image_points` : list of tuples
                List of corresponding points in the image coordinates.
            - `self.new_mtx` : numpy.ndarray
                Camera matrix after undistorsion.
            - `self.dist` : numpy.ndarray
                Distortion coefficients.

        """

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.corr_world_points, self.corr_image_points, self.new_mtx, self.dist
        )

        if not success:
            termcolor.cprint("Error, could not solve PnP", "red")
            sys.exit("Exiting...")
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Inverting the rotation and translation to get camera2 world transformation (used to estimate camera position)

        w2c_mtx = np.hstack((rotation_matrix, translation_vector))
        w2c_mtx = np.vstack((w2c_mtx, [0, 0, 0, 1]))
        self.W2C_mtx = w2c_mtx

        # Get cam to world transformation
        inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
        inverse_translation_vector = -np.dot(
            inverse_rotation_matrix, translation_vector
        )

        c2w_mtx = np.hstack((inverse_rotation_matrix, inverse_translation_vector))
        c2w_mtx = np.vstack((c2w_mtx, [0, 0, 0, 1]))
        self.C2W_mtx = c2w_mtx

        self.EXT_CAL = True

        # Computing Homography matrix for the court plane and the image plane
        self.FindHomography()
        if self.H_mtx is None:
            print("Homography matrix calculation failed!")

        return

    def PosError(self):
        """
        Calculate and display the positional error between the approximate and estimated positions.

        This method computes the Euclidean distance between the approximate_position` and the position derived from the `C2W_mtx` matrix. The result is printed and stored in the instance variable `self.pos_err`.

        Notes
        -----
        - `self.approximate_position` should be a list or array-like structure containing the
          approximate position coordinates (x, y, z).
        - `self.C2W_mtx` should be a 4x4 matrix where the last column represents the estimated
          position in 3D space.
        - The Euclidean distance is calculated as the norm of the difference between the
          approximate and estimated positions.
        - The result is printed in meters with three decimal places.
        """
        actual_pos = np.array(self.approximate_position)
        estimated_pos = np.array(self.C2W_mtx[:3, 3])

        euc_dist = np.linalg.norm(estimated_pos - actual_pos)
        print(f"The euclidean distance error is: {euc_dist:06.3f} m")
        self.pos_err = euc_dist
        return

    @classmethod
    def SaveErrors(cls, camera_list, path):
        """
        Save the positional and re-projection errors of a list of cameras to a CSV file.

        This class method generates a CSV file containing error data for each 
        camera in the provided list. The file is saved at the specified `path` 
        and includes the following columns:

        - `CAM`: Camera identifier or number.
        - `x, y, z`: Approximate position coordinates of the camera.
        - `est. x, est. y, est. z`: Estimated position coordinates from the camera's
          `C2W_mtx` matrix.
        - `distance`: Euclidean distance between the approximate and estimated positions.
        - `re-projection error`: Error value representing the difference between the actual
          and projected positions, resulting from the intrinsic calibration procedure.

        The CSV file is formatted with headers and includes error values formatted to two
        decimal places for positions and distances, and three decimal places for re-projection
        errors.

        Parameters
        ----------
        camera_list : list
            A list of camera objects, where each object should have the following attributes:
            - `approximate_position` (list or array-like): The known approximate position of
              the camera (x, y, z).
            - `C2W_mtx` (numpy.ndarray): A 4x4 matrix containing the estimated position in
              the last column.
            - `camera_number` (int or str): An identifier for the camera.
            - `pos_err` (float): The positional error of the camera.
            - `rep_err` (float): The re-projection error of the camera.

        path : str
            The directory path where the CSV file should be saved.

        Notes
        -----
        - The method assumes that each camera object in `camera_list` has the required attributes.
        - The CSV file will be created with the name "Errors.csv" in the specified directory.
        - Ensure that the `path` provided is a valid directory path.
        
        """

        n = len(camera_list)
        header = "CAM,x,y,z,est. x,est. y,est. z, distance, re-projection error"
        # units = ",m,m,m,m,m,m,m,a.u."
        with open(path + "/Errors.csv", "w") as file:
            file.write(header + "\n")
            # file.write(units + "\n")

            for cam in camera_list:
                act = ",".join(f"{x:.2f}" for x in np.array(cam.approximate_position))
                est = ",".join(f"{x:.2f}" for x in cam.C2W_mtx[:3, 3])

                line = f"{cam.camera_number},{act},{est},{cam.pos_err:.2f},{cam.rep_err:.3f}\n"
                file.write(line)

        print("Errors saved to file.")
        return

    def PlotCamera(self, ax):
        """
        Plot the camera's position and direction on a 3D axis.

        This method visualizes the camera's approximate position and estimated position
        on a 3D plot. It also displays a line indicating the camera's direction vector and
        a dashed line connecting the approximate and estimated positions.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes
            The 3D axis object on which to plot the camera information.

        Notes
        -----
        This method is intended to bella called by the PlotMultipleCameras() class method.

        See Also
        --------
        PlotMultipleCameras() : Class Method to plot multiple cameras on 3D scheme of the court.
        """
        act_pos = self.approximate_position
        ax.scatter(act_pos[0], act_pos[1], act_pos[2], marker="v", color="gray")

        cam_pos = self.C2W_mtx[:3, 3]
        ax.scatter(
            cam_pos[0],
            cam_pos[1],
            cam_pos[2],
            marker="o",
            label=f"CAM{self.camera_number} [Err: {self.pos_err:.2f} m]",
        )
        vec = 3  # direction vector size
        cam_dir = self.C2W_mtx[:3, :3] @ np.array([0, 0, vec]) + cam_pos
        ax.plot(
            [cam_pos[0], cam_dir[0]],
            [cam_pos[1], cam_dir[1]],
            [cam_pos[2], cam_dir[2]],
            color="darkgray",
            # label=f"CAM{self.camera_number} Direction",
        )
        x = [act_pos[0], cam_pos[0]]
        y = [act_pos[1], cam_pos[1]]
        z = [act_pos[2], cam_pos[2]]
        ax.plot(x, y, z, "r--", alpha=0.5)

        return

    def PlotCamera2D(self, ax):
        """
        Plot the camera's position and direction on a 2D axis over a court image.

        This method visualizes the camera's estimated and approximate positions on a 2D plot
        with an overlay of a court image. It also shows the direction of the camera with a line
        and connects the estimated and approximate positions with a dashed line.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes
            The 2D axis object on which to plot the camera information.

        Notes
        -----
        - *This method is intended to bella called by the PlotMultipleCameras2D() class method.*
        - The court image is loaded from the path specified by `conf.COURT_IMG_XL`.
        - The camera's estimated position is plotted as a circular marker, while the
          approximate position is plotted as a gray downward-pointing triangle marker.
        - A dashed red line connects the estimated position to the approximate position to
          indicate positional error.
        - The camera's direction is shown as a line extending from the estimated position
          in the direction of the camera's field of view.
        - An annotation with the camera number is placed near the approximate position,
          with a text path effect for better visibility.


        See Also
        --------
        PlotMultipleCameras2D() : Class Method to plot multiple cameras on 2D scheme of the court.
        """
        cam_pos = self.C2W_mtx[:3, 3]
        approx_pos = self.approximate_position
        court_img = cv2.imread(conf.COURT_IMG_XL)

        tx, ty = court_img.shape[:2][::-1]
        t_vector = np.array([tx / 2 - 1, ty / 2 - 1 - 201, 0])
        position = RW2courtIMG(cam_pos, 50, t_vector)
        app_pos = RW2courtIMG(approx_pos, 50, t_vector)

        ax.scatter(
            position[0],
            position[1],
            marker="o",
            label=f"CAM{self.camera_number}; z:{cam_pos[2]:.2f} (m)",
        )
        ax.scatter(
            app_pos[0], app_pos[1], marker="v", color="gray"
        )  # ,label=f"CAM{self.camera_number} from measurement"

        x = [position[0], app_pos[0]]
        y = [position[1], app_pos[1]]
        ax.plot(x, y, "r--", alpha=0.5)

        text = ax.annotate(
            f"CAM {self.camera_number}",
            # # z = {position[3]}
            # f"\n Error {self.pos_err:.2f} m",
            xy=(app_pos[0], app_pos[1]),
            xytext=(10, 0),
            textcoords="offset points",
            # arrowprops=dict(facecolor="lightgray", arrowstyle="->"),
            fontsize=8,
            color="yellow",
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=1.2, foreground="black"),
                path_effects.Normal(),
            ]
        )

        vec = 3  # direction vector size
        cam_dir = self.C2W_mtx[:3, :3] @ np.array([0, 0, vec]) + cam_pos

        new_cam_dir = RW2courtIMG(cam_dir, 50, t_vector)

        ax.plot(
            [position[0], new_cam_dir[0]],
            [position[1], new_cam_dir[1]],
            color="darkgray",
        )

    @classmethod
    def PlotMultipleCameras(cls, camera_list):
        """
        Plot the positions and orientations of multiple cameras on a 3D plot.

        This class method visualizes all cameras in the provided list on a 3D axis. It
        includes the following features:

        - Plots each camera's approximate and estimated positions.
        - Draws direction vectors for each camera.
        - Optionally overlays a court image or displays predefined court points.

        Parameters
        ----------
        camera_list : list
            A list of camera objects, where each object should have the following methods
            and attributes:
            - `PlotCamera(ax)`: Method to plot a single camera on the 3D axis.
            - `approximate_position` (list or array-like): The known approximate position
              of the camera (x, y, z).
            - `C2W_mtx` (numpy.ndarray): A 4x4 matrix containing the estimated position
              in the last column.
            - `camera_number` (int or str): An identifier for the camera.
            - `pos_err` (float): The positional error of the camera.
            - `rep_err` (float): The re-projection error of the camera.
        
        Returns
        -------
        matplotlib.figure.Figure
        The figure object containing the 2D plot with camera positions.

        Notes
        -----
        - The method creates a 3D plot with camera positions, direction vectors, and
          optional court details.
        - Court image is plotted if `conf.SHOW_COURT` is `True`, otherwise, predefined
          court points are displayed.
        - The plot's axes are labeled and ticked to provide clear context for the camera
          positions.

        Example
        -------
        To plot multiple cameras:
        ::
            fig = CameraUtils.Camera.PlotMultipleCameras(camera_list)
        
        where `Camera` is the class containing this method, and `camera_list` is a list
        of camera objects.

        """

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection="3d")

        for camera in camera_list:
            camera.PlotCamera(ax)

        img = cv2.imread(conf.COURT_IMG_LR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        # ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        xlim = np.array([-16.0, 16.0, 0])
        ylim = np.array([-9.5, 9.5, 0])

        x_img = np.linspace(xlim[0], xlim[1], img.shape[1])
        y_img = np.linspace(ylim[0], ylim[1], img.shape[0])
        x_img, y_img = np.meshgrid(x_img, y_img)
        z_img = np.zeros_like(x_img)

        if conf.SHOW_COURT:
            ax.plot_surface(
                -x_img,
                -y_img,
                z_img,
                rstride=conf.COURT_IMG_QUALITY,
                cstride=conf.COURT_IMG_QUALITY,
                facecolors=img,
                shade=False,
                zorder=1,
            )
        else:
            (
                x_p,
                y_p,
                z_p,
            ) = zip(*conf.COURT_POINTS)

            ax.scatter(x_p, y_p, z_p, color="red", marker="o", zorder=10)

        ax.set_box_aspect((4, 2, 1))
        ax.set_xlim((-30, 30))
        ax.set_ylim((-15, 15))
        ax.set_zlim((0, 15))

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Camera positions with respect to the court")
        ax.tick_params(
            axis="both", which="both", bottom=False, top=False, left=False, right=False
        )
        ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
        ax.set_yticks([-10, 0, 10])
        ax.set_zticks([0, 10])

        ax.tick_params(axis="both", labelsize=8)
        fig.legend(loc="lower left", ncols=2, fontsize="x-small")
        return fig

    @classmethod
    def PlotMultipleCameras2D(cls, camera_list):
        """
        Plot the positions of multiple cameras on a 2D court image.

        This class method creates a 2D plot showing the positions of all cameras in the
        provided list overlaid on a court image. Each camera's position is plotted, and
        their respective direction vectors are displayed. The court image serves as a
        reference for the positions.

        Parameters
        ----------
        camera_list : list
            A list of camera objects, where each object should have the `PlotCamera2D(ax)` method to plot its position on the 2D axis.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the 2D plot with camera positions.

        Notes
        -----
        - The court image is loaded from the path specified by `conf.COURT_IMG_XL`.
        - The x and y coordinates are adjusted to match the dimensions of the image.
        - Each camera is plotted using the `PlotCamera2D` method, which should be defined in the same class as this method.
        - The axis labels are set to "x" and "y" to indicate the coordinate axes.
        - The image axis is turned off for a cleaner display of camera positions.

        Example
        -------
        To plot multiple cameras on a 2D image:
        ::
            fig = CameraClass.PlotMultipleCameras2D(camera_list)
        where `CameraClass` is the class containing this method, and `camera_list` is a list of camera objects.

        See Also
        --------
        PlotCamera2D : Method used to plot individual camera positions on a 2D axis.
        """
        fig = plt.figure(figsize=(7, 3))
        ax = fig.add_subplot()

        img = cv2.imread(conf.COURT_IMG_XL)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        yl, xl, _ = img.shape
        for camera in camera_list:
            camera.PlotCamera2D(ax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_xlim(0, xl - 1)
        ax.set_ylim(yl - 1, 0)

        ax.tick_params(
            axis="both", which="both", bottom=False, top=False, left=False, right=False
        )

        fig.suptitle("Camera positions with reference to the court (2D)")
        ax.axis("off")
        # fig.tight_layout(pad = 3)
        fig.legend(title="CAMs", loc="lower left", ncols=1, fontsize="x-small")
        # plt.show()
        return fig

    def PrintAttributes(self, skip_attributes=["obj_points", "img_points"]):
        """
        Prints all the attributes of the CameraInfo instance, except for those in the skip_attributes list.

        Parameters
        ----------
        skip_attributes : list of str, optional
            A list of attribute names to exclude from printing. Default is:
            ["obj_points","img_points"]

            Input skip_attributes = None to print all attributes
        """
        if skip_attributes is None:
            skip_attributes = []

        for attr, value in self.__dict__.items():

            if attr not in skip_attributes:

                termcolor.cprint(f"{attr}", "blue", attrs=["bold"])
                if isinstance(value, np.ndarray):
                    PrintMtx(value)
                elif isinstance(value, tuple):
                    for val in value:
                        if isinstance(val, np.ndarray):
                            print(
                                f"[{val[0][0]:09.3f}, {val[1][0]:09.3f}, {val[2][0]:09.3f}]"
                            )
                            print(33 * "*")
                        else:
                            print(f"{value}")
                            break
                else:
                    print(f"{value}")
            print(80 * "-")

    def FindHomography(self):
        """
        Compute the homography matrix from world coordinates (court plane) to image coordinates.

        This method calculates the homography matrix using corresponding world and image
        points on a plane. It utilizes the RANSAC algorithm to robustly estimate the homography.

        Notes
        -----
        - The method requires at least 4 corresponding points in both the world and image
          coordinate systems to compute the homography matrix. If fewer than 4 points are
          provided, an error message is printed and the method returns `None`.
        - The computed homography matrix `H` is stored in the `self.H_mtx` attribute.
        - The RANSAC algorithm is used with a reprojection threshold of 5 and a confidence
          level of 0.9 to handle outliers in the point correspondences.

        Example
        -------
        To compute the homography matrix, ensure that `self.corr_world_points` and
        `self.corr_image_points` are properly set with at least 4 corresponding points.
        Then call:
        ::
            self.FindHomography()

        where `self` is an instance of the class containing this method.

        See Also
        --------
        cv2.findHomography : OpenCV function used to compute the homography matrix.
        
        """

        src_pnt = self.corr_world_points
        dst_pnt = self.corr_image_points
        if src_pnt.shape[0] < 4 or dst_pnt.shape[0] < 4:
            print("Error: At least 4 points are required to compute the homography.")
            return None
        (H, mask) = cv2.findHomography(
            srcPoints=src_pnt,
            dstPoints=dst_pnt,
            method=cv2.RANSAC,
            ransacReprojThreshold=5,
            confidence=0.9,
        )
        self.H_mtx = H
        return

    def Court2Image(self, coords):
        """
        Convert court coordinates to image coordinates using the homography matrix.

        This method maps 2D coordinates from the court plane to 2D image coordinates
        using the homography matrix. It is useful for projecting points from the court
        space onto the image plane.

        Parameters
        ----------
        coords : numpy.ndarray
            A 2D numpy array of shape (n, 2) representing coordinates on the court plane in the format [x, y].

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of shape (n, 2) representing the image coordinates in the format [u, v].

        Notes
        -----
        - The method appends a column of ones to the court coordinates to convert them to homogeneous coordinates.
        - The homography matrix `self.H_mtx` must be set before calling this method. If `self.H_mtx` is `None`, the method prints an error message and returns `None`.
        - The coordinates are transformed using the homography matrix and normalized to obtain the (u, v) coordinates on the image plane.

        See Also
        --------
        Image2Court : Method used to convert image coordinates to court coordinates.
        """
        if coords.shape[1] != 2:
            print("Invalid input argument, the point coordinates must be: [x,y,z]")
            sys.exit("Exiting...")

        ones_column = np.ones((coords.shape[0], 1))
        h_coords = coords.copy()
        h_coords = np.hstack((h_coords, ones_column))

        if self.H_mtx is None:
            print(f"Homography matrix is {self.H_mtx}")
            return None

        # Transform coordinates to image plane
        img_pnts = np.dot(h_coords, self.H_mtx.T)

        # Normalize to get (u, v) coordinates
        uv_coords = img_pnts[:, :2] / img_pnts[:, 2:3]

        return uv_coords

    def Image2Court(self, coords):
        """
        Convert image coordinates to court coordinates using the homography matrix.

        This method transforms 2D image coordinates into coordinates on the court plane
        by applying the inverse of the homography matrix. It is useful for mapping points
        from the image space to a corresponding position on the court.

        Parameters
        ----------
        coords : numpy.ndarray
            A 2D numpy array of shape (n, 2) representing image coordinates in the format [u, v].

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of shape (n, 2) representing the coordinates on the court plane in the format [x, y].

        Notes
        -----
        - The method appends a column of ones to the image coordinates to convert them to homogeneous coordinates.
        - The homography matrix `self.H_mtx` must be set before calling this method. If `self.H_mtx` is `None`, the method prints an error message and returns `None`.
        - The inverse of the homography matrix is used to transform the coordinates to the court plane.
        - The resulting coordinates are normalized to obtain the (x, y) coordinates on the court.


        See Also
        --------
        FindHomography : Method used to compute the homography matrix which is required for this conversion.
        Court2Image : Inverse method.
        """
        if coords.shape[1] != 2:
            print(
                "Invalid input argument shape, the image coordinates"
                " must be a np.array([[u,v]])"
            )
            # sys.exit(1)
        if self.H_mtx is None:
            print(f"Homography matrix is {self.H_mtx}")
            return None
        # Append the column of ones to 3D points
        ones_column = np.ones((coords.shape[0], 1))
        coords = np.hstack((coords, ones_column))

        Ainv = np.linalg.inv(self.H_mtx)

        court_pnts = np.dot(coords, Ainv.T)

        XY_coords = court_pnts[:, :2] / court_pnts[:, 2:3]
        return XY_coords


#   ____  _       _    ____    _    __  __ _____ ____      _    ____
#  |  _ \| | ___ | |_ / ___|  / \  |  \/  | ____|  _ \    / \  / ___|
#  | |_) | |/ _ \| __| |     / _ \ | |\/| |  _| | |_) |  / _ \ \___ \
#  |  __/| | (_) | |_| |___ / ___ \| |  | | |___|  _ <  / ___ \ ___) |
#  |_|   |_|\___/ \__|\____/_/   \_\_|  |_|_____|_| \_\/_/   \_\____/


class PlotCameras:
    """
    Handles the plotting of camera views around the court and manages the visualization
    of points on the court surface across all camera views.

    This class is designed to initialize camera views and axes, add new camera views,
    and set up a plot layout with the court plan in the center and camera views around it.

    Attributes
    ----------
    views : dict
        A dictionary mapping camera identifiers (e.g., "CAM1", "CAM2") to instances of
        the `Camera` class. This holds all the camera views.
    axes : dict
        A dictionary mapping camera identifiers and "court" to matplotlib axes objects.
        These axes are used for plotting the camera views and the court plan.
    points : numpy.ndarray
        An array of shape (n, 2) to store the coordinates of points picked on the court.
    court_plan : numpy.ndarray
        An image representing the court plan, used as a background for plotting.
    color_counter : int
        A counter used to assign different colors to highlighted points.

    Methods
    -------
    __init__()
        Initializes the `PlotCameras` object with empty camera views and axes, and loads
        the court plan image.

    AddView(camera)
        Adds a new camera view to the `views` dictionary.

    InitPlot()
        Initializes the plot layout with the court plan in the middle and camera views
        around it. Configures the layout and sets up the plotting environment.

    PlotImages()

        Plots the images for all camera views on their respective axes.

    ShowViews()
        Displays the plotted views and connects mouse and keyboard event handlers.

    _on_click(event)
        
        Handles mouse click events to record points on the court and save them to the points attribute.
    
    _on_key(event):
        
        Handles keyboard events to clear points when 'c' is pressed.    
    
    DrawPoints():
        Draws the most recently added point on the images and the court plan.
    
    ClearPoints():
        Clears all recorded points from the `points` attribute and removes them from the plot.
    """

    def __init__(self):  # , *args):
        """
        Initializes the PlotCameras object with camera views and axes setup.

        This constructor sets up the initial state of the `PlotCameras` instance:
        - Initializes dictionaries for camera views and axes.
        - Loads the court plan image from a predefined configuration path.
        - Initializes an empty array for storing picked points and a color counter.
        """

        self.views = {
            "CAM1": None,
            "CAM2": None,
            "CAM3": None,
            "CAM4": None,
            "CAM5": None,
            "CAM6": None,
            "CAM7": None,
            "CAM8": None,
            "CAM12": None,
            "CAM13": None,
        }

        self.axes = {
            "CAM1": None,  # axc1,
            "CAM2": None,  # axc2,
            "CAM3": None,  # axc3,
            "CAM4": None,  # axc4,
            "CAM5": None,  # axc5,
            "CAM6": None,  # axc6,
            "CAM7": None,  # axc7,
            "CAM8": None,  # axc8,
            "CAM12": None,  # axc12,
            "CAM13": None,  # axc13,
            "court": None,  # axcourt,
        }
        self.points = np.array([]).reshape(0, 2)
        self.court_plan = cv2.imread(conf.COURT_IMG_MR)
        self.color_counter = 0

    def AddView(self, camera):
        """
        Adds a camera view to the `views` dictionary.

        Parameters
        ----------
        camera : Camera
            An instance of the `Camera` class that will be added to the `views` dictionary.

        Notes
        -----
        The `camera` parameter should be an instance of the `Camera` class with a
        `camera_number` attribute that corresponds to the camera identifier (e.g., "CAM1").
        """
        cam_id = f"CAM{camera.camera_number}"
        self.views[cam_id] = camera

    def InitPlot(self):
        """
        Initializes the plot layout with a court plan in the middle and camera views around it.

        This method sets up a matplotlib figure with a grid layout, placing the court plan image in the center and arranging the camera views around it. It also configures the axes for the court plan and each camera view, and sets a figure-wide title with instructions for interacting with the plot.

        Notes
        -----
        The layout consists of a 4x3 grid, with camera views placed in locations reflecting their real distribution around the court, which is displayed in the central subplot.
        """
        row = 4
        col = 3

        self.fig = plt.figure(
            figsize=(16, 9), layout="constrained", facecolor="#2e2e2e"
        )
        gs = self.fig.add_gridspec(ncols=col, nrows=row)

        # Placing camera views according to the real world placement
        # with reference to the court
        self.axes["CAM1"] = self.fig.add_subplot(gs[3, 0])
        self.axes["CAM2"] = self.fig.add_subplot(gs[3, 1])
        self.axes["CAM3"] = self.fig.add_subplot(gs[2, 0])
        self.axes["CAM4"] = self.fig.add_subplot(gs[3, 2])
        self.axes["CAM5"] = self.fig.add_subplot(gs[0, 0])
        self.axes["CAM6"] = self.fig.add_subplot(gs[0, 1])
        self.axes["CAM7"] = self.fig.add_subplot(gs[1, 2])
        self.axes["CAM8"] = self.fig.add_subplot(gs[0, 2])
        self.axes["CAM12"] = self.fig.add_subplot(gs[2, 2])
        self.axes["CAM13"] = self.fig.add_subplot(gs[1, 0])
        # Court axis
        self.axes["court"] = self.fig.add_subplot(gs[1:3, 1])
        # Plot the court plan in the middle of the figure
        self.axes["court"].imshow(cv2.cvtColor(self.court_plan, cv2.COLOR_BGR2RGB))
        self.axes["court"].axis("off")
        # Plot suptitle and text
        #  self.fig.tight_layout()#rect=[0, 0.05, 1, 0.95])
        text = (
            "Right-click a point on the court to highlight it in all camera views,"
            " you can use the GUI tools, e.g. the zoom, to help yourself."
            'Press "c" to clear the highlighted points, "q" to quit.'
        )
        self.fig.suptitle(text, fontsize=10, color="white")

    def PlotImages(self):
        """
        Plots the images for all camera views on their respective axes.

        This method reads images from the specified paths, undistorts them based on calibration parameters, resizes the images to improve plot rendering performance, and then displays them on their respective axes in the matplotlib figure. Each camera view is annotated with its identifier.

        The images are rescaled by a factor defined in `conf.SCALE` to enhance plotting performance. Camera identifiers are added as text annotations on the images.
        """
        for key, cam in self.views.items():

            if key != "court":
                # read image
                image_path = conf.SAMPLE_PATH + f"Cam{cam.camera_number}.jpg"
                image = cv2.imread(image_path)
                # undistort the image
                if cam.FISHEYE:
                    undst = cv2.fisheye.undistortImage(
                        image, cam.mtx, cam.dist, Knew=cam.new_mtx
                    )
                else:
                    undst = cv2.undistort(image, cam.mtx, cam.dist, None, cam.new_mtx)

                # Rescale image with conf.SCALE factor to reduce the of the plot,
                #  so it is (hopefully) faster to render
                h, w = [int(dim * conf.SCALE) for dim in undst.shape[:2]]
                resized = cv2.resize(undst, (w, h), interpolation=cv2.INTER_AREA)

                self.axes[key].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

                # Adding Camera Tag
                box = dict(
                    boxstyle="square,pad=0.2", edgecolor="white", facecolor="steelblue"
                )
                self.axes[key].text(
                    20,
                    40,
                    key,
                    family="sans-serif",
                    style="normal",
                    ha="left",
                    bbox=box,
                    color="white",
                )
                self.axes[key].axis("off")

    def ShowViews(self):
        """
        Displays the plotted views and connects mouse and keyboard event handlers.

        This method activates the interactive GUI for highlighting points across camera views. It sets up event handlers for mouse clicks and keyboard presses, allowing users to interact with the plot, highlight points, and clear them using specific key commands. The GUI is displayed using `plt.show()`.
        """
        cid_button = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        print("Showing the point highlight GUI")
        plt.show()

    def _on_click(self, event):
        """
        Handles mouse click events to record points on the court and save them to the points attribute.

         This method is for internal use only and is called when a mouse click event occurs within the matplotlib figure. It processes right-clicks to determine if a point is being selected on the court or in one of the camera views. The coordinates of the clicked point are converted and stored in the `points` attribute. It also triggers the drawing of the selected points on the plot.

         Parameters
         ----------
         event : matplotlib.backend_bases.MouseEvent
             The mouse click event, which includes coordinates and the axis where the click occurred.
        """
        if event.inaxes in self.axes.values():
            if event.button == 3:
                u = np.array([[event.xdata, event.ydata]])
                clicked_axis = event.inaxes
                clicked_axis_key = None

                for key, ax in self.axes.items():
                    if ax == clicked_axis:
                        clicked_axis_key = key
                        break

                if key != "court":

                    self.points = np.concatenate(
                        (
                            self.points,
                            self.views[clicked_axis_key].Image2Court(u / conf.SCALE),
                        ),
                        axis=0,
                    )

                    self.DrawPoints()
                elif key == "court":

                    uh = np.append(u, [[1]], axis=1)
                    pnt = courtIMG2RW(uh, 25, [-399, 237])
                    self.points = np.concatenate(
                        (self.points, pnt),
                        axis=0,
                    )
                    self.DrawPoints()

    def _on_key(self, event):
        """
        Handles keyboard events to clear points when 'c' is pressed.

        This method is for internal use only and is called when a keyboard event occurs. It listens for the 'c' key press and clears all recorded points from the plot and the `points` attribute. This method is intended for internal use within the GUI interaction logic.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            The keyboard event, which includes the key that was pressed.
        """
        if event.key == "c":
            self.ClearPoints()
            print('Cleared Stored Points ("c" keystroke detected)')

    def DrawPoints(self):
        """
        Draws the most recently added point on the images and the court plan.

        This method plots the last added point on all camera views and the court plan.
        It projects the point from court coordinates to image coordinates for camera views, and to the court plan for visualization. Points outside the view or court plan are excluded. The color of the points is determined by `color_counter`.

        Notes
        -----
        The method updates the plot for each camera view and the court plan, and then increments the `color_counter` for distinguishing multiple points.
        """
        # Check if there are any points to draw
        if len(self.points) == 0:
            return  # No points to draw

        # Get the last added point
        last_point = self.points[-1:]

        # loop through axes
        for key, ax in self.axes.items():
            if key != "court":
                # read the point and project it to the image plane
                Point = self.views[key].Court2Image(last_point)

                if Point is not None:
                    # mask out point that lies outside the image
                    if (
                        Point[0, 0] >= 0
                        and Point[0, 0] <= self.views[key].roi[2]
                        and Point[0, 1] >= 0
                        and Point[0, 1] <= self.views[key].roi[3]
                    ):
                        # scale the point according to the scale factor
                        u, v = Point[0] * conf.SCALE
                        # Plot only the last point
                        ax.scatter(
                            [u],
                            [v],
                            marker=".",
                            color=f"C{self.color_counter}",
                            edgecolor="white",
                            linewidth=1,
                        )
            elif key == "court":
                h, w = self.court_plan.shape[:2]
                RF_origin = np.array([w / 2 - 1, h / 2 - 1])
                # Get translation vector form RW to Img transformation
                point = last_point.copy()
                point = np.hstack((point, [[0]]))  # Add a zero for the z-coordinate
                new_point = RW2courtIMG(point, 25, RF_origin)

                # mask out point that lies outside the court plan
                if (
                    new_point[0, 0] >= 0
                    and new_point[0, 0] <= w
                    and new_point[0, 1] >= 0
                    and new_point[0, 1] <= h
                ):
                    x, y = new_point[0][:2]
                    # Plot only the last point
                    ax.scatter(
                        [x],
                        [y],
                        marker="x",
                        color=f"C{self.color_counter}",
                        edgecolor="white",
                        linewidth=1,
                    )
            else:
                pass
        self.color_counter += 1
        ax.figure.canvas.draw()

    def ClearPoints(self):
        """
        Clears all recorded points from the `points` attribute and removes them from the plot.

        This method resets the `points` attribute to an empty array and iterates through all axes to remove plotted points. The figure is then updated to reflect the cleared points.
        """
        self.points = np.array([]).reshape(0, 2)
        for key, ax in self.axes.items():
            for artist in ax.get_children():
                if isinstance(artist, matplotlib.collections.PathCollection):
                    artist.remove()
            self.fig.canvas.draw()


#    ___  _   _                 _____                 _   _
#   / _ \| |_| |__   ___ _ __  |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
#  | | | | __| '_ \ / _ \ '__| | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#  | |_| | |_| | | |  __/ |    |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
#   \___/ \__|_| |_|\___|_|    |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/

# Function for random file sampling
def SampleFile(folder):
    """
    Randomly selects a file from a specified folder.

    This function scans the provided folder for files and randomly selects one from the list of files found. If no files are found, a warning message is printed.

    :param folder: The path to the folder to scan for files.
    :type folder: str
    :return: The path of the randomly selected file.
    :rtype: str
    :raises FileNotFoundError: If the folder does not contain any files.
    """
    files = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_file():
                files.append(entry.path)

    if not files:
        termcolor.cprint(f"No files found in this folder: {folder}", "yellow")

    # Randomly select one file from the list of files
    sampled_file = random.choice(files)
    return sampled_file


def GetFrame(video_folder, cam_number, n):
    """
    Extracts and saves a specific frame from a video file.

    This function reads the specified frame from a video file associated with a given camera number and saves it as an image file. If the video file cannot be opened or the frame cannot be read, an error message is printed.

    :param video_folder: The folder where the video files are located.
    :type video_folder: str
    :param cam_number: The camera number to identify the video file.
    :type cam_number: int
    :param n: The frame number to extract from the video.
    :type n: int
    :return: None
    :rtype: None
    :raises ValueError: If the video file cannot be opened or the frame cannot be read.
    """
    path = video_folder + f"/out{cam_number}.mp4"

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Could not open the video file: {path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {n}")
        return None
    else:
        cap.release()

    save_to = video_folder + "/Samples"
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    cv2.imwrite(save_to + f"/Cam{cam_number}.jpg", frame)
    print(f"{n}-th frame of camera {cam_number} successfully saved")
    return


def PrintMtx(mtx):
    """
    Print a matrix in human readable format
    """
    for r in mtx:
        print(" ".join(f"{val:09.3f}" for val in r))

    return


def RW2courtIMG(RW_point, scale, RF_Img):
    """
    Transforms a point from the real-world court coordinate system to the image coordinate system
    for a court plan.

    Parameters
    ----------
    RW_point : np.ndarray
        A 3D point in the real-world court reference frame, represented as a numpy array
        in homogeneous coordinates, e.g., np.array([x, y, 1]).
    scale : float
        The scale factor representing the ratio of pixels to meters.
        For example, a scale of 50 means 50 pixels correspond to 1 meter.
    RF_Img : list or tuple
        A list or tuple containing the (u, v) coordinates of the reference frame origin in the image,
        corresponding to the real-world origin (x, y) of the court.

    Returns
    -------
    np.ndarray
        A 2D numpy array with the transformed (u, v) coordinates in the image plane.

    Notes
    -----
    The function uses a transformation matrix to convert the real-world court coordinates
    to the image court coordinates, applying scaling and translation to the point.
    """
    # Go from real court RF world to image court plan
    t_mtx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) * scale
    t_vector = np.array([RF_Img[0], RF_Img[1], 0])

    new_point = np.dot(RW_point, t_mtx.T)
    new_point += t_vector
    return new_point


def courtIMG2RW(img_pnt, scale, RF_Img):
    """
    Transforms a point from the image coordinate system to the real-world court coordinate system
    using a homogenous transformation.

    Parameters
    ----------
    img_pnt : np.ndarray
        A point in the image coordinate system, represented as a numpy array in homogeneous coordinates,
        e.g., np.array([[u, v, 1]]).
    scale : float
        The scale factor representing the ratio of pixels to meters.
        For example, a scale of 50 means 50 pixels correspond to 1 meter.
    RF_Img : list or tuple
        A list or tuple containing the (u, v) coordinates of the image's reference frame origin,
        corresponding to the real-world origin (x, y) of the court.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the transformed (x, y) coordinates in the real-world
        court coordinate system, in meters.

    Notes
    -----
    The function applies an inverse transformation, converting image coordinates into
    real-world court coordinates by applying translation and scaling.
    """
    # using homogenous transformation
    t_mtx = np.array([[1, 0, RF_Img[0]], [0, -1, RF_Img[1]], [0, 0, 1]])  # * scale

    new_point = np.dot(img_pnt, t_mtx.T) / scale

    return new_point[:, :2]
