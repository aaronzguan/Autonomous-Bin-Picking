"""
This module uses the `BerkleyAutomation GQCNN
<https://berkeleyautomation.github.io/gqcnn>`_
grasp detector to compute a valid grasp out of RBG-D data.
"""
import numpy as np
import json
import os
import time
from autolab_core import YamlConfig, Logger
from gqcnn.grasping import (
    RobustGraspingPolicy,
    CrossEntropyRobustGraspingPolicy,
    RgbdImageState,
    FullyConvolutionalGraspingPolicyParallelJaw,
    FullyConvolutionalGraspingPolicySuction,
)
from gqcnn.utils import GripperMode
from perception import (CameraIntrinsics, ColorImage, DepthImage, RgbdImage)
from visualization import Visualizer2D as vis
from quaternion import from_rotation_matrix


class GraspPlanner(object):
    """
    Class used to compute the grasp pose out of the RGB-D data.
    """
    def __init__(self, model="GQCNN-4.0-PJ", config_filepath=None):
        self.logger = Logger.get_logger(__name__)
        self.model = model
        self.grasping_policy = None
        self._get_cfg(config_filepath)

    def _get_cfg(self, config_filepath=None):
        """
        Function retrieves the model and policy configuration files for a given model.
        Parameters
        ----------
        model: type `str`
            Model used for the grasp detection CNN.
        """
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        model_path = os.path.join(model_dir, self.model)
        # Retrieve model related configuration values
        self.model_config = json.load(open(os.path.join(model_path, "config.json"), "r"))
        try:
            gqcnn_config = self.model_config["gqcnn"]
            gripper_mode = gqcnn_config["gripper_mode"]
        except KeyError:
            gqcnn_config = self.model_config["gqcnn_config"]
            input_data_mode = gqcnn_config["input_data_mode"]
            if input_data_mode == "tf_image":
                gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
            elif input_data_mode == "tf_image_suction":
                gripper_mode = GripperMode.LEGACY_SUCTION
            elif input_data_mode == "suction":
                gripper_mode = GripperMode.SUCTION
            elif input_data_mode == "multi_suction":
                gripper_mode = GripperMode.MULTI_SUCTION
            elif input_data_mode == "parallel_jaw":
                gripper_mode = GripperMode.PARALLEL_JAW
            else:
                raise ValueError(
                    "Input data mode {} not supported!".format(input_data_mode))

        # Get config filename corrsponds to the model
        if config_filepath is None:
            if gripper_mode == GripperMode.LEGACY_PARALLEL_JAW or gripper_mode == GripperMode.PARALLEL_JAW:
                if "FC" in self.model:
                    config_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gqcnn/cfg/examples/fc_gqcnn_pj.yaml")
                else:
                    config_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gqcnn/cfg/examples/gqcnn_pj.yaml")
            elif gripper_mode == GripperMode.LEGACY_SUCTION or gripper_mode == GripperMode.SUCTION:
                if "FC" in self.model:
                    config_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gqcnn/cfg/examples/fc_gqcnn_suction.yaml")
                else:
                    config_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gqcnn/cfg/examples/gqcnn_suction.yaml")

        # Read config.
        self.config = YamlConfig(config_filepath)
        self.policy_config = self.config["policy"]
        self.policy_config["metric"]["gqcnn_model"] = model_path

    def plan_grasp(self, depth, rgb, resetting=False, camera_intr=None, segmask=None):
        """
        Computes possible grasps.
        Parameters
        ----------
        depth: type `numpy`
            depth image
        rgb: type `numpy`
            rgb image
        camera_intr: type `perception.CameraIntrinsics`
            Intrinsic camera object.
        segmask: type `perception.BinaryImage`
            Binary segmask of detected object
        Returns
        -------
        type `GQCNNGrasp`
            Computed optimal grasp.
        """
        if "FC" in self.model:
            assert not (segmask is None), "Fully-Convolutional policy expects a segmask."
        if camera_intr is None:
            camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gqcnn/data/calib/primesense/primesense.intr")
            camera_intr = CameraIntrinsics.load(camera_intr_filename)

        depth_im = DepthImage(depth, frame=camera_intr.frame)
        color_im = ColorImage(rgb, frame=camera_intr.frame)

        valid_px_mask = depth_im.invalid_pixel_mask().inverse()
        if segmask is None:
            segmask = valid_px_mask
        else:
            segmask = segmask.mask_binary(valid_px_mask)

        # Inpaint.
        depth_im = depth_im.inpaint(rescale_factor=self.config["inpaint_rescale_factor"])
        # Aggregate color and depth images into a single
        # BerkeleyAutomation/perception `RgbdImage`.
        self.rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        # Create an `RgbdImageState` with the `RgbdImage` and `CameraIntrinsics`.
        state = RgbdImageState(self.rgbd_im, camera_intr, segmask=segmask)

        # Set input sizes for fully-convolutional policy.
        if "FC" in self.model:
            self.policy_config["metric"]["fully_conv_gqcnn_config"]["im_height"] = depth_im.shape[0]
            self.policy_config["metric"]["fully_conv_gqcnn_config"]["im_width"] = depth_im.shape[1]

        return self.execute_policy(state, resetting)

    def _get_grasp_policy(self):
        """
        Get the grasping policy based on the model
        Returns
        ----------
        grasping_policy: type `gqcnn.grasping.policy.policy.GraspingPolicy`
            Grasping policy to use.
        """
        # Get grasping policy
        if "FC" in self.model:
            if self.policy_config["type"] == "fully_conv_suction":
                self.grasping_policy = FullyConvolutionalGraspingPolicySuction(self.policy_config)
            elif self.policy_config["type"] == "fully_conv_pj":
                self.grasping_policy = FullyConvolutionalGraspingPolicyParallelJaw(self.policy_config)
            else:
                raise ValueError("Invalid fully-convolutional policy type: {}".format(self.policy_config["type"]))
        else:
            policy_type = "cem"
            if "type" in self.policy_config:
                policy_type = self.policy_config["type"]
            if policy_type == "ranking":
                self.grasping_policy = RobustGraspingPolicy(self.policy_config)
            elif policy_type == "cem":
                self.grasping_policy = CrossEntropyRobustGraspingPolicy(self.policy_config)
            else:
                raise ValueError("Invalid policy type: {}".format(policy_type))

    def execute_policy(self, rgbd_image_state, resetting=False):
        """
        Executes a grasping policy on an `RgbdImageState`.
        Parameters
        ----------
        rgbd_image_state: type `gqcnn.RgbdImageState`
            The :py:class:`gqcnn.RgbdImageState` that encapsulates the
            depth and color image along with camera intrinsics.
        """
        policy_start = time.time()
        if not self.grasping_policy:
            self._get_grasp_policy()
        try:
            grasping_action = self.grasping_policy(rgbd_image_state)
        except:
            vis.figure(size=(10, 10))
            vis.imshow(self.rgbd_im.color,
                       vmin=0,
                       vmax=255)
            vis.title("No Valid Grasp, Task Finished")
            vis.show()

        self.logger.info("Planning took %.3f sec" % (time.time() - policy_start))

        # Angle of grasping point w.r.t the x-axis of camera frame
        angle_wrt_x = grasping_action.grasp.angle
        angle_degree = angle_wrt_x * 180 / np.pi
        if angle_degree <= -270:
            angle_degree += 360
        elif (angle_degree > -270 and angle_degree <= -180) or (angle_degree > -180 and angle_degree <= -90):
            angle_degree += 180
        elif (angle_degree > 90 and angle_degree <= 180) or (angle_degree > 180 and angle_degree <= 270):
            angle_degree -= 180
        elif (angle_degree > 270 and angle_degree <= 360):
            angle_degree -= 360
        angle_wrt_x = angle_degree * np.pi / 180

        if resetting:
            angle_wrt_x += np.pi/2
            # Translation of grasping point w.r.t the camera frame
            grasping_translation = np.array([grasping_action.grasp.pose().translation[0] * -1,
                                             grasping_action.grasp.pose().translation[1],
                                             grasping_action.grasp.pose().translation[2] * -1])
            # Rotation matrix from world frame to camera frame
            world_to_cam_rotation = np.dot(np.array([[1, 0, 0],
                                                     [0, np.cos(np.pi), -np.sin(np.pi)],
                                                     [0, np.sin(np.pi), np.cos(np.pi)]]),
                                           np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                                                     [np.sin(np.pi), np.cos(np.pi), 0],
                                                     [0, 0, 1]]))
            # Rotation matrix from camera frame to gripper frame
            cam_to_gripper_rotation = np.array([[np.cos(angle_wrt_x), -np.sin(angle_wrt_x), 0],
                                                [np.sin(angle_wrt_x), np.cos(angle_wrt_x), 0],
                                                [0, 0, 1]])
        else:
            # Translation of grasping point w.r.t the camera frame
            grasping_translation = np.array([grasping_action.grasp.pose().translation[1],
                                             grasping_action.grasp.pose().translation[0],
                                             grasping_action.grasp.pose().translation[2]]) * -1

            # Rotation matrix from world frame to camera frame
            world_to_cam_rotation = np.dot(np.array([[1, 0, 0],
                                                     [0, np.cos(np.pi), -np.sin(np.pi)],
                                                     [0, np.sin(np.pi), np.cos(np.pi)]]),
                                           np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
                                                     [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
                                                     [0, 0, 1]]))
            # Rotation matrix from camera frame to gripper frame
            cam_to_gripper_rotation = np.dot(np.array([[np.cos(angle_wrt_x), -np.sin(angle_wrt_x), 0],
                                                       [np.sin(angle_wrt_x), np.cos(angle_wrt_x), 0],
                                                       [0, 0, 1]]),
                                             np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
                                                       [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
                                                       [0, 0, 1]]))

        world_to_gripper_rotation = np.dot(world_to_cam_rotation, cam_to_gripper_rotation)
        quat_wxyz = from_rotation_matrix(world_to_gripper_rotation)
        grasping_quaternion = np.array([quat_wxyz.x, quat_wxyz.y, quat_wxyz.z, quat_wxyz.w])

        grasping_pose = np.hstack((grasping_translation, grasping_quaternion))

        vis.figure(size=(10, 10))
        vis.imshow(self.rgbd_im.color,
                   vmin=0,
                   vmax=255)
        vis.grasp(grasping_action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title("Planned grasp at depth {0:.3f}m \n".format(grasping_action.grasp.depth)
                  + 'grasping pose {}'.format(grasping_pose))
        vis.show()
        return grasping_pose