import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion, as_rotation_matrix

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import *

from pyrep.const import ConfigurationPathAlgorithms as Algos
from grasp_planner import GraspPlanner
from perception import CameraIntrinsics

from object_detector import container_detector
import cv2
import matplotlib.pyplot as plt
import time


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)

    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)

    return pos, quat_wxyz


def noisy_object(pose):
    _pos_scale = [0.005] * 3
    _rot_scale = [0.01] * 3
    pos, quat_wxyz = sample_normal_pose(_pos_scale, _rot_scale)
    gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
    perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz
    pose[:3] += pos
    pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]
    return pose


class GraspController:
    def __init__(self, action_mode, static_positions=True):
        # Initialize environment with Action mode and observations
        # Resize the write camera to fit the GQCNN
        wrist_camera = CameraConfig(image_size=(1032, 772))
        self.env = Environment(action_mode, '', ObservationConfig(wrist_camera=wrist_camera), False, static_positions=static_positions)
        self.env.launch()
        # Load specified task into the environment
        self.task = self.env.get_task(EmptyContainer)

    def reset(self):
        descriptions, obs = self.task.reset()
        return descriptions, obs

    def get_objects(self, add_noise=False):
        objs = self.env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        objs_dict = {}

        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()
            if add_noise:
                pose = noisy_object(pose)
            objs_dict[name] = [obj, pose]

        return objs_dict

    def get_path(self, pose, set_orientation=False):
        # TODO deal with situations when path not found
        if set_orientation:
            path = self.env._robot.arm.get_path(pose[:3], quaternion=pose[3:],
                                                ignore_collisions=True, algorithm=Algos.RRTConnect, trials=1000)
        else:
            path = self.env._robot.arm.get_path(pose[:3], quaternion=np.array([0, 1, 0, 0]),
                                                ignore_collisions=True, algorithm=Algos.RRTConnect, trials=1000)
        return path

    def grasp(self):
        # TODO get feedback to check if grasp is successfull
        done_grab_action = False
        # Repeat unitil successfully grab the object
        while not done_grab_action:
            # gradually close the gripper
            done_grab_action = self.env._robot.gripper.actuate(0, velocity=0.2)  # 0 is close
            self.env._pyrep.step()
            # self.task._task.step()
            # self.env._scene.step()

        grasped_objects = {}
        obj_list = ['Shape', 'Shape1', 'Shape3']
        objs = self.env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        for obj in objs:
            if obj.get_name() in obj_list:
                grasped_objects[obj.get_name()] = self.env._robot.gripper.grasp(obj)
        return grasped_objects
        # return self.env._robot.gripper.get_grasped_objects()

    def release(self):
        done = False
        while not done:
            done = self.env._robot.gripper.actuate(1, velocity=0.2)  # 1 is release
            self.env._pyrep.step()
            # self.task._task.step()
            # self.env._scene.step()
        self.env._robot.gripper.release()

    def execute_path(self, path, open_gripper=True):
        path = path._path_points.reshape(-1, path._num_joints)
        for i in range(len(path)):
            action = list(path[i]) + [int(open_gripper)]
            obs, reward, terminate = self.task.step(action)
        return obs, reward, terminate

        ### The following codes can work as well ###
        # done = False
        # path.set_to_start()
        # while not done:
        #     done = path.step()
        #     a = path.visualize()
        #     self.env._scene.step()
        # return done


if __name__ == "__main__":
    # Get grasp planner using GQCNN
    grasp_planner = GraspPlanner(model="GQCNN-2.0")
    # Get large container empty detector
    large_container_detector = container_detector(model='large_container_detector_model.pth')
    # Get small container empty detector
    small_container_detector = container_detector(model='small_container_detector_model.pth')
    # Set Action Mode, See rlbench/action_modes.py for other action modes
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
    # Create grasp controller with initialized environment and task
    grasp_controller = GraspController(action_mode, static_positions=True)
    # Reset task
    descriptions, obs = grasp_controller.reset()

    # The camera intrinsic in RLBench
    camera_intr = CameraIntrinsics(fx=893.738, fy=893.738, cx=516, cy=386, frame='world', height=772, width=1032)
    # The translation between camera and gripper
    # TODO: Change the whole logic into detecting the object using GQCNN
    object_initial_poses = {}
    while True:
        camera_to_gripper_translation = [0.022, 0, 0.095]
        while True:
            objs = grasp_controller.get_objects(add_noise=True)
            # Go to home position
            home_pose = np.copy(objs['waypoint0'][1])
            home_pose[0] -= 0.022
            path = grasp_controller.get_path(home_pose)
            obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=True)

            # Scale the image and change the type to uint8 to fit the neural network
            rgb = np.array(obs.wrist_rgb * 255, dtype='uint8')
            # Change the image to BGR to fit the neural network
            # p.s. The network is trained on BGR images
            wrist_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Use network with trained model to check if the large container is empty or not
            detector_start = time.time()
            container_is_empty = large_container_detector.check_empty(image=wrist_image)
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB))
            if container_is_empty:
                plt.title('The large container is empty? \n Prediction Result: True. Time used: {0:.2f}sec '
                          '\n Forward Finished, Start Resetting'.format(time.time()-detector_start))
                plt.show()
                break
            else:
                plt.title('The large container is empty? \n Prediction Result: False. Time used: {0:.2f}sec '
                          '\n Continue Grasping'.format(time.time() - detector_start))
                plt.show()

            # Take depth picture and use GQCNN to predict grasping pose
            # p.s. Need to scale the depth by 10 to fit GQCNN
            depth = obs.wrist_depth*10
            # Get the grasping pose relative to the current camera position (home position)
            graspping_pose = np.copy(grasp_planner.plan_grasp(depth, rgb, camera_intr=camera_intr))
            # Convert the relative grasping position to global grasping position
            graspping_pose[:3] += home_pose[:3]
            # Add extra distance between camera and gripper
            graspping_pose[:3] += camera_to_gripper_translation
            # Getting the path of reaching the target position
            path = grasp_controller.get_path(graspping_pose, set_orientation=True)
            # Execute the path
            obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=True)

            # grasp the object and return a list of grasped objects
            grasped_objects = grasp_controller.grasp()
            print('Object graspping status:', grasped_objects)
            for object in grasped_objects:
                if grasped_objects[object]:
                    object_initial_poses[object] = graspping_pose

                    # move to home position
                    pose = np.copy(objs['waypoint0'][1])
                    path = grasp_controller.get_path(pose)
                    obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=False)

                    # move above small container
                    rot = np.dot(as_rotation_matrix(quaternion(0, 0, 1, 0)),
                                 np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
                                           [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
                                           [0, 0, 1]]))
                    quat_wxyz = from_rotation_matrix(rot)

                    quat = np.array([quat_wxyz.x, quat_wxyz.y, quat_wxyz.z, quat_wxyz.w])
                    pose = np.copy(objs['waypoint3'][1])
                    pose[3:] = quat
                    path = grasp_controller.get_path(pose, set_orientation=True)
                    obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=False)

                    pose[2] -= 0.15
                    path = grasp_controller.get_path(pose, set_orientation=True)
                    obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=False)
                    # release the object
                    grasp_controller.release()

                    # move above small container
                    pose = np.copy(objs['waypoint3'][1])
                    pose[3:] = quat
                    path = grasp_controller.get_path(pose, set_orientation=True)
                    obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=True)

                    break

        camera_to_gripper_translation = [-0.013, -0.028, 0.1]
        # TODO reset the task
        while True:
            objs = grasp_controller.get_objects(add_noise=True)
            # move above small container
            home_pose = np.copy(objs['waypoint3'][1])

            home_pose[0] -= 0.01
            home_pose[1] += 0.028
            home_pose[2] -= 0.13

            rot = np.dot(as_rotation_matrix(quaternion(0, 0, 1, 0)),
                         np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
                                   [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
                                   [0, 0, 1]]))
            quat_wxyz = from_rotation_matrix(rot)

            grasping_quaternion = np.array([quat_wxyz.x, quat_wxyz.y, quat_wxyz.z, quat_wxyz.w])
            home_pose[3:] = grasping_quaternion
            path = grasp_controller.get_path(home_pose, set_orientation=True)
            obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=True)

            # Get the rgb image and scale it by 255
            rgb = np.array(obs.wrist_rgb * 255, dtype='uint8')
            # use vision to detect if the small container is empty or not
            detector_start = time.time()
            container_is_empty = small_container_detector.check_empty(image=rgb)
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb)
            if container_is_empty:
                plt.title('The small container is empty? \n Prediction Result: True. Time used: {0:.2f}sec '
                          '\n Resetting Finished'.format(time.time() - detector_start))
                plt.show()
                break
            else:
                plt.title('The small container is empty? \n Prediction Result: False. Time used: {0:.2f}sec '
                          '\n Continue Grasping'.format(time.time() - detector_start))
                plt.show()
            # Take depth picture and use GQCNN to predict grasping pose
            # p.s. Need to scale the depth by 10 to fit GQCNN
            depth = obs.wrist_depth * 10
            # Get the grasping pose relative to the current camera position (home position)
            graspping_pose = np.copy(grasp_planner.plan_grasp(depth, rgb, resetting=True, camera_intr=camera_intr))
            # Convert the relative grasping position to global grasping position
            graspping_pose[:3] += home_pose[:3]
            # Add extra distance between camera and gripper
            graspping_pose[:3] += camera_to_gripper_translation
            graspping_pose[3:] = grasping_quaternion

            path = grasp_controller.get_path(graspping_pose, set_orientation=True)
            obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=True)
            # grasp the object and return a list of grasped objects
            grasped_objects = grasp_controller.grasp()
            print('Object graspping status:', grasped_objects)
            target_pose = None
            for object in grasped_objects:
                if grasped_objects[object]:
                    target_pose = object_initial_poses[object]

                    # move above small container
                    pose = np.copy(objs['waypoint3'][1])
                    path = grasp_controller.get_path(pose)
                    obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=False)

                    # move above large container
                    pose = np.copy(objs['waypoint0'][1])
                    path = grasp_controller.get_path(pose)
                    obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=False)

                    # move to reset position
                    path = grasp_controller.get_path(target_pose)
                    obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=False)

                    # release the object
                    grasp_controller.release()

                    # move above large container
                    pose = np.copy(objs['waypoint0'][1])
                    path = grasp_controller.get_path(pose)
                    obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=True)

                    break
