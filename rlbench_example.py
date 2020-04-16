import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import *

from pyrep.const import ConfigurationPathAlgorithms as Algos
from grasp_planner import GraspPlanner
from perception import CameraIntrinsics

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
            self.task._task.step()
            self.env._scene.step()
        return self.env._robot.gripper.get_grasped_objects()

    def release(self):
        done = False
        while not done:
            done = self.env._robot.gripper.actuate(1, velocity=0.2)  # 1 is release
            self.env._pyrep.step()
            self.task._task.step()
            self.env._scene.step()
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
    grasp_planner = GraspPlanner(model="GQCNN-4.0-PJ")
    # Set Action Mode, See rlbench/action_modes.py for other action modes
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
    # Create grasp controller with initialized environment and task
    grasp_controller = GraspController(action_mode, static_positions=True)
    # Reset task
    descriptions, obs = grasp_controller.reset()

    camera_intr = CameraIntrinsics(fx=893.738, fy=893.738, cx=516, cy=386, frame='world', height=772, width=1032)
    camera_to_gripper_translation = [0.03, 0, 0.1]
    # TODO: Change the whole logic into detecting the object using GQCNN
    while True:
        objs = grasp_controller.get_objects(add_noise=True)
        # go back to home position
        home_pose = objs['waypoint0'][1]
        path = grasp_controller.get_path(home_pose)
        obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=True)
        # Getting object poses, noisy or not
        # TODO detect the pose using vision and handle the noisy pose

        # Take depth picture and use GQCNN to predict grasping pose
        depth = obs.wrist_depth*10
        # Get the grasping pose relative to the current camera position (home position)
        graspping_pose = grasp_planner.plan_grasp(depth, camera_intr=camera_intr)
        # Convert the relative grasping position to global grasping position
        graspping_pose[:3] += home_pose[:3]
        # Add extra distance between camera and gripper
        graspping_pose[:3] += camera_to_gripper_translation
        print(graspping_pose)
        # Getting the path of reaching the target position
        path = grasp_controller.get_path(graspping_pose, set_orientation=True)
        # Execute the path
        obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=True)

        # grasp the object and return a list of grasped objects
        grasped_objects = grasp_controller.grasp()
        # TODO get feedback to check if grasp is successfull

        # move to home position
        pose = objs['waypoint0'][1]
        path = grasp_controller.get_path(pose)
        obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=False)

        # move above small container
        pose = objs['waypoint3'][1]
        path = grasp_controller.get_path(pose)
        obs, reward, terminate = grasp_controller.execute_path(path, open_gripper=False)

        # release the object
        grasp_controller.release()

        # TODO check if large container is empty and finish the forward task(using vision)

    # TODO reset the task