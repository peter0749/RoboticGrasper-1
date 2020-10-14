# Code base from pybullet examples https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/ kuka_diverse_object_gym_env.py


import random
import os
from gym import spaces
import time
import json
import pybullet as p
import numpy as np
import pybullet_data
import pdb
import distutils.dir_util
import glob
from pkg_resources import parse_version
import gym
from bullet.tm700 import tm700
from bullet.tm700_possensor_Gym import tm700_possensor_gym
import multiprocessing as mp
import point_cloud_utils as pcu
from gpg_sampler_nopcl import GpgGraspSampler
#from mayavi import mlab

with open('./gripper_config.json', 'r') as fp:
    config = json.load(fp)
    config['thickness'] = 0.003

num_grasps = 500 # Still slower than GDN
num_dy = 0 # for faster sampling
range_dtheta = 30 # search -+ 30 degrees
safety_dis_above_table = -0.003 # remove points on table
sample_time_limit = 30.0 # prevent gpg sample forever
num_workers = 15
max_num_samples = 200 # Same as PointnetGPD
minimal_points_send_to_point_net = 25 # need > 20 points to compute normal
max_ik_tries = 5 # too many IK will take the simulation too long to finish...
input_points_num = 1000
ags = GpgGraspSampler(config)


def cal_grasp(points_, cam_pos_):
    points_ = points_.astype(np.float32)
    # In ideal environment we don't to denoise the point cloud
    point_cloud = points_
    surface_normal = pcu.estimate_normals(point_cloud, k=20)
    surface_normal = surface_normal[:, 0:3]
    vector_p2cam = points_ - cam_pos_ #cam_pos_ - points_
    vector_p2cam = vector_p2cam / np.linalg.norm(vector_p2cam, axis=1).reshape(-1, 1)
    #tmp = np.dot(vector_p2cam, surface_normal.T).diagonal()
    tmp = np.einsum('ij,ij->i', vector_p2cam, surface_normal)
    angel = np.arccos(np.clip(tmp, -1.0, 1.0))
    wrong_dir_norm = np.where(angel > np.pi * 0.5)[0]
    tmp = np.ones([len(angel), 3])
    tmp[wrong_dir_norm, :] = -1
    surface_normal = surface_normal * tmp
    points_for_sample = points_
    if len(points_for_sample) == 0:
        return [], points_, surface_normal
    def grasp_task(num_grasps_, ags_, queue_):
        ret = ags_.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps_,
                                 max_num_samples=max_num_samples, num_dy=num_dy, time_limit=sample_time_limit, range_dtheta=range_dtheta, safety_dis_above_table=safety_dis_above_table)
        queue_.put(ret)

    queue = mp.Queue()
    num_grasps_p_worker = int(num_grasps/num_workers)
    workers = [mp.Process(target=grasp_task, args=(num_grasps_p_worker, ags, queue)) for _ in range(num_workers)]
    [i.start() for i in workers]

    grasps_together_ = []
    for i in range(num_workers):
        grasps_together_ = grasps_together_ + queue.get()
    return grasps_together_, points_, surface_normal

def check_collision_square(grasp_bottom_center, approach_normal, binormal,
                           minor_pc, points_, p, way="p_open"):
    '''
    Same behavior as in training stage
    '''
    width = config['gpd_gripper_width'] # Here we shoud use the same parameter in training stage instead of real gripper.
    approach_normal = approach_normal.reshape(1, 3)
    approach_normal = approach_normal / np.linalg.norm(approach_normal)
    binormal = binormal.reshape(1, 3)
    binormal = binormal / np.linalg.norm(binormal)
    minor_pc = minor_pc.reshape(1, 3)
    minor_pc = minor_pc / np.linalg.norm(minor_pc)
    matrix_ = np.concatenate((approach_normal.T, binormal.T, minor_pc.T), axis=1) # column
    grasp_matrix = matrix_.T
    center = grasp_bottom_center.reshape(1, 3) + config['gpd_hand_height'] * approach_normal # Need convertion
    points_c = points_ - center
    tmp = np.dot(grasp_matrix, points_c.T)
    points_g = tmp.T

    x_limit = width / 4 # same in training stage
    z_limit = width / 4
    y_limit = width / 2

    x1 = points_g[:, 0] > -x_limit
    x2 = points_g[:, 0] < x_limit
    y1 = points_g[:, 1] > -y_limit
    y2 = points_g[:, 1] < y_limit
    z1 = points_g[:, 2] > -z_limit
    z2 = points_g[:, 2] < z_limit
    a = np.vstack([x1, x2, y1, y2, z1, z2])
    points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
    if len(points_in_area) == 0:
        has_p = False
    else:
        has_p = True

    return has_p, points_in_area, points_g

def collect_pc(grasp_, pc):
    """
    grasp_bottom_center, normal, major_pc, minor_pc
    """
    grasp_num = len(grasp_)
    grasp_ = np.array(grasp_)
    grasp_ = grasp_.reshape(-1, 5, 3)  # prevent to have grasp that only have number 1
    grasp_bottom_center = grasp_[:, 0]
    approach_normal = grasp_[:, 1] # X
    binormal = grasp_[:, 2] # Y
    minor_pc = grasp_[:, 3] # Z

    in_ind_ = []
    in_ind_points_ = []
    p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    for i_ in range(grasp_num):
        has_p, in_ind_tmp, points_g = check_collision_square(grasp_bottom_center[i_], approach_normal[i_],
                                                             binormal[i_], minor_pc[i_], pc, p)
        in_ind_.append(in_ind_tmp)
        in_ind_points_.append(points_g[in_ind_[i_]])
    return in_ind_, in_ind_points_

class tm700_rgbd_gym(tm700_possensor_gym):
  """Class for tm700 environment with diverse objects.

  """

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               objRoot='',
               actionRepeat=80,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=True,
               maxSteps=11,
               dv=0.06,
               removeHeightHack=False,
               blockRandom=0.30,
               cameraRandom=0,
               width=64,
               height=64,
               numObjects=1,
               isTest=False):
    """Initializes the tm700DiverseObjectEnv.

    Args:
      urdfRoot: The diretory from which to load environment URDF's.
      actionRepeat: The number of simulation steps to apply for each action.
      isEnableSelfCollision: If true, enable self-collision.
      renders: If true, render the bullet GUI.
      isDiscrete: If true, the action space is discrete. If False, the
        action space is continuous.
      maxSteps: The maximum number of actions per episode.
      dv: The velocity along each dimension for each action.
      removeHeightHack: If false, there is a "height hack" where the gripper
        automatically moves down for each action. If true, the environment is
        harder and the policy chooses the height displacement.
      blockRandom: A float between 0 and 1 indicated block randomness. 0 is
        deterministic.
      cameraRandom: A float between 0 and 1 indicating camera placement
        randomness. 0 is deterministic.
      width: The image width.
      height: The observation image height.
      numObjects: The number of objects in the bin.
      isTest: If true, use the test set of objects. If false, use the train
        set of objects.
    """

    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._objRoot = objRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self._dv = dv
    self._p = p
    self._removeHeightHack = removeHeightHack
    self._blockRandom = blockRandom
    self._cameraRandom = cameraRandom
    self._width = width
    self._height = height
    self._numObjects = numObjects
    self._isTest = isTest
    self.observation_space = spaces.Box(low=0,
                                         high=255,
                                         shape=(self._height, self._width, 3),
                                         dtype=np.uint8)
    self._urdfList = self._get_all_urdf()
    self._objNameList = [ x.split('/')[-2] for x in self._urdfList ]

    if self._renders:
      self.cid = p.connect(p.SHARED_MEMORY)
      if (self.cid < 0):
        self.cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33]) # cameraposition of rendering
    else:
      self.cid = p.connect(p.DIRECT)
    self.seed()

    if (self._isDiscrete):
      if self._removeHeightHack:
        self.action_space = spaces.Discrete(9)
      else:
        self.action_space = spaces.Discrete(7)
    else:
      self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
      if self._removeHeightHack:
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # dx, dy, dz, da
    self.viewer = None

  def reset(self):
    """Environment reset called at the beginning of an episode.
    """
    # Set the camera settings.
    look = [0.00, -0.15, 0.60]
    self._cam_pos = look
    distance = 0.1
    pitch = -45
    yaw = -75
    roll = 120
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
    self.fov = 35.
    '''
    look = [0.90, -0.28, 0.43]
    distance = 0.15
    pitch = -45
    yaw = 45 # -45
    roll = 180
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
    self.fov = 40.
    '''
    self.focal_length_x = self._width / np.tan(np.radians(self.fov)/2.0)
    self.focal_length_y = self._height / np.tan(np.radians(self.fov)/2.0)
    aspect = self._width / self._height
    self.d_near = 0.01
    self.d_far = 1.5
    self._proj_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.d_near, self.d_far)

    self._attempted_grasp = False
    self._env_step = 0
    self.terminated = 0

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

    self.table_pose = [0.5000000, 0.00000, -.640000, 0.000000, 0.000000, 0.0, 1.0]
    self.tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), *self.table_pose)

    p.setGravity(0, 0, -10)
    self._tm700 = tm700(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)

    self._envStepCounter = 0
    p.stepSimulation()

    # Choose the objects in the bin.
    ind = np.random.choice(len(self._urdfList), self._numObjects, replace=False)
    self._current_urdfList = [self._urdfList[i] for i in ind]
    self._current_objList  = [self._objNameList[i] for i in ind]
    self._objectUids = self._randomly_place_objects(self._current_urdfList)
    self._observation = self._get_observation()
    return np.array(self._observation)

  def _randomly_place_objects(self, urdfList):
    """Randomly places the objects in the bin.

    Args:
      urdfList: The list of urdf files to place in the bin.

    Returns:
      The list of object unique ID's.
    """

    # Randomize positions of each object urdf.
    objectUids = []
    grid_size = 7
    xgrid = np.linspace( 0.0, 1.0, grid_size) * self._blockRandom + 0.40
    ygrid = np.linspace(-0.5, 0.5, grid_size) * self._blockRandom
    xx, yy = np.meshgrid(xgrid, ygrid)
    random_placement = np.random.choice(grid_size*grid_size, len(urdfList), replace=False)
    inds_x, inds_y = np.unravel_index(random_placement, (grid_size, grid_size))
    for urdf_name, ix, iy in zip(urdfList, inds_x, inds_y):
      xpos = xx[ix, iy]
      ypos = yy[ix, iy]
      orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(-np.pi, np.pi)])
      uid = p.loadURDF(urdf_name, [xpos, ypos, 0.001], [orn[0], orn[1], orn[2], orn[3]])
      objectUids.append(uid)
      # Let each object fall to the tray individual, to prevent object
      # intersection.
      for _ in range(1000):
        p.stepSimulation()
    return objectUids

  def _get_observation(self):
    """Return the observation as an image.


    """

    img_arr = p.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=self._view_matrix,
                               projectionMatrix=self._proj_matrix)
    rgb = img_arr[2]
    depth = img_arr[3]
    min = 0.97
    max=1.0
    depthnormalized = [(i - min)/(max-min) for i in depth]
    segmentation = img_arr[4]
    depth = np.reshape(depthnormalized, (self._height, self._width,1) )
    segmentation = np.reshape(segmentation, (self._height, self._width,1) )

    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    np_img_arr = np_img_arr.astype(np.float64)

    test = np.concatenate([np_img_arr[:, :, 1:3], depth], axis=-1)

    return test


  def step(self, action):
    """Environment step.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    dv = self._dv  # velocity per physics step.
    if self._isDiscrete:
      # Static type assertion for integers.
      action = int(action)
      assert isinstance(action, int)
      if self._removeHeightHack:
        dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
        dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
        da = [0, 0, 0, 0, 0, 0, 0, 0.05, 0.05][action]
      else:
        dx = [0, -dv, dv, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0][action]
        dz = -dv
        da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
    else:
      dx = dv * action[0]
      dy = dv * action[1]
      if self._removeHeightHack:
        dz = dv * action[2]
        da = 0.25 * action[3]
      else:
        dz = -dv
        da = 0.25 * action[2]

    return self._step_continuous([dx, dy, dz, da, 0.15])

  def _step_continuous(self, action):
    """Applies a continuous velocity-control action.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    # Perform commanded action.
    self._env_step += 1
    self._tm700.applyAction(action)
    for _ in range(self._actionRepeat):
      p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      if self._termination():
        break

    # If we are close to the bin, attempt grasp.
    state = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmEndEffectorIndex)
    end_effector_pos = state[0]
    if end_effector_pos[2] <= 0.15:
      finger_angle = 0.15
      for _ in range(1000):
        grasp_action = [0, 0, 0.0005, 0, finger_angle]
        self._tm700.applyAction(grasp_action)
        p.stepSimulation()
        if self._renders:
         time.sleep(self._timeStep)
        finger_angle -= 0.3 / 100.
        if finger_angle < 0:
          finger_angle = 0
      for _ in range(1000):
        grasp_action = [0, 0, 0.001, 0, finger_angle]
        self._tm700.applyAction(grasp_action)
        p.stepSimulation()
        if self._renders:
          time.sleep(self._timeStep)
        finger_angle -= 0.15 / 100.
        if finger_angle < 0:
          finger_angle = 0
      self._attempted_grasp = True
    observation = self._get_observation()
    done = self._termination()
    reward = self._reward()
    debug = {'grasp_success': self._graspSuccess}
    return observation, reward, done, debug

  def check_if_grasp_success(self, radius, uid):
    gripper_pos = np.asarray(p.getLinkState(self._tm700.tm700Uid, self._tm700.tmGripperBottomCenterIndex)[0], dtype=np.float32)
    # simple check: if there is points of the object near the gripper finger tip
    blockPos, blockOrn = p.getBasePositionAndOrientation(uid)
    blockPos = np.asarray(blockPos, dtype=np.float32)
    # test if this object is lifted and gripper is "holding" the object (not kicked out by the gripper).
    if blockPos[2]>0.20 and np.linalg.norm(gripper_pos-blockPos, ord=2) < radius:
        return True
    return False


  def _reward(self):

    self.blockUid = self._objectUids[0]

    #rewards is height of target object
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    closestPoints1 = p.getClosestPoints(self.blockUid, self._tm700.tm700Uid, 10, -1,
                                       self._tm700.tmFingerIndexL)
    closestPoints2 = p.getClosestPoints(self.blockUid, self._tm700.tm700Uid, 10, -1,
                                       self._tm700.tmFingerIndexR) # id of object a, id of object b, max. separation, link index of object a (base is -1), linkindex of object b

    # fingerL = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmFingerIndexL)
    # fingerR = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmFingerIndexR)
    # print('infi', np.mean(list(fingerL[0])))


    reward = -1000
    self._graspSuccess = False

    # print(closestPoints1[0][8])
    closestPoints = closestPoints1[0][8]
    numPt = len(closestPoints1)
    #print(numPt)
    if (numPt > 0):
      #print("reward:")
      # reward = -1./((1.-closestPoints1[0][8] * 100 + 1. -closestPoints2[0][8] * 100 )/2)
      reward = -((closestPoints1[0][8]) + (closestPoints2[0][8]) )*(1/2)*(1/0.17849278457978357)
      # reward = 1/((abs(closestPoints1[0][8])   + abs(closestPoints2[0][8])*10 )**2 / 2)
      # reward = 1/closestPoints1[0][8]+1/closestPoints2[0][8]
    if (blockPos[2] > 0.2):
      reward = reward + 1000
      #print("successfully grasped a block!!!")
      self._graspSuccess = True
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("reward")
      #print(reward)
    # print("reward")
    # print(reward)
    return reward


  def _multipleobjreward(self):
    """Calculates the reward for the episode.

    The reward is 1 if one of the objects is above height .2 at the end of the
    episode.
    """
    reward = 0
    self._graspSuccess = 0
    for uid in self._objectUids:
      pos, _ = p.getBasePositionAndOrientation(uid)
      # If any block is above height, provide reward.
      if pos[2] > 0.2:
        self._graspSuccess += 1
        reward = 1
        break
    return reward

  def _termination(self):
    """Terminates the episode if we have tried to grasp or if we are above
    maxSteps steps.
    """
    return self._attempted_grasp or self._env_step >= self._maxSteps

  def _get_all_urdf(self):
    urdf_pattern = self._objRoot + '/**/*.urdf'
    found_object_directories = glob.glob(urdf_pattern, recursive=True)
    return found_object_directories

  def _get_random_urdf(self, num_objects):
    """Randomly choose an object urdf from the random_urdfs directory.

    Args:
      num_objects:
        Number of graspable objects.

    Returns:
      A list of urdf filenames.
    """
    urdf_pattern = self._objRoot + '/**/*.urdf'
    found_object_directories = glob.glob(urdf_pattern, recursive=True)
    total_num_objects = len(found_object_directories)
    selected_objects = np.random.choice(np.arange(total_num_objects), num_objects, replace=num_objects>total_num_objects)
    selected_objects_filenames = []
    for object_index in selected_objects:
      selected_objects_filenames += [found_object_directories[object_index]]
    return selected_objects_filenames

  def _get_random_object(self, num_objects, test):
    """Randomly choose an object urdf from the random_urdfs directory.

    Args:
      num_objects:
        Number of graspable objects.

    Returns:
      A list of urdf filenames.
    """
    if test:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
    else:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[1-9]/*.urdf')
    found_object_directories = glob.glob(urdf_pattern)
    total_num_objects = len(found_object_directories)
    selected_objects = np.random.choice(np.arange(total_num_objects), num_objects, replace=num_objects>total_num_objects)
    selected_objects_filenames = []
    for object_index in selected_objects:
      selected_objects_filenames += [found_object_directories[object_index]]
    return selected_objects_filenames

  def _get_block(self):
    jenga = ["jenga/jenga.urdf"]
    return jenga

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _reset = reset
    _step = step


def get_name_to_link(model_id):
  link_name_to_index = {p.getBodyInfo(model_id)[0].decode('UTF-8'):-1,}
  for _id in range(p.getNumJoints(model_id)):
    _name = p.getJointInfo(model_id, _id)[12].decode('UTF-8')
    link_name_to_index[_name] = _id
  return link_name_to_index

if __name__ == '__main__':
  import sys
  import torch
  torch.backends.cudnn.benchmark = True
  from nms import initEigen
  from nms import crop_index, generate_gripper_edge
  from PointNetCls import PointNetCls
  from scipy.spatial.transform import Rotation

  os.environ['OMP_NUM_THREADS'] = '8'
  initEigen(0)

  output_path = sys.argv[3]
  assert output_path.endswith(('.txt', '.out', '.log'))
  total_n = int(sys.argv[4])
  cls_k = 2

  gripper_length = config['hand_height']
  deepen_hand = gripper_length + 0.01
  model = PointNetCls(num_points=input_points_num, input_chann=3, k=cls_k, return_features=False)
  model = model.eval()
  model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))

  with open(output_path, 'w') as result_fp:
      #p.connect(p.GUI)
      #p.setAdditionalSearchPath(datapath)
      start_obj_id = 3
      input_points = 2048
      ts = None #1/240.
      test = tm700_rgbd_gym(width=720, height=720, numObjects=1, objRoot=sys.argv[2])

      complete_n = 0
      max_tries = 3
      obj_success_rate = {}
      with torch.no_grad():
          for ite in range(total_n):
              test.reset()
              tm_link_name_to_index = get_name_to_link(test._tm700.tm700Uid)
              table_link_name_to_index = get_name_to_link(test.tableUid)

              obj_link_name_to_index = []
              for uid in test._objectUids:
                  obj_link_name_to_index.append((uid, get_name_to_link(uid)))

              object_set = list(test._objectUids)
              object_name = list(test._objNameList)
              grasp_success_obj = np.zeros(len(object_set), dtype=np.bool)
              grasp_failure_obj = np.zeros(len(object_set), dtype=np.int32)
              total_grasp_tries_count_down = max_tries * len(object_set)
              while total_grasp_tries_count_down>0 and (not grasp_success_obj.all()) and grasp_failure_obj.max()<max_tries:
                  total_grasp_tries_count_down -= 1
                  grasp_order = np.random.permutation(len(grasp_failure_obj)) # Randomly specify an object to grasp
                  for obj_i in grasp_order:
                      id_ = object_set[obj_i]
                      if grasp_success_obj[obj_i]:
                          continue
                      test._tm700.home()
                      # Clear out velocity of objects for consistancy
                      for _uid in test._objectUids:
                          p.resetBaseVelocity(_uid, [0, 0, 0], [0, 0, 0])
                      point_cloud, segmentation = test.getTargetGraspObservation()
                      obj_seg = segmentation.reshape(-1)==id_
                      if not obj_seg.any(): # Not visible now. Pick up another object to make it visible
                          print("Cant see anything. Next object!")
                          continue
                      pc_flatten = point_cloud.reshape(-1,3).astype(np.float32)
                      pc_no_arm = pc_flatten[segmentation.reshape(-1)>0,:] # (N, 3)
                      pc_npy = pc_flatten[obj_seg,:] # (N, 3)
                      start_ts = time.time()
                      real_grasp, points, _ = cal_grasp(pc_npy, test._cam_pos)
                      in_ind, in_ind_points = collect_pc(real_grasp, points)
                      sample_ts = time.time()
                      print("Sample in %.4f seconds"%(sample_ts-start_ts))
                      score_value = []
                      assert len(real_grasp) == len(in_ind_points)
                      for ii in range(len(in_ind_points)):
                          if in_ind_points[ii].shape[0] < minimal_points_send_to_point_net:
                              score_value.append(-np.inf)
                          else:
                              score = -np.inf
                              if len(in_ind_points[ii]) >= input_points_num:
                                  points_modify = in_ind_points[ii][np.random.choice(len(in_ind_points[ii]),input_points_num, replace=False)].astype(np.float32)
                              else:
                                  cropped_points = in_ind_points[ii]
                                  additional_points = cropped_points[np.random.choice(len(cropped_points),input_points_num-len(cropped_points), replace=True)]
                                  points_modify = np.append(cropped_points, additional_points, axis=0).astype(np.float32)
                              try:
                                  out = model(torch.from_numpy(points_modify.T).unsqueeze(0))
                                  if isinstance(out, tuple):
                                      out = out[0]
                                  score = float(out[0,-1].cpu()) # (#batch,)
                              except TypeError: # FIXME: I don't know why some times model will return complex128...
                                  score = -np.inf
                              score_value.append(score)
                      ind = np.argsort(-np.asarray(score_value))
                      score_value = [ score_value[i] for i in ind ]
                      real_grasp = [ real_grasp[i] for i in ind   ]
                      model_ts = time.time()
                      pred_poses = []
                      for score, grasp in zip(score_value, real_grasp):
                          grasp_position_model = grasp[4]
                          approach = grasp[1] # X (normal)
                          binormal = grasp[2] # Y (major)
                          #minor_pc = grasp[3] # Z (minor)
                          approach = approach / max(1e-10, np.linalg.norm(approach))
                          binormal = binormal / max(1e-10, np.linalg.norm(binormal))
                          minor_pc = np.cross(approach, binormal)
                          if minor_pc[2]<0:
                              binormal *= -1.
                              minor_pc *= -1.
                          center = grasp_position_model
                          pose = np.c_[approach, binormal, minor_pc, center].astype(np.float32) # (3, 4)
                          assert pose.shape == (3, 4)
                          pred_poses.append(pose)
                      print('Generated0 %d grasps.'%len(pred_poses))
                      pred_poses = np.asarray(pred_poses, dtype=np.float32)
                      end_ts = time.time()
                      print('Generated1 %d grasps in %.2f seconds.'%(len(pred_poses), end_ts-start_ts))
                      result_fp.write('Generated1 %d grasps in %.2f seconds.\n'%(len(pred_poses), end_ts-start_ts))
                      result_fp.flush()

                      new_pred_poses = []
                      for pose in pred_poses:
                          rotation = pose[:3,:3]
                          trans    = pose[:3, 3]
                          approach = rotation[:3,0]
                          if np.arccos(np.dot(approach.reshape(1,3), np.array([1, 0,  0]).reshape(3,1))) > np.radians(85):
                              continue
                          tmp_pose = np.append(rotation, trans[...,np.newaxis], axis=1)

                          # Sanity test
                          gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width']+config['thickness']*2,
                                                                                 config['hand_height'], tmp_pose,
                                                                                 config['thickness_side'], deepen_hand)
                          gripper_inner1, gripper_inner2 = generate_gripper_edge(config['gripper_width'], config['hand_height'],
                                                                                 tmp_pose, config['thickness_side'], 0.0)[1:]
                          outer_pts = crop_index(pc_no_arm, gripper_outer1, gripper_outer2)
                          inner_pts = crop_index(pc_no_arm[outer_pts], gripper_inner1, gripper_inner2)
                          gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge
                          if not (gripper_l_t[2] > -0.001 and gripper_r_t[2] > -0.001 and \
                                  gripper_l[2]   > -0.001 and gripper_r[2]   > -0.001 and \
                                  len(outer_pts) - len(inner_pts) < 30 and len(outer_pts) > 100):
                              continue

                          trans_backward = trans - approach * deepen_hand
                          tmp_pose = np.append(rotation, trans_backward[...,np.newaxis], axis=1)
                          gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'],
                                                                                                     config['hand_height'],
                                                                                                     tmp_pose,
                                                                                                     config['thickness_side'],
                                                                                                     0.0
                                                                                                     )
                          gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge
                          if gripper_l_t[2] < -0.001 or gripper_r_t[2] < -0.001 or \
                             gripper_l[2]   < -0.001 or gripper_r[2]   < -0.001: # ready pose will collide with table
                              continue

                          new_pose = np.append(rotation, trans_backward[...,np.newaxis], axis=1)
                          new_pred_poses.append(new_pose)
                      pred_poses = new_pred_poses
                      print('Generated2 %d grasps'%len(pred_poses))
                      if len(pred_poses)==0:
                          print("No suitable grasp found.")
                          grasp_failure_obj[obj_i] += 1
                          if grasp_failure_obj[obj_i]>=max_tries:
                              break
                          else:
                              continue

                      tried_top1_grasp = None
                      for best_grasp in pred_poses[:max_ik_tries]:
                          rotation = best_grasp[:3,:3]
                          trans_backward = best_grasp[:,3]
                          approach = best_grasp[:3,0]
                          trans = trans_backward + approach*deepen_hand
                          pose = np.append(rotation, trans[...,np.newaxis], axis=1)
                          pose_backward = np.append(rotation, trans_backward[...,np.newaxis], axis=1)
                          for link_name, link_id in tm_link_name_to_index.items():
                              p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, link_id, -1, 0)
                              for obj_id, obj in obj_link_name_to_index:
                                  if obj_id in test._objectUids:
                                      for obj_name, obj_link in obj.items():
                                        # temporary disable collision detection and move to ready pose
                                        p.setCollisionFilterPair(test._tm700.tm700Uid, obj_id, link_id, obj_link, 0)
                          # Ready to grasp pose
                          test._tm700.home()
                          info = test.step_to_target_pose([pose, -0.0],  ts=ts, max_iteration=3000, min_iteration=1)[-1]
                          info_backward = test.step_to_target_pose([pose_backward, -0.0],  ts=ts, max_iteration=2000, min_iteration=1)[-1]
                          if tried_top1_grasp is None:
                              tried_top1_grasp = (pose_backward, pose)
                          if info['planning'] and info_backward['planning']:
                              break # Feasible Pose found
                          else:
                              print("Inverse Kinematics failed.")
                      if (not (info['planning'] and info_backward['planning'])) and (not tried_top1_grasp is None):
                          pose_backward, pose = tried_top1_grasp
                          test.step_to_target_pose([pose_backward, -0.0],  ts=ts, max_iteration=2000, min_iteration=1)[-1]
                      # Enable collision detection to test if a grasp is successful.
                      for link_name in ['finger_r_link', 'finger_l_link']:
                          link_id = tm_link_name_to_index[link_name]
                          for obj_id, obj in obj_link_name_to_index:
                              if obj_id in test._objectUids:
                                  for obj_name, obj_link in obj.items():
                                    p.setCollisionFilterPair(test._tm700.tm700Uid, obj_id, link_id, obj_link, 1)
                      # Enable collision detection for gripper head, fingers
                      #p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['gripper_link'], -1, 1)
                      p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['finger_r_link'], -1, 1)
                      p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['finger_l_link'], -1, 1)
                      # Deepen gripper hand
                      for d in np.linspace(0, 1, 60):
                          info = test.step_to_target_pose([pose*d+pose_backward*(1.-d), -0.0],  ts=ts, max_iteration=100, min_iteration=1)[-1]
                      info = test.step_to_target_pose([pose, -0.0],  ts=ts, max_iteration=500, min_iteration=1)[-1]
                      if not info['planning']:
                          print("Inverse Kinematics failed.")
                      # Grasp it
                      test.step_to_target_pose([pose, 0.2],  ts=ts, max_iteration=500, min_iteration=5)
                      # Test if we can lift the object
                      pose[2,3] = 0.5
                      test.step_to_target_pose([pose, 0.2],  ts=ts, max_iteration=5000, min_iteration=5)
                      for _ in range(1000):
                          p.stepSimulation()
                      # Compute success rate for each object
                      this_grasp_success = False
                      if test.check_if_grasp_success(0.50, id_):
                          print("Grasp success!")
                          this_grasp_success = True
                      else:
                          print("Grasp failed!")
                          if False:
                              pc_subset = np.copy(pc_no_arm)
                              if len(pc_subset)>5000:
                                  pc_subset = pc_subset[np.random.choice(len(pc_subset), 5000, replace=False)]
                              mlab.clf()
                              mlab.points3d(pc_subset[:,0], pc_subset[:,1], pc_subset[:,2], scale_factor=0.004, mode='sphere', color=(1.0,1.0,0.0), opacity=1.0)
                              for n, pose_ in enumerate(pred_poses):
                                  pose = np.copy(pose_)
                                  pose[:,3] += pose[:,0] * deepen_hand
                                  gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'], 0.0)
                                  gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge

                                  mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=config['thickness']/4., color=(0,0,1) if n>0 else (1,0,0), opacity=0.5)
                                  mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=config['thickness']/4., color=(0,0,1) if n>0 else (1,0,0), opacity=0.5)
                                  mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=config['thickness']/4., color=(0,0,1) if n>0 else (1,0,0), opacity=0.5)
                              mlab.show()
                              input()
                      if this_grasp_success:
                          test._objectUids.remove(id_)
                          p.removeBody(id_)
                          grasp_success_obj[obj_i] = True
                          if grasp_success_obj.all():
                              break
                      else:
                          grasp_failure_obj[obj_i] += 1
                          if grasp_failure_obj[obj_i]>=max_tries:
                              break
                  if grasp_success_obj.all():
                      complete_n += 1
                  for obj_i in range(len(object_set)):
                      name = object_name[obj_i]
                      fail_n = grasp_failure_obj[obj_i]
                      success_n = 1 if grasp_success_obj[obj_i] else 0
                      if not name in obj_success_rate:
                          obj_success_rate[name] = (0, 0) # success , fail
                      success_n_old, fail_n_old = obj_success_rate[name]
                      obj_success_rate[name] = (success_n+success_n_old, fail_n+fail_n_old)
                  for obj_name, (success_n, fail_n) in obj_success_rate.items():
                      total_grasp_obj = success_n + fail_n
                      if total_grasp_obj==0:
                          result_fp.write("%s : Unknown\n"%obj_name)
                      else:
                          result_fp.write("%s : %.4f (%d / %d)\n"%(obj_name, success_n/total_grasp_obj, success_n, total_grasp_obj))
                      result_fp.flush()
              result_fp.write("Complete rate: %.6f (%d / %d)\n"%(complete_n/(ite+1), complete_n, ite+1))
              result_fp.flush()
