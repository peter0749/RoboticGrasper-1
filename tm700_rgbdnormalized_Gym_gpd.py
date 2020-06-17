# Code base from pybullet examples https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/ kuka_diverse_object_gym_env.py


import random
import os
from gym import spaces
import time
import pybullet as p
import numpy as np
import pybullet_data
import pdb
import distutils.dir_util
import json
import glob
from pkg_resources import parse_version
import gym
from bullet.tm700 import tm700
from bullet.tm700_possensor_Gym import tm700_possensor_gym
#from mayavi import mlab
import pcl
import point_cloud_utils as pcu # Use pcu to compute normal feature (same as training stage)
import multiprocessing as mp
from gpg_sampler import GpgGraspSamplerPcl


with open('./gripper_config.json', 'r') as fp:
    config = json.load(fp)
    # GPDs are easy to collide
    '''
    shrink_width = 0.005
    expand_thick = 0.002
    config['gripper_width'] -= shrink_width
    config['thickness'] += shrink_width*0.5 + expand_thick
    '''

num_grasps = 3000 # Same as GPD and GDN
num_workers = 24
max_num_samples = 150 # Same as PointnetGPD

project_size = 60 # For GPD
projection_margin = 1 # For GPD
voxel_point_num = 50 # For GPD
project_chann = 12 # We only compare GPD with 12 channels
minimal_points_send_to_point_net = 150 # need > 20 points to compute normal
input_points_num = 1000
ags = GpgGraspSamplerPcl(config)

def cal_grasp(points_, cam_pos_):
    points_ = points_.astype(np.float32)
    # In ideal environment we don't to denoise the point cloud
    point_cloud = pcl.PointCloud(points_)
    norm = point_cloud.make_NormalEstimation()
    norm.set_KSearch(20)  # critical parameter when calculating the norms
    normals = norm.compute()
    surface_normal = normals.to_array()
    surface_normal = surface_normal[:, 0:3]
    vector_p2cam = points_ - cam_pos_ #cam_pos_ - points_
    vector_p2cam = vector_p2cam / np.linalg.norm(vector_p2cam, axis=1).reshape(-1, 1)
    tmp = np.dot(vector_p2cam, surface_normal.T).diagonal()
    angel = np.arccos(np.clip(tmp, -1.0, 1.0))
    wrong_dir_norm = np.where(angel > np.pi * 0.5)[0]
    tmp = np.ones([len(angel), 3])
    tmp[wrong_dir_norm, :] = -1
    surface_normal = surface_normal * tmp
    points_for_sample = points_
    if len(points_for_sample) == 0:
        return [], points_, surface_normal
    #grasps_together_ = ags.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps,
    #                                     max_num_samples=max_num_samples)
    def grasp_task(num_grasps_, ags_, queue_):
        ret = ags_.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps_,
                                 max_num_samples=max_num_samples)
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
    width = ags.config['gripper_width']
    approach_normal = approach_normal.reshape(1, 3)
    approach_normal = approach_normal / np.linalg.norm(approach_normal)
    binormal = binormal.reshape(1, 3)
    binormal = binormal / np.linalg.norm(binormal)
    minor_pc = minor_pc.reshape(1, 3)
    minor_pc = minor_pc / np.linalg.norm(minor_pc)
    matrix_ = np.concatenate((approach_normal.T, binormal.T, minor_pc.T), axis=1) # column
    grasp_matrix = matrix_.T
    center = grasp_bottom_center.reshape(1, 3) + config['hand_height'] * approach_normal # Need convertion
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

def cal_projection(point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, gripper_width):
    occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
    norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
    norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

    max_x = point_cloud_voxel[:, order[0]].max()
    min_x = point_cloud_voxel[:, order[0]].min()
    max_y = point_cloud_voxel[:, order[1]].max()
    min_y = point_cloud_voxel[:, order[1]].min()
    min_z = point_cloud_voxel[:, order[2]].min()

    tmp = max((max_x - min_x), (max_y - min_y))
    if tmp == 0:
        print("WARNING : the num of input points seems only have one, no possilbe to do learning on"
              "such data, please throw it away.  -- Hongzhuo")
        return occupy_pic, norm_pic
    # Here, we use the gripper width to cal the res:
    res = gripper_width / (m_width_of_pic-margin)

    voxel_points_square_norm = []
    x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
    y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
    z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
    x_coord_r = np.floor(x_coord_r).astype(int)
    y_coord_r = np.floor(y_coord_r).astype(int)
    z_coord_r = np.floor(z_coord_r).astype(int)
    voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
    coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
    K = len(coordinate_buffer)
    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=K, dtype=np.int64)
    feature_buffer = np.zeros(shape=(K, voxel_point_num, 6), dtype=np.float32)
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

    for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < voxel_point_num:
            feature_buffer[index, number, :3] = point
            feature_buffer[index, number, 3:6] = normal
            number_buffer[index] += 1

    voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1)/number_buffer[:, np.newaxis]
    voxel_points_square = coordinate_buffer

    if len(voxel_points_square) == 0:
        return occupy_pic, norm_pic
    x_coord_square = voxel_points_square[:, 0]
    y_coord_square = voxel_points_square[:, 1]
    norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
    occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
    occupy_max = occupy_pic.max()
    assert(occupy_max > 0)
    occupy_pic = occupy_pic / occupy_max
    return occupy_pic, norm_pic

def project_pc(pc, gripper_width, in_ind):
    """
    for gpd baseline, only support input_chann == [3, 12]
    """
    '''
    pc = pc.astype(np.float32)
    pc = pcl.PointCloud(pc)
    norm = pc.make_NormalEstimation()
    norm.set_KSearch(10) # same in training stage
    normals = norm.compute()
    surface_normal = normals.to_array()[:, 0:3]
    '''

    # Use pcu here. Because we trained GPD with pcu.
    surface_normal = pcu.estimate_normals(pc, k=10)
    surface_normal = surface_normal[:, 0:3]

    grasp_pc = pc[in_ind]
    grasp_pc_norm = surface_normal[in_ind]
    bad_check = (grasp_pc_norm != grasp_pc_norm)
    if np.sum(bad_check)!=0:
        bad_ind = np.where(bad_check == True)
        grasp_pc = np.delete(grasp_pc, bad_ind[0], axis=0)
        grasp_pc_norm = np.delete(grasp_pc_norm, bad_ind[0], axis=0)
    assert(np.sum(grasp_pc_norm != grasp_pc_norm) == 0)

    if len(grasp_pc)<minimal_points_send_to_point_net: # No points left
        return None

    m_width_of_pic = project_size
    margin = projection_margin
    order = np.array([0, 1, 2])
    occupy_pic1, norm_pic1 = cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                 order, gripper_width)
    if project_chann == 12:
        order = np.array([1, 2, 0])
        occupy_pic2, norm_pic2 = cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
        order = np.array([0, 2, 1])
        occupy_pic3, norm_pic3 = cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                 order, gripper_width)
        output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
    else:
        raise NotImplementedError

    return output

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
        feature = project_pc(points_g, config['gripper_width'], in_ind_[i_])
        #in_ind_points_.append(points_g[in_ind_[i_]])
        in_ind_points_.append(feature)
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
               blockRandom=0.05,
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
    look = [0.10, -0.20, 0.60]
    self._cam_pos = look
    distance = 0.1
    pitch = -45
    yaw = -75
    roll = 120
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
    self.fov = 30.
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

    #p.setGravity(0, 0, -9.8)
    p.setGravity(0, 0, 0)
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
    for urdf_name in urdfList:
      xpos = 0.65 + self._blockRandom * random.random()
      ypos = self._blockRandom * (random.random() - .5)
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

  def check_if_grasp_success(self, radius):
    gripper_pos = np.asarray(p.getLinkState(self._tm700.tm700Uid, self._tm700.tmGripperBottomCenterIndex)[0], dtype=np.float32)
    for uid in self._objectUids:
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
  from gdn.utils import *
  from gdn.baseline.gpd import GPDClassifier
  from scipy.spatial.transform import Rotation

  output_path = sys.argv[2]
  assert output_path.endswith(('.txt', '.out', '.log'))
  total_n = int(sys.argv[3])
  cls_k = 2

  gripper_length = config['hand_height']
  deepen_hand = gripper_length * 1.2
  model = GPDClassifier(12)
  model = model.cuda()
  model = model.eval()
  model.load_state_dict(torch.load(sys.argv[1]))

  with open(output_path, 'w') as result_fp:
      #p.connect(p.GUI)
      #p.setAdditionalSearchPath(datapath)
      start_obj_id = 3
      ts = None #1/240.
      #test = tm700_rgbd_gym(width=480, height=480, numObjects=1, objRoot='/home/peter0749/Simple_urdf')
      test = tm700_rgbd_gym(width=720, height=720, numObjects=1, objRoot='/tmp2/peter0749/YCB_valset_urdf')

      test.reset()
      tm_link_name_to_index = get_name_to_link(test._tm700.tm700Uid)
      table_link_name_to_index = get_name_to_link(test.tableUid)

      obj_link_name_to_index = []
      for uid in test._objectUids:
          obj_link_name_to_index.append((uid, get_name_to_link(uid)))

      success_n = 0
      fail_n = 0
      fail_and_ik_fail = 0
      no_solution_fail = 0
      obj_success_rate = {}
      with torch.no_grad():
          for ite in range(total_n):
              test.reset()
              # Naive baseline for testing
              point_cloud, segmentation = test.getTargetGraspObservation()
              pc_flatten = point_cloud.reshape(-1,3).astype(np.float32)
              pc_no_arm = pc_flatten[segmentation.reshape(-1)>0,:] # (N, 3)
              pc_npy = pc_flatten[segmentation.reshape(-1)==start_obj_id,:] # (N, 3)

              real_grasp, points, _ = cal_grasp(pc_npy, test._cam_pos)
              in_ind, in_ind_points = collect_pc(real_grasp, points)
              score_value = []
              assert len(real_grasp) == len(in_ind_points)
              for ii in range(len(in_ind_points)):
                  if in_ind_points[ii] is None or in_ind_points[ii].shape[0] < minimal_points_send_to_point_net:
                      score_value.append(-np.inf)
                  else:
                      score = -np.inf
                      feature = np.transpose(in_ind_points[ii], (2, 0, 1)) # (H, W, 12) -> (12, H, W)
                      try:
                          out = model(torch.from_numpy(feature).float().unsqueeze(0).cuda())
                          if isinstance(out, tuple):
                              out = out[0]
                          score = float(out[0,-1].cpu()) # (#batch,)
                      except TypeError: # FIXME: I don't know why some times model will return complex128...
                          score = -np.inf
                      score_value.append(score)
              ind = np.argsort(-np.asarray(score_value))
              score_value = [ score_value[i] for i in ind  ]
              real_grasp = [ real_grasp[i] for i in ind  ]
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
                  pred_poses.append((score, pose))

              print('Generated1 %d grasps'%len(pred_poses))

              new_pred_poses = []
              for pose in pred_poses:
                  score    = pose[0]
                  rotation = pose[1][:3,:3]
                  trans    = pose[1][:3, 3]
                  approach = rotation[:3,0]
                  # if there is no suitable IK solution can be found. found next
                  # Find more grasp for GPDs since it might not be able to find feasible grasps
                  if np.arccos(np.dot(approach.reshape(1,3), np.array([1, 0,  0]).reshape(3,1))) > np.radians(70):
                      continue
                  if np.arccos(np.dot(approach.reshape(1,3), np.array([0, 0, -1]).reshape(3,1))) > np.radians(89.9):
                      continue
                  tmp_pose = np.append(rotation, trans[...,np.newaxis], axis=1)

                  # Sanity test
                  gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width']+config['thickness']*2,
                                                                         config['hand_height'], tmp_pose,
                                                                         config['thickness_side'], backward=deepen_hand)[1:]
                  gripper_inner1, gripper_inner2 = generate_gripper_edge(config['gripper_width'], config['hand_height'],
                                                                         tmp_pose, config['thickness_side'])[1:]
                  outer_pts = crop_index(pc_no_arm, gripper_outer1, gripper_outer2)
                  if len(outer_pts) == 0: # No points between fingers
                      continue
                  inner_pts = crop_index(pc_no_arm, gripper_inner1, gripper_inner2, search_idx=outer_pts)
                  if len(outer_pts) - len(np.intersect1d(inner_pts, outer_pts)) > 0: # has collision
                      continue

                  trans_backward = trans - approach * deepen_hand

                  tmp_pose = np.append(rotation, trans_backward[...,np.newaxis], axis=1)
                  gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'],
                                                                                             config['hand_height'],
                                                                                             tmp_pose,
                                                                                             config['thickness_side'])
                  gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge
                  if gripper_l_t[2] < 0.003 or gripper_r_t[2] < 0.003 or \
                     gripper_l[2]   < 0.003 or gripper_r[2]   < 0.003: # ready pose will collide with table
                      continue

                  new_pose = np.append(rotation, trans_backward[...,np.newaxis], axis=1)
                  new_pred_poses.append((score, new_pose))
              pred_poses = new_pred_poses
              print('Generated2 %d grasps'%len(pred_poses))

              if len(pred_poses)==0:
                  print("No suitable grasp found.")
                  no_solution_fail += 1
              else:
                  best_grasp = pred_poses[0][1] # (3, 4)
                  print("Confidence: %.4f"%pred_poses[0][0])
                  rotation = best_grasp[:3,:3]
                  trans_backward = best_grasp[:,3]
                  approach = best_grasp[:3,0]
                  trans = trans_backward + approach * deepen_hand
                  pose = np.append(rotation, trans[...,np.newaxis], axis=1)
                  pose_backward = np.append(rotation, trans_backward[...,np.newaxis], axis=1)
                  for link_name, link_id in tm_link_name_to_index.items():
                      p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, link_id, -1, 0)
                      for obj_id, obj in obj_link_name_to_index:
                          for obj_name, obj_link in obj.items():
                            # temporary disable collision detection and move to ready pose
                            p.setCollisionFilterPair(test._tm700.tm700Uid, obj_id, link_id, obj_link, 0)
                  # Ready to grasp pose
                  test.step_to_target_pose([pose_backward, -0.0],  ts=ts, max_iteration=5000, min_iteration=1)
                  # Enable collision detection to test if a grasp is successful.
                  for link_name, link_id in tm_link_name_to_index.items():
                      for obj_id, obj in obj_link_name_to_index:
                          for obj_name, obj_link in obj.items():
                            p.setCollisionFilterPair(test._tm700.tm700Uid, obj_id, link_id, obj_link, 1)
                  # Enable collision detection for gripper head, fingers
                  p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['gripper_link'], -1, 1)
                  p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['finger_r_link'], -1, 1)
                  p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['finger_l_link'], -1, 1)
                  # Deepen gripper hand. May return infeasible pose?
                  for d in np.linspace(0, 1, 100): # linear trajectory
                      info = test.step_to_target_pose([pose*d+pose_backward*(1.-d), -0.0],  ts=ts, max_iteration=100, min_iteration=1)[-1]
                  if not info['planning']:
                      print("Inverse Kinematics failed.")
                  # Grasp it
                  test.step_to_target_pose([pose, 0.2],  ts=ts, max_iteration=500, min_iteration=5)
                  # Test if we can lift the object
                  p.setGravity(0, 0, -10)
                  pose[:3,3] += np.array([0, 0, 0.30])
                  test.step_to_target_pose([pose, 0.2],  ts=ts, max_iteration=5000, min_iteration=5)
                  for _ in range(1000):
                      p.stepSimulation()
                  if not test._current_objList[0] in obj_success_rate:
                      obj_success_rate[test._current_objList[0]] = (0, 0)
                  # Compute success rate for each object
                  obj_iter_n = obj_success_rate[test._current_objList[0]][1] + 1
                  obj_success_n = obj_success_rate[test._current_objList[0]][0]
                  if test.check_if_grasp_success(0.50):
                      print("Grasp success!")
                      success_n += 1
                      obj_success_n += 1
                  else:
                      print("Grasp failed!")
                      fail_n += 1
                      if not info['planning']:
                          fail_and_ik_fail += 1
                      if False:
                          pc_subset = np.copy(pc_no_arm)
                          if len(pc_subset)>5000:
                              pc_subset = pc_subset[np.random.choice(len(pc_subset), 5000, replace=False)]
                          mlab.clf()
                          mlab.points3d(pc_subset[:,0], pc_subset[:,1], pc_subset[:,2], scale_factor=0.004, mode='sphere', color=(1.0,1.0,0.0), opacity=1.0)
                          for i in range(min(30, len(pred_poses))):
                              pose = np.copy(pred_poses[i][1]).astype(np.float32)
                              pose[:,3] += pose[:,0] * deepen_hand
                              gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
                              gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge

                              mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=config['thickness']/4., color=(0,0,1) if i>0 else (1,0,0), opacity=0.5)
                              mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=config['thickness']/4., color=(0,0,1) if i>0 else (1,0,0), opacity=0.5)
                              mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=config['thickness']/4., color=(0,0,1) if i>0 else (1,0,0), opacity=0.5)
                          mlab.show()
                          input()
                  obj_success_rate[test._current_objList[0]] = (obj_success_n, obj_iter_n)
              result_fp.write("Iteration %d:\n"%(ite+1))
              for obj_name in sorted(list(obj_success_rate)):
                  s_n = obj_success_rate[obj_name]
                  rate = float(s_n[0]) / float(s_n[1])
                  result_fp.write("%s : %.2f (%d / %d)\n"%(obj_name, rate, *s_n))
              result_fp.write("Success rate (current): %.4f (%d | %d | %d | %d)\n"%(success_n / (ite+1), success_n, fail_n, fail_and_ik_fail, no_solution_fail))
              result_fp.flush()
              assert success_n + fail_n + no_solution_fail == ite+1
      result_fp.write("Final Success rate: %.6f\n"%(success_n/total_n))
      result_fp.flush()
