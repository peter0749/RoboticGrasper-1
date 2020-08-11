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
               blockRandom=0.15,
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

  def rotate_camera(self, roll=0.0, pitch=0.0, yaw=0.0):
    self._cam_roll += roll
    self._cam_pitch += pitch
    self._cam_yaw += yaw
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(self._cam_pos, self._cam_distance, self._cam_yaw, self._cam_pitch, self._cam_roll, 2)

  def reset(self):
    """Environment reset called at the beginning of an episode.
    """
    # Set the camera settings.
    self._cam_pos = [0.40, 0.00, 0.00]
    self._cam_distance = 1.0
    self._cam_pitch = -45
    self._cam_yaw = -75
    self._cam_roll = 120
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(self._cam_pos, self._cam_distance, self._cam_yaw, self._cam_pitch, self._cam_roll, 2)
    self.fov = 40.

    self.focal_length_x = self._width / np.tan(np.radians(self.fov)/2.0)
    self.focal_length_y = self._height / np.tan(np.radians(self.fov)/2.0)
    self._cam_aspect = self._width / self._height
    self.d_near = 0.01
    self.d_far = 1.5
    self._proj_matrix = p.computeProjectionMatrixFOV(self.fov, self._cam_aspect, self.d_near, self.d_far)

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
  #from mayavi import mlab
  #import pcl
  import open3d as o3d

  output_prefix = sys.argv[1]
  start_ite = int(sys.argv[2])
  total_n = int(sys.argv[3])

  if not os.path.exists(output_prefix):
      os.makedirs(output_prefix)

  #p.connect(p.GUI)
  ts = None #1/240.
  test = tm700_rgbd_gym(width=300, height=300, numObjects=10, objRoot='./YCB_trainset_urdf')

  start_ts = time.time()
  elapsed_ite = 0
  for ite in range(start_ite, start_ite+total_n):
      test.reset()
      table_link_name_to_index = get_name_to_link(test.tableUid)

      pcs = None
      for _ in range(8):
          point_cloud, seg = test.getTargetGraspObservation()
          seg = seg.reshape(-1)
          if pcs is None:
              pcs = point_cloud.reshape(-1, 3)[seg>=1].astype(np.float32)
          else:
              pcs = np.append(pcs, point_cloud.reshape(-1, 3)[seg>=1].astype(np.float32), axis=0)
          test.rotate_camera(yaw=45.0)

      '''
      pc_subset = pcl.PointCloud(pcs)
      vx = pc_subset.make_voxel_grid_filter()
      vx.set_leaf_size(0.003, 0.003, 0.003)
      pc_subset = np.array(vx.filter(), dtype=np.float32)
      pc_subset = pc_subset[np.abs(pc_subset[:,0]-(test._blockRandom*0.5+0.40))<0.30]
      pc_subset = pc_subset[np.abs(pc_subset[:,1])<0.30]
      pc_pcd    = pcl.PointCloud(pc_subset)
      '''
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(pcs)
      downpcd = pcd.voxel_down_sample(voxel_size=0.003)
      del pcd
      pc_subset = np.array(downpcd.points, dtype=np.float32)
      del downpcd

      pc_subset = pc_subset[np.abs(pc_subset[:,0]-(test._blockRandom*0.5+0.40))<0.30]
      pc_subset = pc_subset[np.abs(pc_subset[:,1])<0.30]

      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(pc_subset)

      '''
      mlab.points3d(pc_subset[:,0], pc_subset[:,1], pc_subset[:,2], scale_factor=0.01)
      mlab.show()
      input()
      '''

      output_pcd = output_prefix + '/scene-%d.pcd'%(ite+1)
      #output_ply = output_prefix + '/scene-%d.ply'%(ite+1)
      output_npy = output_prefix + '/scene-%d.npy'%(ite+1)

      np.save(output_npy, pc_subset, allow_pickle=True, fix_imports=True)
      #pcl.save(pc_pcd, output_pcd, format='pcd', binary=True)
      o3d.io.write_point_cloud(output_pcd, pcd, write_ascii=False, compressed=True)
      #pcl.save(pc_pcd, output_ply, format='ply')
      save_ts = time.time()
      elapsed_ite += 1
      eta_min = (save_ts - start_ts) / elapsed_ite * (total_n-elapsed_ite) / 60.0
      print("[ETA: %.1f min] Saved %s w/ %d points"%(eta_min, output_pcd, len(pc_subset)))
      sys.stdout.flush()
