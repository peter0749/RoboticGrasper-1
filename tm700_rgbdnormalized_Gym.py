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
               blockRandom=0.1,
               cameraRandom=0,
               width=64,
               height=64,
               numObjects=5,
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
    look = [0.80, -0.30, 0.45]
    distance = 0.15
    pitch = -45
    yaw = 45 # -45
    roll = 180
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
    self.fov = 40.
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

    #p.setGravity(0, 0, -10)
    p.setGravity(0, 0, 0)
    self._tm700 = tm700(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)

    self._envStepCounter = 0
    p.stepSimulation()

    # Choose the objects in the bin.
    # urdfList = self._get_random_object(self._numObjects, self._isTest)
    urdfList = self._get_random_urdf(self._numObjects)
    # urdfList = self._get_block()
    self._objectUids = self._randomly_place_objects(urdfList)
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
      xpos = 0.5 + self._blockRandom * random.random()
      ypos = self._blockRandom * (random.random() - .5)
      orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(-np.pi/2.0, np.pi/2.0)])
      #uid = p.loadURDF(urdf_name, [xpos, ypos, -0.012], [orn[0], orn[1], orn[2], orn[3]])
      uid = p.loadURDF(urdf_name, [xpos, ypos, 0.15], [orn[0], orn[1], orn[2], orn[3]])
      objectUids.append(uid)
      # Let each object fall to the tray individual, to prevent object
      # intersection.
      for _ in range(500):
        p.stepSimulation()
    return objectUids

  # def _get_onrobot_observation(self):
  #   """Return the observation as an image.
  #
  #
  #   """
  #   camerapos = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmEndEffectorIndex+2)
  #   blockPos, blockOrn = p.getBasePositionAndOrientation(self._objectUids[0])
  #   camerapos = camerapos[0]
  #   print(camerapos)
  #   distance = 1# camerapos[2]
  #   pitch = -90 #-60 + self._cameraRandom * np.random.uniform(-3, 3) #original -56
  #   yaw =0# 245 + self._cameraRandom * np.random.uniform(-3, 3)
  #   roll = 0
  #   # self._view_matrix = p.computeViewMatrixFromYawPitchRoll(camerapos, distance, yaw, pitch, roll, 2)
  #   self._view_matrix = p.computeViewMatrix(
  #     cameraEyePosition=camerapos,
  #     cameraTargetPosition=blockPos,
  #     cameraUpVector=[0,0, 1])
  #   img_arr = p.getCameraImage(width=self._width,
  #                              height=self._height,
  #                              viewMatrix=self._view_matrix,
  #                              projectionMatrix=self._proj_matrix)
  #   rgb = img_arr[2]
  #   depth = img_arr[3]
  #   min = 0.97
  #   max=1.0
  #   print(depth)
  #   depthnormalized = (float(depth) - min)/(max-min)
  #   segmentation = img_arr[4]
  #   depth = np.reshape(depth, (self._height, self._width,1) )
  #   segmentation = np.reshape(segmentation, (self._height, self._width,1) )
  #
  #   np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
  #   np_img_arr = np_img_arr.astype(np.float64)
  #
  #   test = np.concatenate([np_img_arr[:, :, 1:3], depth], axis=-1)
  #
  #   return test

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

  def check_if_grasp_success(self):
    for uid in self._objectUids:
      blockPos, blockOrn = p.getBasePositionAndOrientation(uid)
      if blockPos[2]>0.20:
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
      print("successfully grasped a block!!!")
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



if __name__ == '__main__':
  import sys
  import pcl
  import json
  import torch
  from gdn.representation.euler import *
  from gdn.utils import *
  from gdn.detector.edgeconv.backbone import EdgeDet
  from scipy.spatial.transform import Rotation

  with open('./gripper_config.json', 'r') as fp:
      config = json.load(fp)

  gripper_length = config['hand_height']
  deepen_hand = 0.10
  model = EdgeDet(config, activation_layer=EulerActivation())
  model = model.cuda()
  model = model.eval()
  model.load_state_dict(torch.load(sys.argv[1])['base_model'])
  representation = EulerRepresentation(config)
  subsampling_util = val_collate_fn_setup(config)

  p.connect(p.GUI)
  #p.setAdditionalSearchPath(datapath)
  start_obj_id = 3
  ts = 1/240.
  #test = tm700_rgbd_gym(width=480, height=480, numObjects=1, objRoot='/home/peter0749/Simple_urdf')
  test = tm700_rgbd_gym(width=480, height=480, numObjects=1, objRoot='/home/peter0749/YCB_valset_urdf')
  #test.reset()
  #test.step_to_target_pose([0.4317596244807792, 0.1470447615125933, 0.40, 0, 0, 0, 0], ts=1/240.)
  #time.sleep(3)
  #exit(0)

  first = True
  with torch.no_grad():
      while True:
          test.reset()
          # Naive baseline for testing
          point_cloud, segmentation = test.getTargetGraspObservation()
          pc_flatten = point_cloud.reshape(-1,3).astype(np.float32)
          '''
          random_pos = np.mean(pc_flatten[segmentation.reshape(-1)==start_obj_id,:] + np.array([[0, 0, 0.12]]), axis=0)
          test.step_to_target_pose([*random_pos, 0, np.pi/2.0, 0, 0.0],   ts=ts, max_iteration=1000)
          test.step_to_target_pose([*random_pos, 0, np.pi/2.0, 0, 0.2],  ts=ts, max_iteration=500)
          up_ops = random_pos + np.array([0, 0, 0.23])
          test.step_to_target_pose([*up_ops, 0, np.pi/2.0, 0, 0.2],  ts=ts, max_iteration=1000)
          continue
          '''
          if first:
              pc = pcl.PointCloud(pc_flatten)
              pc.to_file(b'test.pcd')
          pc_npy = pc_flatten[segmentation.reshape(-1)==start_obj_id,:] # (N, 3)
          if pc_npy.shape[0]<2048:
              pc_npy = np.append(pc_npy, pc_npy[np.random.choice(len(pc_npy), 2048-len(pc_npy), replace=True)], axis=0)
          trans_to_frame = (np.max(pc_npy, axis=0) + np.min(pc_npy, axis=0)) / 2.0
          trans_to_frame[2] = np.min(pc_npy[:,2])
          pc_npy -= trans_to_frame
          if first:
              pc = pcl.PointCloud(pc_npy)
              pc.to_file(b'test_local.pcd')
          first = False
          pc_npy = pc_npy[np.newaxis] # (1, N, 3)
          pc_batch, indices, reverse_lookup_index, _ = subsampling_util([(pc_npy[0],None),])
          pred = model(pc_batch.cuda(), [pt_idx.cuda() for pt_idx in indices]).cpu().numpy()
          pred_poses = representation.retrive_from_feature_volume_batch(pc_npy, reverse_lookup_index, pred, n_output=1000, threshold=-1e8, nms=False)[0]
          pred_poses = [(x[0], np.append(x[1][:3,:3], x[1][:3,3:4]+trans_to_frame[...,np.newaxis], axis=1)) for x in pred_poses]
          pred_poses = representation.filter_out_invalid_grasp_batch(pc_flatten[np.newaxis], [pred_poses])[0]
          new_pred_poses = []
          for pose in pred_poses:
              score    = pose[0]
              rotation = pose[1][:3,:3]
              trans    = pose[1][:3, 3]
              approach = rotation[:3,0]
              trans_backward = trans - approach*deepen_hand
              trans_backward = trans
              new_pred_poses.append((score, np.append(rotation, trans_backward[...,np.newaxis], axis=1)))
          pred_poses = representation.filter_out_invalid_grasp_batch(pc_flatten[np.newaxis], [new_pred_poses])[0]
          best_grasp = pred_poses[0][1] # (3, 4)
          rpy = Rotation.from_matrix(best_grasp[:3,:3]).as_euler('xyz')
          trans_backward = best_grasp[:,3]
          print(best_grasp[:3,0])
          trans = trans_backward + best_grasp[:3,0]*deepen_hand
          test.step_to_target_pose([*trans_backward, *rpy, 0.0],  ts=ts, max_iteration=1000)
          time.sleep(10)
          test.step_to_target_pose([*trans, *rpy, 0.0],  ts=ts, max_iteration=1000)
          test.step_to_target_pose([*trans, *rpy, 0.2],  ts=ts, max_iteration=500)
          up_ops = trans + np.array([0, 0, 0.23])
          test.step_to_target_pose([*up_ops,*rpy, 0.2],  ts=ts, max_iteration=1000)
          if test._graspSuccess:
              print("Grasp success!")
          else:
              print("Grasp failed!")


