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
#from mayavi import mlab


with open('./gripper_config.json', 'r') as fp:
    config = json.load(fp)
    config['thickness'] = 0.003 # Force overwrite


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
               blockRandom=0.25,
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
    self.fov = 40.
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
    for urdf_name in urdfList:
      xpos = 0.50 + self._blockRandom * random.random()
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
    success_uids = []
    for uid in self._objectUids:
      # simple check: if there is points of the object near the gripper finger tip
      blockPos, blockOrn = p.getBasePositionAndOrientation(uid)
      blockPos = np.asarray(blockPos, dtype=np.float32)
      # test if this object is lifted and gripper is "holding" the object (not kicked out by the gripper).
      if blockPos[2]>0.20 and np.linalg.norm(gripper_pos-blockPos, ord=2) < radius:
          success_uids.append(uid)
    return success_uids


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
  import pcl
  import torch
  torch.backends.cudnn.benchmark = True
  torch.multiprocessing.set_start_method('forkserver')
  from gdn.representation.euler import *
  #from gdn.utils import crop_index
  #from gdn.utils import generate_gripper_edge
  from gdn.detector.edgeconv.backbone import EdgeDet
  #from nms import nms as gripper_nms
  from nms import decode_euler_feature
  from nms import initEigen, sanity_check
  from nms import crop_index, generate_gripper_edge
  from scipy.spatial.transform import Rotation

  os.environ['OMP_NUM_THREADS'] = '8'
  initEigen(0)

  output_path = sys.argv[2]
  assert output_path.endswith(('.txt', '.out', '.log'))
  total_n = int(sys.argv[3])

  gripper_length = config['hand_height']
  deepen_hand = gripper_length + 0.01
  model = EdgeDet(config, activation_layer=EulerActivation())
  model = model.cuda()
  model = model.eval()
  model.load_state_dict(torch.load(sys.argv[1])['base_model'])
  representation = EulerRepresentation(config)
  subsampling_util = val_collate_fn_setup(config)

  with open(output_path, 'w') as result_fp:
      #p.connect(p.GUI)
      #p.setAdditionalSearchPath(datapath)
      start_obj_id = 3
      #desired_input_points = 2048
      #desired_obj_range = 0.20 # assume 20cm of value range
      input_points = 2048
      ts = None #1/240.
      #test = tm700_rgbd_gym(width=480, height=480, numObjects=1, objRoot='//peter0749/Simple_urdf')
      test = tm700_rgbd_gym(width=720, height=720, numObjects=7, objRoot='/home/peter/YCB_valset_urdf')

      test.reset()
      tm_link_name_to_index = get_name_to_link(test._tm700.tm700Uid)
      table_link_name_to_index = get_name_to_link(test.tableUid)

      obj_link_name_to_index = []
      for uid in test._objectUids:
          obj_link_name_to_index.append((uid, get_name_to_link(uid)))

      success_n = 0
      max_tries = 3
      with torch.no_grad():
          for ite in range(total_n):
              test.reset()
              object_set = list(test._objectUids)
              object_set.sort()
              object_to_ind = { k:n for (n,k) in enumerate(object_set)  }
              grasp_success_obj = np.zeros(len(object_set), dtype=np.bool)
              grasp_failure_obj = np.zeros(len(object_set)+1, dtype=np.int32)
              while (not grasp_success_obj.all()) and grasp_failure_obj.max()<max_tries:
                  point_cloud, segmentation = test.getTargetGraspObservation()
                  pc_flatten = point_cloud.reshape(-1,3).astype(np.float32)
                  pc_no_arm = pc_flatten[segmentation.reshape(-1)>0,:] # (N, 3)
                  pc_npy = pc_flatten[segmentation.reshape(-1)>=3,:] # (N, 3)
                  if len(pc_npy)==0:
                      break
                  pc_npy_max = np.max(pc_npy, axis=0)
                  pc_npy_min = np.min(pc_npy, axis=0)
                  trans_to_frame = (pc_npy_max + pc_npy_min) / 2.0
                  trans_to_frame[2] = np.min(pc_npy[:,2])
                  objs_value_range = (pc_npy_max - pc_npy_min).max()
                  # Normalize points density
                  #input_points = min(4096, int(desired_input_points * max(1.0, (objs_value_range / desired_obj_range)**1.25)))
                  #print("Input points: %d"%input_points)
                  while pc_npy.shape[0]<input_points:
                      new_pts = pc_npy[np.random.choice(len(pc_npy), input_points-len(pc_npy), replace=True)]
                      new_pts = new_pts + np.random.randn(*new_pts.shape) * 1e-6
                      pc_npy = np.append(pc_npy, new_pts, axis=0)
                      pc_npy = np.unique(pc_npy)
                  if pc_npy.shape[0]>input_points:
                      pc_npy = pc_npy[np.random.choice(len(pc_npy), input_points, replace=False)]

                  pc_npy -= trans_to_frame
                  start_ts = time.time()
                  pc_batch, indices, reverse_lookup_index, _ = subsampling_util([(pc_npy,None),])
                  ss_ts = time.time()
                  print("Subsampling in %.2f seconds."%(ss_ts-start_ts))
                  try:
                      pred = model(pc_batch.cuda(), [pt_idx.cuda() for pt_idx in indices]).cpu().numpy()
                  except:
                      continue
                  inf_ts = time.time()
                  print("Inference in %.2f seconds."%(inf_ts-ss_ts))
                  feat_ts = time.time()
                  pred_poses = np.asarray(decode_euler_feature(
                        pc_npy[reverse_lookup_index[0]]+trans_to_frame[np.newaxis],
                        pred[0].reshape(1,-1),
                        *pred[0].shape[:-1],
                        config['hand_height'],
                        config['gripper_width'],
                        config['thickness_side'],
                        config['rot_th'],
                        config['trans_th'],
                        5000, # max number of candidate
                        -np.inf, # threshold of candidate
                        1000,  # max number of grasp in NMS
                        11,    # number of threads
                        True  # use NMS
                      ), dtype=np.float32)
                  print('Generated0 %d grasps.'%len(pred_poses))
                  filter_ts = time.time()
                  print("Decode in %.2f seconds."%(filter_ts-feat_ts))
                  pred_poses = sanity_check(pc_no_arm, pred_poses, 30,
                          config['gripper_width'],
                          config['thickness'],
                          config['hand_height'],
                          config['thickness_side'],
                          11 # num threads
                          )
                  end_ts = time.time()
                  print("Filter in %.2f seconds."%(end_ts-filter_ts))
                  print('Generated1 %d grasps in %.2f seconds.'%(len(pred_poses), end_ts-start_ts))

                  new_pred_poses = []
                  for pose in pred_poses:
                      rotation = pose[:3,:3]
                      trans    = pose[:3, 3]
                      approach = rotation[:3,0]
                      # if there is no suitable IK solution can be found. found next
                      if np.arccos(np.dot(approach.reshape(1,3), np.array([1, 0,  0]).reshape(3,1))) > np.radians(70):
                          continue
                      if np.arccos(np.dot(approach.reshape(1,3), np.array([0, 0, -1]).reshape(3,1))) > np.radians(89):
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
                      if not (gripper_l_t[2] > -0.003 and gripper_r_t[2] > -0.003 and \
                              gripper_l[2]   > -0.003 and gripper_r[2]   > -0.003 and \
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
                      if gripper_l_t[2] < -0.003 or gripper_r_t[2] < -0.003 or \
                         gripper_l[2]   < -0.003 or gripper_r[2]   < -0.003: # ready pose will collide with table
                          continue

                      new_pose = np.append(rotation, trans_backward[...,np.newaxis], axis=1)
                      new_pred_poses.append(new_pose)
                  pred_poses = new_pred_poses
                  print('Generated2 %d grasps'%len(pred_poses))
                  if len(pred_poses)==0:
                      grasp_failure_obj[-1] += 1
                      print("No suitable grasp found.")
                      print("Failure hist: ", grasp_failure_obj)
                      continue
                  this_grasp_success = False

                  tried_top1_pose = None
                  for best_grasp in pred_poses:
                      rotation = best_grasp[:3,:3]
                      trans_backward = best_grasp[:,3]
                      approach = best_grasp[:3,0]
                      trans = trans_backward + approach*deepen_hand
                      pose = np.append(rotation, trans[...,np.newaxis], axis=1)
                      pose_backward = np.append(rotation, trans_backward[...,np.newaxis], axis=1)

                      gripper_inner1, gripper_inner2 = generate_gripper_edge(config['gripper_width'], config['hand_height'],
                                                                             pose, config['thickness_side'], 0.0)[1:]
                      outer_pts = crop_index(pc_flatten, gripper_inner1, gripper_inner2)
                      region = segmentation.reshape(-1)[outer_pts]
                      region = region[region>=min(object_set)]
                      if len(region)==0:
                          grasp_uid_guess = len(object_set) # no grasp found
                      else:
                          hist = np.bincount(region)
                          grasp_uid_guess = hist.argmax()
                      for link_name, link_id in tm_link_name_to_index.items():
                          p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, link_id, -1, 0)
                          for obj_id, obj in obj_link_name_to_index:
                              for obj_name, obj_link in obj.items():
                                # temporary disable collision detection and move to ready pose
                                p.setCollisionFilterPair(test._tm700.tm700Uid, obj_id, link_id, obj_link, 0)

                      # Ready to grasp pose
                      test._tm700.home()
                      info = test.step_to_target_pose([pose_backward, -0.0],  ts=ts, max_iteration=5000, min_iteration=1)[-1]
                      if tried_top1_pose is None:
                          tried_top1_pose = (pose_backward, pose)
                      if info['planning']:
                          break
                      else:
                          print("Inverse Kinematics failed.")
                  if (not info['planning']) and (not tried_top1_pose is None): # Planning failed
                      pose_backward, pose = tried_top1_pose
                  # Enable collision detection to test if a grasp is successful.
                  for link_name, link_id in tm_link_name_to_index.items():
                      for obj_id, obj in obj_link_name_to_index:
                          for obj_name, obj_link in obj.items():
                            p.setCollisionFilterPair(test._tm700.tm700Uid, obj_id, link_id, obj_link, 1)
                  # Enable collision detection for gripper head, fingers
                  p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['gripper_link'], -1, 1)
                  p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['finger_r_link'], -1, 1)
                  p.setCollisionFilterPair(test._tm700.tm700Uid, test.tableUid, tm_link_name_to_index['finger_l_link'], -1, 1)
                  # Deepen gripper hand
                  for d in np.linspace(0, 1, 100):
                      info = test.step_to_target_pose([pose*d+pose_backward*(1.-d), -0.0],  ts=ts, max_iteration=50, min_iteration=1)[-1]
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
                  uids = test.check_if_grasp_success(0.50)
                  if len(uids)>0:
                      print("Grasp success!")
                      this_grasp_success = True
                  else:
                      print("Grasp failed!")

                  if this_grasp_success:
                      for uid in uids:
                          test._objectUids.remove(uid)
                          p.removeBody(uid)
                          grasp_success_obj[object_to_ind[uid]] = True
                  else:
                      grasp_failure_obj[object_to_ind[grasp_uid_guess]] += 1
                  print("Failure hist: ", grasp_failure_obj)
                  print("Success hist: ", grasp_success_obj)
                  if grasp_success_obj.all():
                      success_n += 1
                      break
                  result_fp.write("Complete rate: %.6f (%d / %d)\n"%(success_n/(ite+1), success_n, ite+1))
                  result_fp.flush()
