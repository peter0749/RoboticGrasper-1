# Code base from pybullet examples https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/ kuka_diverse_object_gym_env.py
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


import random
import os
from gym import spaces
import time
import pybullet as p
import numpy as np
import pybullet_data
import glob
from pkg_resources import parse_version
import gym
from bullet.tm700 import tm700
from unused_code.tm700_possensorbothgrippers_Gym import tm700_possensor_gym
from pathlib import Path
from tqdm import tqdm


class tm700_rgbd_gym(tm700_possensor_gym):
  """Class for tm700 environment with diverse objects.

  """

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
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
    # self._timeStep = 1. / 240.
    self._timeStep = 1. / 60.
    self._urdfRoot = urdfRoot
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
    # height and width of camera images
    self._width = 256
    self._height = 256

    self._numObjects = numObjects
    self._isTest = isTest
    self.observation_space = spaces.Box(low=0,
                                         high=255,
                                         shape=(self._height, self._width, 3),
                                         dtype=np.uint8)
    self.img_save_cnt = 0
    self.model_paths = self.get_data_path()


    # disable GUI or not
    self.cid = p.connect(p.DIRECT)

    # if self._renders:
    #   self.cid = p.connect(p.SHARED_MEMORY)
    #   if (self.cid < 0):
    #     self.cid = p.connect(p.GUI)
    #   p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33]) # cameraposition of rendering
    # else:
    #   self.cid = p.connect(p.DIRECT)

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


  def get_data_path(self):
    root = Path('/data/ShapeNet_subset/')
    class_paths = [c for c in root.iterdir() if c.is_dir() and 'ipynb' not in str(c)]
    object_paths = []
    for class_path in class_paths:
        obj_paths = [o for o in class_path.iterdir() if o.is_dir()]
        tmp = []
        for o in obj_paths:
            tmp.append(str(o / Path('models/model_normalized.obj')))
        object_paths += tmp
        
    random.seed(5)
    random.shuffle(object_paths)
    return object_paths


  def reset(self):
    """Environment reset called at the beginning of an episode.
    """
    # Set the camera settings.

    # look = [0.1, 0., 0.44]
    # distance = 0.8
    # pitch = -90
    # yaw = -90
    # roll = 180

    #[0.4317558029454219, 0.1470448842904527, 0.2876218894185256]#[0.23, 0.2, 0.54] # from where the input is
    # set 1
    # look = [0.1, -0.3, 0.54] 
    # distance = 1.0
    # pitch = -45 + self._cameraRandom * np.random.uniform(-3, 3)
    # yaw = -45 + self._cameraRandom * np.random.uniform(-3, 3)
    # roll = 145
    # pos_range = [0.3, 0.7, -0.2, 0.3]

    # set 2
    # look = [-0.3, -0.8, 0.54] 
    # distance = 0.7
    # pitch = -28 + self._cameraRandom * np.random.uniform(-3, 3)
    # yaw = -45 + self._cameraRandom * np.random.uniform(-3, 3)
    # roll = 180
    # pos_range = [0.1, 0.8, -0.45, 0.15]

    # set 3
    look = [0.4, 0.1, 0.54] 
    distance = 2.0
    pitch = -90
    yaw = -90
    roll = 180
    pos_range = [0.3, 0.7, -0.2, 0.3]

    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
    fov = 20. + self._cameraRandom * np.random.uniform(-2, 2)
    aspect = self._width / self._height
    near = 0.01
    far = 10
    self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    self._attempted_grasp = False
    self._env_step = 0
    self.terminated = 0

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

    self.tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.640000,
                               0.000000, 0.000000, 0.0, 1.0)

    p.setGravity(0, 0, -10)
    # self._tm700 = tm700(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)

    self._envStepCounter = 0
    p.stepSimulation()

    # num of objs to be placed
    while len(self.model_paths) != 0:
      urdfList = []
      num_obj = random.randint(1, 3)
      for i in range(num_obj):
        urdfList.append(self.model_paths.pop())
      print('img {img_cnt}, {n_obj} objects'.format(img_cnt=self.img_save_cnt+1, n_obj=len(urdfList)))
      self._objectUids = self._randomly_place_objects(urdfList, pos_range)
      self._observation = self._get_observation()
      self.img_save_cnt += 1
      for uid in self._objectUids:
        p.removeBody(uid)
      if self.img_save_cnt == 2: 
        raise Exception('stop')

    return np.array(self._observation)


  def _randomly_place_objects(self, urdfList, pos_range):
    objectUids = []

    shift = [0, -0.02, 0]
    meshScale = [0.3, 0.3, 0.3]

    list_x_pos = []
    list_y_pos = []
    random.seed(self.img_save_cnt)
    for idx, urdf_path in enumerate(urdfList):
      list_x_pos.append(random.uniform(pos_range[0], pos_range[1]))
      list_y_pos.append(random.uniform(pos_range[2], pos_range[3]))

    #the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
    for idx, urdf_path in enumerate(urdfList):
      visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                          fileName=urdf_path,
                                          rgbaColor=[1, 1, 1, 1],
                                          specularColor=[0.4, .4, 0],
                                          visualFramePosition=shift,
                                          meshScale=meshScale)
      collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName=urdf_path,
                                          collisionFramePosition=shift,
                                          meshScale=meshScale)

      # xpos = 0.2 + self._blockRandom * random.random() + 0.1*idx
      # ypos = 0.1 + self._blockRandom * (random.random() - .5) + 0.1* (random.randint(0,6) - 3) 
      xpos = list_x_pos[idx]
      ypos = list_y_pos[idx]

      uid = p.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=[xpos, ypos, .15],
                      useMaximalCoordinates=True)

      objectUids.append(uid)
      for _ in range(100):
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
    segmentation = img_arr[4]
    depth = np.reshape(depth, (self._height, self._width,1) )
    segmentation = np.reshape(segmentation, (self._height, self._width,1) )

    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    np_img_arr = np_img_arr[:, :, :3].astype(np.float64)

    view_mat = np.asarray(self._view_matrix).reshape(4, 4)
    proj_mat = np.asarray(self._proj_matrix).reshape(4, 4)
    # pos = np.reshape(np.asarray(list(p.getBasePositionAndOrientation(self._objectUids[0])[0])+[1]), (4, 1))

    AABBs = np.zeros((len(self._objectUids), 2, 3))
    for i, _uid in enumerate(self._objectUids):
      AABBs[i] = np.asarray(p.getAABB(_uid)).reshape(2, 3)

    np.save('/home/tony/Desktop/obj_save/view_mat_'+str(self.img_save_cnt), view_mat)
    np.save('/home/tony/Desktop/obj_save/proj_mat_'+str(self.img_save_cnt), proj_mat)
    np.save('/home/tony/Desktop/obj_save/img_'+str(self.img_save_cnt), np_img_arr.astype(np.int16))
    np.save('/home/tony/Desktop/obj_save/AABB_'+str(self.img_save_cnt), AABBs)

    test = np.concatenate([np_img_arr[:, :, 0:2], segmentation], axis=-1)

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
      reward = -((closestPoints1[0][8])**2 + (closestPoints2[0][8])**2 )*(1/2)*(1/0.17849278457978357)
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
    selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
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

  p.connect(p.GUI)
  #p.setAdditionalSearchPath(datapath)
  test =tm700_rgbd_gym()
  test.step([0, 0, 0, 0, 0, -0.25, 0.25])
  time.sleep(50)
