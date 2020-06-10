# Code base from pybullet examples https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/ kuka_diverse_object_gym_env.py

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
from pkg_resources import parse_version
import tm700
largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960
maxSteps = 700
Dv = 0.004


class tm700_possensor_gym(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               width=64,
               height=64,
               maxSteps=maxSteps):
    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
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
    self._height = width
    self._width = height

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid < 0):
        cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      p.connect(p.DIRECT)
    self.seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
      action_dim = 3
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def reset(self):

    look = [0.4, 0.1, 0.54]
    distance = 1.5
    pitch = -90
    yaw = -90
    roll = 180
    pos_range = [0.45, 0.5, 0.0, 0.1]
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
    fov = 20.
    aspect = self._width / self._height
    near = 0.01
    far = 10
    self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

    self.tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.640000,
               0.000000, 0.000000, 0.0, 1.0)

    xpos = 0.55 + 0.12 * random.random()
    ypos = 0 + 0.2 * random.random()
    ang = 3.14 * 0.5 +1.5 #* random.random()
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "jenga/jenga.urdf"), xpos, ypos, 0.1,
                               orn[0], orn[1], orn[2], orn[3])

    p.setGravity(0, 0, -10)
    self._tm700 = tm700.tm700(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getTargetGraspObservation(self, return_camera=True, **kwargs):
    if return_camera:
        img_arr = p.getCameraImage(width=self._width,
            height=self._height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix)
        depth = img_arr[3]
        segmentation = img_arr[4]
        depth = np.reshape(depth, (self._height, self._width, 1) )
        segmentation = np.reshape(segmentation, (self._height, self._width, 1) )
        return depth, segmentation, self._view_matrix, self._proj_matrix
    else:
        return self._view_matrix, self._proj_matrix

  def getExtendedObservation(self):
    self._observation = self._tm700.getObservation()
    gripperState = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmFingerIndexL)
    gripperStateR = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmFingerIndexR)

    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    gripperPosR = gripperStateR[0]
    gripperOrnR = gripperStateR[1]
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)

    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    invGripperPosR, invGripperOrnR = p.invertTransform(gripperPosR, gripperOrnR)

    gripperMat = p.getMatrixFromQuaternion(gripperOrn)
    gripperMatR = p.getMatrixFromQuaternion(gripperOrnR)

    blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                blockPos, blockOrn)
    blockPosInGripperR, blockOrnInGripperR = p.multiplyTransforms(invGripperPosR, invGripperOrnR,
                                                                blockPos, blockOrn)
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
    blockEulerInGripperR = p.getEulerFromQuaternion(blockOrnInGripperR)

    #we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]
    blockInGripperPosXYEulZR = [blockPosInGripperR[0], blockPosInGripperR[1], blockEulerInGripper[2]]

    self._observation.extend(list(blockInGripperPosXYEulZ))
    self._observation.extend(list(blockInGripperPosXYEulZR))

    return self._observation


  def step(self, action):
    if (self._isDiscrete):
      dv = Dv
      dx = [0, -dv, dv, 0, 0, 0, 0][action]
      dy = [0, 0, 0, -dv, dv, 0, 0][action]
      da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
      f = 0.15
      realAction = [dx, dy, -0.0005, da, f]
    else:
      dv = Dv
      dx = action[0] * dv
      dy = action[1] * dv
      da = action[2] * 0.05
      f = 0.15
      realAction = [dx, dy, -0.0005, da, f]
    return self.step2(realAction)

  def step_to_target_pose(self, action, max_iteration=5000, min_iteration=100, trans_eps=0.003, rot_eps=2.2, ts=None, **kwargs):
    for ite in range(max_iteration):
      observation, reward, done, state, info = self.step3(action, **kwargs)
      print(ite)
      if done:
        break
      target_t = np.asarray(state[0])
      target_r = np.asarray(state[1])
      trans_d =   np.linalg.norm(target_t-action[:3], ord=2)
      rot_d = min(np.linalg.norm(target_r-action[3:], ord=2), np.linalg.norm(target_r+action[3:], ord=2))
      if trans_d<trans_eps and rot_d<rot_eps and ite>=min_iteration:
        break
      if not ts is None and ts>0:
          time.sleep(ts)
    return observation, reward, done, state, info

  def step3(self, action, **kwargs):
    state = self._tm700.applyActionIK(action)
    p.stepSimulation()
    self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getTargetGraspObservation(**kwargs)
    done = self._termination()
    reward = self._reward()
    return self._observation, reward, done, state, {}

  def step2(self, action):
    for i in range(self._actionRepeat):
      self._tm700.applyAction(action)
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()

    done = self._termination()
    npaction = np.array([
        action[3]
    ])  #only penalize rotation until learning works well [action[0],action[1],action[3]])
    actionCost = np.linalg.norm(npaction) * 10.
    reward = self._reward() - actionCost

    #print("len=%r" % len(self._observation))

    return np.array(self._observation), reward, done, {}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos, orn = self._p.getBasePositionAndOrientation(self._tm700.tm700Uid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
    #renderer=self._p.ER_TINY_RENDERER)

    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    state = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmEndEffectorIndex)
    actualEndEffectorPos = state[0]

    if (self.terminated or self._envStepCounter > self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.006
    closestPoints = p.getClosestPoints(self.tableUid, self._tm700.tm700Uid, maxDist, -1, self._tm700.tmFingerIndexL)

    if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1

      #start grasp and terminate
      fingerAngle = 0.15
      for i in range(1000):
        graspAction = [0, 0, 0.0005, 0, fingerAngle]
        self._tm700.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle - (0.3 / 100.)
        if (fingerAngle < 0):
          fingerAngle = 0

      for i in range(10000):
        graspAction = [0, 0, 0.001, 0, fingerAngle]
        self._tm700.applyAction(graspAction)
        p.stepSimulation()
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2] > 0.23):
          break
        state = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2] > 0.5):
          break

      self._observation = self.getExtendedObservation()
      return True
    return False

  def _reward(self):

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

    # print(closestPoints1[0][8])
    closestPoints = closestPoints1[0][8]
    numPt = len(closestPoints1)
    #print(numPt)
    if (numPt > 0):
      #print("reward:")
      # reward = -1./((1.-closestPoints1[0][8] * 100 + 1. -closestPoints2[0][8] * 100 )/2)
      reward = -((closestPoints1[0][8])**2 + (closestPoints2[0][8])**2 )*(1/0.17849278457978357)
      # reward = 1/((abs(closestPoints1[0][8])   + abs(closestPoints2[0][8])*10 )**2 / 2)
      # reward = 1/closestPoints1[0][8]+1/closestPoints2[0][8]
    if (blockPos[2] > 0.2):
      reward = reward + 1000
      print("successfully grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("reward")
      #print(reward)
    # print("reward")
    # print(reward)
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step


if __name__ == '__main__':

  p.connect(p.GUI, options="--opencl2")
  test = tm700_possensor_gym()
  test.reset()
  p.stepSimulation()
  while True:
      test.step_to_target_pose([0.4317596244807792, 0.1470447615125933, 0.30, 0, -np.pi, 0, 0], ts=1/240., return_camera=False)
      test.step_to_target_pose([0.4317596244807792, 0.1470447615125933, 0.30, 0, -np.pi, 0, 0.6], ts=1/10., return_camera=True)

