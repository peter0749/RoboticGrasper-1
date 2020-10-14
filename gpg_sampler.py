"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Classes for sampling grasps.
Author: Jeff Mahler
"""
"""
This is the PointnetGPD sampler with only GPG sampler for real-world grasping
The code is from the author's repo of PointnetGPD https://github.com/lianghongzhuo/PointNetGPD
We Used it for benchmark purpose only.
"""

from abc import ABCMeta, abstractmethod
import itertools
import logging
import numpy as np
# import os, IPython, sys
import random
import time
import scipy.stats as stats
import pcl

import scipy
from scipy.spatial.transform import Rotation as sciRotation
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

'''
try:
    from mayavi import mlab
except ImportError:
    mlab = None
    logger.warning('Do not have mayavi installed, please set the vis to False')
'''

mlab = None

class GraspSampler:
    """ Base class for various methods to sample a number of grasps on an object.
    Should not be instantiated directly.
    """
    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def sample_grasps(self, graspable, num_grasps_generate, vis, **kwargs):
        """
        Create a list of candidate grasps for a given object.
        Must be implemented for all grasp sampler classes.
        Parameters
        ---------
        graspable : :obj:`GraspableObject3D`
            object to sample grasps on
        num_grasps_generate : int
        vis : bool
        """
        grasp = []
        return grasp
        # pass

    def show_points(self, point, color='lb', scale_factor=.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = (1, 1, 1)
        if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
            point = point.reshape(3, )
            mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
        else:  # vis for multiple points
            mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)

    def show_line(self, un1, un2, color='g', scale_factor=0.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        else:
            color_f = (1, 1, 1)
        mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)

    def show_grasp_norm_oneside(self, grasp_bottom_center,
                                grasp_normal, grasp_axis, minor_pc, scale_factor=0.001):

        un2 = grasp_bottom_center
        self.show_points(grasp_bottom_center, color='g', scale_factor=scale_factor * 4)
        # self.show_points(un1, scale_factor=scale_factor * 4)
        # self.show_points(un3, scale_factor=scale_factor * 4)
        # self.show_points(un5, scale_factor=scale_factor * 4)
        # self.show_line(un1, un2, color='g', scale_factor=scale_factor)  # binormal/ major pc
        # self.show_line(un3, un4, color='b', scale_factor=scale_factor)  # minor pc
        # self.show_line(un5, un6, color='r', scale_factor=scale_factor)  # approach normal
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_axis[0], grasp_axis[1], grasp_axis[2],
                      scale_factor=.03, line_width=0.25, color=(0, 1, 0), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], minor_pc[0], minor_pc[1], minor_pc[2],
                      scale_factor=.03, line_width=0.1, color=(0, 0, 1), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_normal[0], grasp_normal[1], grasp_normal[2],
                      scale_factor=.03, line_width=0.05, color=(1, 0, 0), mode='arrow')

    def get_hand_points(self, grasp_bottom_center, approach_normal, binormal):
        hh = self.config['thickness_side'] #self.gripper.hand_height
        fw = self.config['thickness'] #self.gripper.finger_width
        hod = self.config['gripper_width'] + 2 * self.config['thickness'] #self.gripper.hand_outer_diameter
        hd = self.config['hand_height'] #self.gripper.hand_depth
        open_w = hod - fw * 2
        minor_pc = np.cross(approach_normal, binormal)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)
        p5_p6 = minor_pc * hh * 0.5 + grasp_bottom_center
        p7_p8 = -minor_pc * hh * 0.5 + grasp_bottom_center
        p5 = -binormal * open_w * 0.5 + p5_p6
        p6 = binormal * open_w * 0.5 + p5_p6
        p7 = binormal * open_w * 0.5 + p7_p8
        p8 = -binormal * open_w * 0.5 + p7_p8
        p1 = approach_normal * hd + p5
        p2 = approach_normal * hd + p6
        p3 = approach_normal * hd + p7
        p4 = approach_normal * hd + p8

        p9 = -binormal * fw + p1
        p10 = -binormal * fw + p4
        p11 = -binormal * fw + p5
        p12 = -binormal * fw + p8
        p13 = binormal * fw + p2
        p14 = binormal * fw + p3
        p15 = binormal * fw + p6
        p16 = binormal * fw + p7

        p17 = -approach_normal * hh + p11
        p18 = -approach_normal * hh + p15
        p19 = -approach_normal * hh + p16
        p20 = -approach_normal * hh + p12
        p = np.vstack([np.array([0, 0, 0]), p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                       p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
        return p

    def show_grasp_3d(self, hand_points, color=(0.003, 0.50196, 0.50196)):
        # for i in range(1, 21):
        #     self.show_points(p[i])
        if color == 'd':
            color = (0.003, 0.50196, 0.50196)
        triangles = [(9, 1, 4), (4, 9, 10), (4, 10, 8), (8, 10, 12), (1, 4, 8), (1, 5, 8),
                     (1, 5, 9), (5, 9, 11), (9, 10, 20), (9, 20, 17), (20, 17, 19), (17, 19, 18),
                     (14, 19, 18), (14, 18, 13), (3, 2, 13), (3, 13, 14), (3, 6, 7), (3, 6, 2),
                     (3, 14, 7), (14, 7, 16), (2, 13, 15), (2, 15, 6), (12, 20, 19), (12, 19, 16),
                     (15, 11, 17), (15, 17, 18), (6, 7, 8), (6, 8, 5)]
        mlab.triangular_mesh(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2],
                             triangles, color=color, opacity=0.5)

    def check_collision_square(self, grasp_bottom_center, approach_normal, binormal,
                               minor_pc, graspable, p, way, vis=False):
        approach_normal = approach_normal.reshape(1, 3)
        approach_normal = approach_normal / np.linalg.norm(approach_normal)
        binormal = binormal.reshape(1, 3)
        binormal = binormal / np.linalg.norm(binormal)
        minor_pc = minor_pc.reshape(1, 3)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)
        matrix = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
        grasp_matrix = matrix.T  # same as cal the inverse
        points = graspable
        points = points - grasp_bottom_center.reshape(1, 3)
        # points_g = points @ grasp_matrix
        tmp = np.dot(grasp_matrix, points.T)
        points_g = tmp.T
        if way == "p_open":
            s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
        elif way == "p_left":
            s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
        elif way == "p_right":
            s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
        elif way == "p_bottom":
            s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
        else:
            raise ValueError('No way!')
        a1 = s1[1] < points_g[:, 1]
        a2 = s2[1] > points_g[:, 1]
        a3 = s1[2] > points_g[:, 2]
        a4 = s4[2] < points_g[:, 2]
        a5 = s4[0] > points_g[:, 0]
        a6 = s8[0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True

        return has_p, points_in_area

    def show_all_grasps(self, all_points, grasps_for_show):

        for grasp_ in grasps_for_show:
            grasp_bottom_center = grasp_[4]  # new feature: ues the modified grasp bottom center
            approach_normal = grasp_[1]
            binormal = grasp_[2]
            hand_points = self.get_hand_points(grasp_bottom_center, approach_normal, binormal)
            self.show_grasp_3d(hand_points)
        # self.show_points(all_points)
        # mlab.show()

    def check_collide(self, grasp_bottom_center, approach_normal, binormal, minor_pc, graspable, hand_points):
        bottom_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                    binormal, minor_pc, graspable, hand_points, "p_bottom")
        if bottom_points[0]:
            return True

        left_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                  binormal, minor_pc, graspable, hand_points, "p_left")
        if left_points[0]:
            return True

        right_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                   binormal, minor_pc, graspable, hand_points, "p_right")
        if right_points[0]:
            return True

        return False


class GpgGraspSamplerPcl(GraspSampler):
    """
    Sample grasps by GPG with pcl directly.
    http://journals.sagepub.com/doi/10.1177/0278364917735594
    Code from https://github.com/lianghongzhuo/PointNetGPD
    """

    def sample_grasps(self, point_cloud,points_for_sample, all_normal, num_grasps=20, max_num_samples=200, show_final_grasp=False, num_dy=2, range_dtheta=45, time_limit=np.inf, safety_dis_above_table=0.003, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF
        Parameters
        ----------
        point_cloud :
        all_normal :
        num_grasps : int
            the number of grasps to generate
        show_final_grasp :
        max_num_samples :
        time_limit : Time limit to perform sampling
        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_dy': num_dy,   # number
            'dy_step': 0.005,
            'dtheta': 5,  # unit degree
            'range_dtheta': range_dtheta,
            'r_ball': 0.01,
            'approach_step': 0.005,
            'step_back': 0.01, # step back 1cm to avoid collision in real grasp
        }

        # get all surface points
        all_points = point_cloud.to_array()
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        hh = self.config['thickness_side'] #self.gripper.hand_height
        fw = self.config['thickness'] #self.gripper.finger_width
        hod = self.config['gripper_width'] + 2 * self.config['thickness'] #self.gripper.hand_outer_diameter
        hd = self.config['hand_height'] #self.gripper.hand_depth

        # get all grasps
        start_ts = time.time()
        while True:
            # begin of modification 5: Gaussian over height
            # we can use the top part of the point clouds to generate more sample points
            # min_height = min(all_points[:, 2])
            # max_height = max(all_points[:, 2])
            # selected_height = max_height - abs(np.random.normal(max_height, (max_height - min_height)/3)
            #                                    - max_height)
            # ind_10 = np.argsort(abs(all_points[:, 2] - selected_height))[:10]
            # ind = ind_10[np.random.choice(len(ind_10), 1)]
            # end of modification 5

            # for ros, we neded to judge if the robot is at HOME

            scipy.random.seed()  # important! without this, the worker will get a pseudo-random sequences.
            ind = np.random.choice(points_for_sample.shape[0], size=1, replace=False)
            selected_surface = points_for_sample[ind, :].reshape(3, )
            if show_final_grasp:
                mlab.points3d(selected_surface[0], selected_surface[1], selected_surface[2],
                              color=(1, 0, 0), scale_factor=0.005)

            r_ball = params['r_ball']  # FIXME: for some relative small obj, we need to use pre-defined radius

            M = np.zeros((3, 3))

            selected_surface_pc = pcl.PointCloud(selected_surface.reshape(1, 3))
            kd = point_cloud.make_kdtree_flann()
            kd_indices, sqr_distances = kd.radius_search_for_cloud(selected_surface_pc, r_ball, 100)
            for _ in range(len(kd_indices[0])):
                if sqr_distances[0, _] != 0:
                    # neighbor = point_cloud[kd_indices]
                    normal = all_normal[kd_indices[0, _]]
                    normal = normal.reshape(-1, 1)
                    if np.linalg.norm(normal) != 0:
                        normal /= np.linalg.norm(normal)
                    else:
                        normal = np.array([0, 0, 1], dtype=np.float32)
                    M += np.matmul(normal, normal.T)
            if sum(sum(M)) == 0:
                print("M matrix is empty as there is no point near the neighbour")
                print("Here is a bug, if points amount is too little it will keep trying and never go outside.")
                continue
            else:
                logger.info("Selected a good sample point.")

            eigval, eigvec = np.linalg.eig(M)  # compared computed normal
            minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)  # minor principal curvature !!! Here should use column!
            minor_pc /= np.linalg.norm(minor_pc)
            new_normal = eigvec[:, np.argmax(eigval)].reshape(3)  # estimated surface normal !!! Here should use column!
            new_normal /= np.linalg.norm(new_normal)
            major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
            if np.linalg.norm(major_pc) != 0:
                major_pc = major_pc / np.linalg.norm(major_pc)

            # Judge if the new_normal has the same direction with old_normal, here the correct
            # direction in modified meshpy is point outward.
            # But we also need to reverse the final normal vector as described in GPG
            if np.dot(all_normal[ind], new_normal) < 0:
                new_normal = -new_normal
                major_pc = -major_pc

            new_normal = -new_normal
            major_pc = -major_pc

            # Search rotation long z-axis (yaw)
            rotation_search_space = np.arange(-params['range_dtheta'],
                                        params['range_dtheta'] + 1,
                                        params['dtheta'])
            r_abs = np.abs(rotation_search_space)
            r_ind = np.argsort(r_abs)
            r_abs = r_abs[r_ind]
            rotation_search_space = rotation_search_space[r_ind]
            rotation_search_space[0] = 0 # sample on origin first
            if params['num_dy']>0:
                translation_search_space = np.arange(-params['num_dy'] * params['dy_step'],
                                            (params['num_dy'] + 1) * params['dy_step'],
                                            params['dy_step'])
            else:
                translation_search_space = np.array([0,], dtype=np.float32)
            np.random.shuffle(translation_search_space)
            rs_ts = list(itertools.product(rotation_search_space, translation_search_space))
            random.shuffle(rs_ts)

            approach_dist = hd  # use gripper depth
            num_approaches = int(approach_dist / params['approach_step'])

            for dtheta, dy in rs_ts:
                try:
                    rotation = sciRotation.from_quat([minor_pc[0], minor_pc[1], minor_pc[2], dtheta / 180 * np.pi]).as_dcm()
                except AttributeError:
                    rotation = sciRotation.from_quat([minor_pc[0], minor_pc[1], minor_pc[2], dtheta / 180 * np.pi]).as_matrix()
                # compute centers and axes
                tmp_major_pc = np.dot(rotation, major_pc) # Y
                tmp_grasp_normal = np.dot(rotation, new_normal) # X
                tmp_minor_pc = minor_pc # Z

                tmp_grasp_bottom_center = selected_surface + tmp_major_pc * dy
                # go back a bite after rotation dtheta and translation dy!
                tmp_grasp_bottom_center = 0.02 * (
                        -tmp_grasp_normal) + tmp_grasp_bottom_center

                open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                             tmp_major_pc, tmp_minor_pc, all_points,
                                                             hand_points, "p_open")
                bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                               tmp_major_pc, tmp_minor_pc, all_points,
                                                               hand_points,
                                                               "p_bottom")
                if open_points is True and bottom_points is False:

                    left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                 tmp_major_pc, tmp_minor_pc, all_points,
                                                                 hand_points,
                                                                 "p_left")
                    right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, tmp_minor_pc, all_points,
                                                                  hand_points,
                                                                  "p_right")

                    if left_points is False and right_points is False:
                        ptg = [tmp_grasp_bottom_center, tmp_grasp_normal, tmp_major_pc, tmp_minor_pc]

                        for approach_s in range(num_approaches):
                            tmp_grasp_bottom_center = ptg[1] * approach_s * params['approach_step'] + ptg[0]
                            tmp_grasp_normal = ptg[1]
                            tmp_major_pc = ptg[2]
                            minor_pc = ptg[3]
                            is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                            tmp_major_pc, minor_pc, point_cloud, hand_points)

                            if is_collide:
                                # if collide, go back one step to get a collision free hand position
                                tmp_grasp_bottom_center += (-tmp_grasp_normal) * (params['approach_step'] + params['step_back'])

                                # here we check if the gripper collide with the table.
                                hand_points_ = self.get_hand_points(tmp_grasp_bottom_center,
                                                                    tmp_grasp_normal,
                                                                    tmp_major_pc)[1:]
                                min_finger_end = hand_points_[:, 2].min()
                                min_finger_end_pos_ind = np.where(hand_points_[:, 2] == min_finger_end)[0][0]

                                # Lots of tricks: This section remove the grippers collided with table
                                if min_finger_end < safety_dis_above_table:
                                    min_finger_pos = hand_points_[min_finger_end_pos_ind]  # the lowest point in a gripper
                                    x = -min_finger_pos[2]*tmp_grasp_normal[0]/tmp_grasp_normal[2]+min_finger_pos[0]
                                    y = -min_finger_pos[2]*tmp_grasp_normal[1]/tmp_grasp_normal[2]+min_finger_pos[1]
                                    p_table = np.array([x, y, 0])  # the point that on the table
                                    dis_go_back = np.linalg.norm([min_finger_pos, p_table]) + safety_dis_above_table
                                    tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center-tmp_grasp_normal*dis_go_back
                                else:
                                    # if the grasp is not collide with the table, do not change the grasp
                                    tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center

                                # final check
                                _, open_points = self.check_collision_square(tmp_grasp_bottom_center_modify,
                                                                             tmp_grasp_normal,
                                                                             tmp_major_pc, minor_pc, all_points,
                                                                             hand_points, "p_open")
                                is_collide = self.check_collide(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, all_points, hand_points)
                                if (len(open_points) > 20) and not is_collide:
                                    processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                      tmp_major_pc, minor_pc,
                                                                      tmp_grasp_bottom_center_modify])
                                    break
                if len(processed_potential_grasp) >= num_grasps or sampled_surface_amount >= max_num_samples:
                    return processed_potential_grasp
            sampled_surface_amount += 1
            logger.info("processed_potential_grasp %d / %d", len(processed_potential_grasp), num_grasps)
            logger.info("current amount of sampled surface %d / %d", sampled_surface_amount, max_num_samples)
            print("The grasps number got by modified GPG: %d / %d"%(len(processed_potential_grasp), num_grasps))
            print("current amount of sampled surface: %d / %d"%(sampled_surface_amount, max_num_samples))
            iteration_ts = time.time()
            if iteration_ts-start_ts>time_limit:
                print("Time's up!")
                break
        return processed_potential_grasp
