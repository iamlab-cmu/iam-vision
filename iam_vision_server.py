#!/usr/bin/env python

import os
import glob
import sys
import rospy
import cv2
import string
import json
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from iam_vision_msgs.srv import *
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from geometry_msgs.msg import Point as Point3D

class IAMVisionServer:

    def __init__(self):
        self.azure_kinect_intrinsics = CameraIntrinsics.load('../camera-calibration/calib/azure_kinect.intr')
        self.azure_kinect_extrinsics = RigidTransform.load('../camera-calibration/calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf')

        self.bridge = CvBridge()
        self.service = rospy.Service('iam_vision_server', IAMVision, self.callback)
        self.iam_vision_path = os.path.dirname(os.path.realpath(__file__))
        self.object_type_path = self.iam_vision_path+'/images/object_types.txt'
        self.object_types = []
        self.unlabeled_image_path = self.iam_vision_path+'/images/unlabeled/'
        self.labeled_image_path = self.iam_vision_path+'/images/labeled/'
        self.parse_object_types()
        self.get_lowest_and_highest_unlabeled_image_nums()
        self.get_lowest_and_highest_labeled_image_nums()

    def parse_object_types(self):
        self.object_types = []
        with open(self.object_type_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                self.object_types.append(line.strip())

    def add_new_object_types(self, object_types):
        for object_type in object_types:
            if object_type not in self.object_types:
                self.object_types.append(object_type)

        self.object_types.sort()
        os.remove(self.object_type_path) 

        with open(self.object_type_path, 'w+') as f:
            for object_type in self.object_types:
                f.write(object_type+'\n')

    def get_lowest_and_highest_unlabeled_image_nums(self):
        self.unlabeled_image_files = glob.glob(self.unlabeled_image_path+'*.png')
        if len(self.unlabeled_image_files) == 0:
            self.lowest_unlabeled_image_num = -1
            self.highest_unlabeled_image_num = -1
        else:
            image_nums = []
            for image_path in self.unlabeled_image_files:
                image_nums.append(int(image_path[image_path.rfind('_')+1:image_path.rfind('.')]))
            self.lowest_unlabeled_image_num = min(image_nums)
            self.highest_unlabeled_image_num = max(image_nums)

    def get_lowest_and_highest_labeled_image_nums(self):
        self.labeled_image_files = glob.glob(self.labeled_image_path+'*.png')

        if len(self.labeled_image_files) == 0:
            self.lowest_labeled_image_num = -1
            self.highest_labeled_image_num = -1
        else:
            image_nums = []
            for image_path in self.labeled_image_files:
                if 'mask' not in image_path:
                    image_nums.append(int(image_path[image_path.rfind('_')+1:image_path.rfind('.')]))
            self.lowest_labeled_image_num = min(image_nums)
            self.highest_labeled_image_num = max(image_nums)

    def callback(self, req):

        ## Save RGB Camera Image
        if req.request_type == 0:
            try:
                image_msg = rospy.wait_for_message(req.camera_topic_name, Image, timeout=5)
                cv_image = self.bridge.imgmsg_to_cv2(image_msg)
                self.get_lowest_and_highest_unlabeled_image_nums()
                rgb_image_path = self.unlabeled_image_path+'image_'+str(self.highest_unlabeled_image_num+1)+'.png'
                if image_msg.encoding == 'bgra8':    
                    cv2.imwrite(rgb_image_path, cv_image)
                else:
                    cv2.imwrite(rgb_image_path, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
                
                self.highest_unlabeled_image_num += 1
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = True
                response_msg.rgb_image_path = rgb_image_path
                return response_msg
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
                return response_msg
        ## Save Depth Camera Image
        elif req.request_type == 1:
            try:
                image_msg = rospy.wait_for_message(req.camera_topic_name, Image, timeout=5)
                cv_image = self.bridge.imgmsg_to_cv2(image_msg)

                cv2.imwrite(req.depth_image_path, cv_image)

                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = True
                response_msg.depth_image_path = req.depth_image_path
                return response_msg
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
                return response_msg
        ## Get Latest RGB Image in Unlabeled
        elif req.request_type == 2:
            if len(req.rgb_image_path) > 0:
                try:
                    cv_image = cv2.imread(req.rgb_image_path)
                    response_msg = IAMVisionResponse() 
                    response_msg.response_type = req.request_type
                    response_msg.request_success = True

                    response_msg.rgb_image_path = req.rgb_image_path
                    response_msg.image = self.bridge.cv2_to_imgmsg(cv_image)

                    return response_msg
                except:
                    response_msg = IAMVisionResponse() 
                    response_msg.response_type = req.request_type
                    response_msg.request_success = False
                    return response_msg
            else:
                self.get_lowest_and_highest_unlabeled_image_nums()

                if self.highest_unlabeled_image_num == -1:
                    response_msg = IAMVisionResponse() 
                    response_msg.response_type = req.request_type
                    response_msg.request_success = False
                    return response_msg
                else:
                    try:
                        rgb_image_path = self.unlabeled_image_path+'image_'+str(self.highest_unlabeled_image_num)+'.png'
                        cv_image = cv2.imread(rgb_image_path)

                        response_msg = IAMVisionResponse() 
                        response_msg.response_type = req.request_type
                        response_msg.request_success = True
                        response_msg.rgb_image_path = rgb_image_path
                        response_msg.image = self.bridge.cv2_to_imgmsg(cv_image)
                        return response_msg
                    except:
                        response_msg = IAMVisionResponse() 
                        response_msg.response_type = req.request_type
                        response_msg.request_success = False
                        return response_msg
        ## Save Masks and Labeled Image Info
        elif req.request_type == 3:
            try:
                new_image_path = self.labeled_image_path+'image_'+str(self.highest_labeled_image_num+1)+'.png'
                os.rename(req.rgb_image_path, new_image_path)
                new_mask_image_path = self.labeled_image_path+'image_'+str(self.highest_labeled_image_num+1)+'_mask.png'
                mask_image = self.bridge.imgmsg_to_cv2(req.masks)
                cv2.imwrite(new_mask_image_path, mask_image)

                image_info_path = self.labeled_image_path+'image_'+str(self.highest_labeled_image_num+1)+'.json'
                image_info = {}
                image_info['file_name'] = new_image_path
                image_info['mask_file_name'] = new_mask_image_path
                image_info['height'] = mask_image.shape[0]
                image_info['width'] = mask_image.shape[1]
                image_info['image_id'] = self.highest_labeled_image_num+1

                capitalized_object_names = []
                for object_name in req.object_names:
                    capitalized_object_names.append(string.capwords(object_name))

                image_info['object_names'] = capitalized_object_names
                image_info['bounding_boxes'] = [[bounding_box.min_x, bounding_box.min_y, bounding_box.max_x, bounding_box.max_y] for bounding_box in req.bounding_boxes]

                self.add_new_object_types(capitalized_object_names)

                with open(image_info_path, 'w') as fp:
                    json.dump(image_info, fp)
    
                self.highest_labeled_image_num += 1

                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.rgb_image_path = new_image_path
                response_msg.request_success = True
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
        ## Get Positions in Robot Coordinates
        elif req.request_type == 4:
            try:
                depth_image = cv2.imread(req.depth_image_path)
                    
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type

                for point in req.points:
                    point_center = Point(np.array([point.x, point.y]), 'azure_kinect_overhead')
                    point_depth = depth_image[point.y, point.x]
                    world_point = self.azure_kinect_extrinsics * self.azure_kinect_intrinsics.deproject_pixel(point_depth, point_center)    
                    response_msg.points.append(Point3D(world_point.x, world_point.y, world_point.z))
                
                response_msg.request_success = True

                return response_msg
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
                return response_msg
        ## Save RGB Image
        elif req.request_type == 5:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(req.image)
                cv2.imwrite(req.rgb_image_path, cv_image)

                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.rgb_image_path = req.rgb_image_path
                response_msg.request_success = True
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
        ## Save Depth Image
        elif req.request_type == 6:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(req.image)
                cv2.imwrite(req.depth_image_path, cv_image)

                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.depth_image_path = req.depth_image_path
                response_msg.request_success = True
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
        ## Get Depth Image
        elif req.request_type == 7:
            try:
                cv_image = cv2.imread(req.depth_image_path)
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = True

                response_msg.depth_image_path = req.depth_image_path
                response_msg.image = self.bridge.cv2_to_imgmsg(cv_image)
                return response_msg
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
                return response_msg

        return response_msg
     

def main(args):
    iam_vision = IAMVisionServer()
    print("IAM Vision Server Ready.")

    rospy.init_node('iam_vision_server', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
    
