#!/usr/bin/env python

import os
import glob
import sys
import rospy
import cv2
import string
import json
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from iam_vision_msgs.srv import *

class IAMVisionServer:

    def __init__(self):
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
                image_nums.append(int(image_path[image_path.rfind('_')+1:image_path.rfind('.')]))
            self.lowest_labeled_image_num = min(image_nums)
            self.highest_labeled_image_num = max(image_nums)

    def callback(self, req):

        ## Save Camera Image
        if req.request_type == 0:
            try:
                image_msg = rospy.wait_for_message(req.camera_topic_name, Image, timeout=5)
                cv_image = self.bridge.imgmsg_to_cv2(image_msg)
                image_path = self.unlabeled_image_path+'image_'+str(self.highest_unlabeled_image_num+1)+'.png'

                if image_msg.encoding == 'bgra8':    
                    cv2.imwrite(image_path, cv_image)
                else:
                    cv2.imwrite(image_path, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
                
                self.highest_unlabeled_image_num += 1
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = True
                response_msg.image_path = image_path
                return response_msg
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
                return response_msg
        ## Get Latest Image in Unlabeled
        elif req.request_type == 1:
            if len(req.image_path) > 0:
                try:
                    cv_image = cv2.imread(req.image_path)
                    response_msg = IAMVisionResponse() 
                    response_msg.response_type = req.request_type
                    response_msg.request_success = True

                    response_msg.image_path = req.image_path
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
                        image_path = self.unlabeled_image_path+'image_'+str(self.highest_unlabeled_image_num)+'.png'
                        cv_image = cv2.imread(image_path)

                        response_msg = IAMVisionResponse() 
                        response_msg.response_type = req.request_type
                        response_msg.request_success = True
                        response_msg.image_path = image_path
                        response_msg.image = self.bridge.cv2_to_imgmsg(cv_image)
                        return response_msg
                    except:
                        response_msg = IAMVisionResponse() 
                        response_msg.response_type = req.request_type
                        response_msg.request_success = False
                        return response_msg
        ## Save Image
        elif req.request_type == 2:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(req.image)
                cv2.imwrite(req.image_path, cv_image)

                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.image_path = req.image_path
                response_msg.request_success = True
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
        ## Save Masks and Labeled Image Info
        elif req.request_type == 3:
            try:
                new_image_path = self.labeled_image_path+'image_'+str(self.highest_labeled_image_num+1)+'.png'
                os.rename(req.image_path, new_image_path)
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
                response_msg.image_path = new_image_path
                response_msg.request_success = True
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
        ## Get Positions in Robot Coordinates
        elif req.request_type == 4:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(req.image)
                cv2.imwrite(req.image_path, cv_image)

                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.image_path = req.image_path
                response_msg.request_success = True
            except:
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False

        
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
    