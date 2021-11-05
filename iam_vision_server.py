#!/usr/bin/env python

import os
import glob
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from iam_vision_msgs.msg import *
from iam_vision_msgs.srv import *

class IAMVisionServer:

    def __init__(self):
        self.bridge = CvBridge()
        self.service = rospy.Service('iam_vision_server', IAMVision, self.callback)
        self.iam_vision_path = os.path.dirname(os.path.realpath(__file__))
        self.unlabeled_image_path = self.iam_vision_path+'/images/unlabeled/'
        self.unlabeled_image_files = glob.glob(self.unlabeled_image_path+'*.png')
        if len(self.unlabeled_image_files) == 0:
            self.lowest_image_num = -1
            self.highest_image_num = -1
        else:
            image_nums = []
            for image_path in self.unlabeled_image_files:
                image_nums.append(int(image_path[image_path.rfind('_')+1:image_path.rfind('.')]))
            self.lowest_image_num = min(image_nums)
            self.highest_image_num = max(image_nums)

    def callback(self, req):

        if req.request_type == 0:
            try:
                image_msg = rospy.wait_for_message(req.camera_topic_name, Image, timeout=5)
                cv_image = self.bridge.imgmsg_to_cv2(image_msg)
                image_path = self.unlabeled_image_path+'image_'+str(self.highest_image_num+1)+'.png'

                if image_msg.encoding == 'bgra8':    
                    cv2.imwrite(image_path, cv_image)
                else:
                    cv2.imwrite(image_path, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
                
                self.highest_image_num += 1
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
                self.unlabeled_image_files = glob.glob(self.unlabeled_image_path+'*.png')

                if len(self.unlabeled_image_files) == 0:
                    self.lowest_image_num = -1
                    self.highest_image_num = -1
                    response_msg = IAMVisionResponse() 
                    response_msg.response_type = req.request_type
                    response_msg.request_success = False
                    return response_msg
                else:
                    image_nums = []
                    for image_path in self.unlabeled_image_files:
                        image_nums.append(int(image_path[image_path.rfind('_')+1:image_path.rfind('.')]))
                    self.lowest_image_num = min(image_nums)
                    self.highest_image_num = max(image_nums)

                    response_msg = IAMVisionResponse() 
                    response_msg.response_type = req.request_type
                    response_msg.request_success = True

                    image_path = self.unlabeled_image_path+'image_'+str(self.highest_image_num)+'.png'
                    cv_image = cv2.imread(image_path)

                    response_msg.image_path = image_path
                    response_msg.image = self.bridge.cv2_to_imgmsg(cv_image)
                    return response_msg
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
        elif req.request_type == 3:
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
    