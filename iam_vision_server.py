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
            image_msg = rospy.wait_for_message(req.image_topic_name, Image, timeout=5)

            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg)
                cv2.imwrite(self.unlabeled_image_path+'image_'+str(self.highest_image_num+1)+'.png', cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
                self.highest_image_num += 1
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = True
                return response_msg
            except CvBridgeError as e:
                print(e)
                response_msg = IAMVisionResponse() 
                response_msg.response_type = req.request_type
                response_msg.request_success = False
                return response_msg
        elif req.request_type == 1:
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


                image_path = self.unlabeled_image_path+'image_'+str(self.lowest_image_num)+'.png'
                cv_image = cv2.imread(image_path)
                self.lowest_image_num += 1

                response_msg.image_path = image_path
                response_msg.image = self.bridge.cv2_to_imgmsg(cv_image)
                return response_msg
        elif req.request_type == 2:
            response_msg = IAMVisionResponse() 
            response_msg.response_type = req.request_type
            response_msg.request_success = True

            image_path = req.image_path
        
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
    