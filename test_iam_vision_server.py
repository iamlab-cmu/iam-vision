#!/usr/bin/env python

import rospy
from iam_vision_msgs.srv import *

def IAMVisionClient():
    rospy.wait_for_service('iam_vision_server')
    try:
        iam_vision_client = rospy.ServiceProxy('iam_vision_server', IAMVision)
        request = IAMVisionRequest()
        request.request_type = 0
        request.image_topic_name = '/rgb/image_raw'
        
        resp = iam_vision_client(request)
        return resp.request_success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":

    IAMVisionClient()