#!/usr/bin/env python

import rospy
from iam_vision_msgs.srv import *
from web_interface_msgs.msg import Request as webrequest
from bokeh_server_msgs.msg import Request as bokehrequest

def IAMVisionClient():
    rospy.wait_for_service('iam_vision_server')
    try:
        iam_vision_client = rospy.ServiceProxy('iam_vision_server', IAMVision)
        request = IAMVisionRequest()
        request.request_type = 0
        request.image_topic_name = '/usb_cam/image_raw'
        resp = iam_vision_client(request)

        print(resp.request_success)

        request = IAMVisionRequest()
        request.request_type = 1
        resp = iam_vision_client(request)

        print(resp.request_success)

        web_pub = rospy.Publisher('/human_interface_request', webrequest, queue_size=1)
        bokeh_pub = rospy.Publisher('/bokeh_request', bokehrequest, queue_size=1)
        
        web_request_msg = webrequest()
        web_request_msg.instruction_text = 'Label the image below by tapping on 4 extreme points. You can press on a point again to remove it. Then type in the name of the object and press submit. Press Done when you have finished labeling.'
        web_request_msg.display_type = 3
        web_pub.publish(web_request_msg)
        rospy.sleep(1)

        bokeh_request_msg = bokehrequest()
        bokeh_request_msg.display_type = 1
        bokeh_request_msg.image = resp.image
        bokeh_pub.publish(bokeh_request_msg)
        rospy.sleep(1)

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    rospy.init_node('iam_vision_server_test', anonymous=True)

    try:
        IAMVisionClient()
    except rospy.ROSInterruptException:
        pass