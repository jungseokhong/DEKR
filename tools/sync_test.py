#!/usr/bin/env python3
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import rospy
import math
import message_filters
import numpy as np




class SyncTest:
    def __init__(self):
        rospy.init_node('sync_test')

        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.imageCallBack, queue_size=3, buff_size=2**24)
        self.image_sub = message_filters.Subscriber('/loco_cams/right/image_raw', Image)
        self.pose_sub = message_filters.Subscriber('/detection/output_image', Image)

        # ts = message_filters.TimeSynchronizer([self.image_sub, self.image_sub], 10)
        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.pose_sub], 10, 2075792840, allow_headerless=True)
        ts.registerCallback(self.callback)
        self.sync_pub = rospy.Publisher('/sync_image', Image, queue_size=10)
        # self.jds_pub = rospy.Publisher('/detection/jds', Float64MultiArray, queue_size=1)

        self.correctness = False

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Rospy shutting down.")

    def callback(self, img_topic, pose_topic):
        print("INSIDE CALLBACK")
        self.img_raw1 = self.bridge.imgmsg_to_cv2(img_topic, "bgr8")
        self.img_raw2 = self.bridge.imgmsg_to_cv2(pose_topic, "bgr8")
        vis = np.concatenate((self.img_raw1, self.img_raw2), axis=1)
        print(img_topic.header)

        msg_frame = CvBridge().cv2_to_imgmsg(vis, encoding="bgr8")
        self.sync_pub.publish(msg_frame)


if __name__ == '__main__':
    SyncTest()
    # img = cv2.imread('/home/irvlab/diver_id_ws2/src/diver_joint/scripts/DEKR/tools/img1.jpg')
    # image_process(img)