#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import csv
import os
import shutil
import time
import sys
sys.path.append("../lib")

import cv2
import numpy as np
from PIL import Image
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
import models

from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import up_interpolate

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import rospy
import math
import message_filters

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

## using 5 - 14
COCO_KEYPOINT_INDEXES_SHORT = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'ls',
    6: 'rs',
    7: 'le',
    8: 're',
    9: 'lw',
    10: 'rw',
    11: 'lh',
    12: 'rh',
    13: 'lk',
    14: 'rk',
    15: 'left_ankle',
    16: 'right_ankle'
}

class_dict = {'CE':0, 'JH':1, 'DK':2, 'CO':3, 'JZ':4, 'CK':5, 'JS':6, 'MF': 7, 'SS': 8}

def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    # if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
    #     shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--visthre', type=float, default=0)
    parser.add_argument('--srcfolder', type=str, required=True)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args



## MODEL DEFINITION
# transformation
pose_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

# cudnn related setting
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

args = parse_args()
update_config(cfg, args)
pose_dir = prepare_output_dirs(args.outputDir)
csv_output_rows = []

pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=False
)

if cfg.TEST.MODEL_FILE:
    print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    pose_model.load_state_dict(torch.load(
        cfg.TEST.MODEL_FILE), strict=False)
else:
    raise ValueError('expected model defined in config at TEST.MODEL_FILE')

pose_model.to(CTX)
pose_model.eval()



class PosePredictor():
    def __init__(self):

        return 1



def get_pose_estimation_prediction(cfg, model, image, vis_thre, transforms):
    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(
        image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
    )

    with torch.no_grad():
        heatmap_sum = 0
        poses = []

        for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            heatmap, posemap = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap, posemap, scale
            )
        
        heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(cfg, heatmap_avg, poses)

        if len(scores) == 0:
            return []
        else:
            if cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

            final_poses = get_final_preds(
                poses, center, scale_resized, base_size
            )

        final_results = []
        for i in range(len(scores)):
            if scores[i] > vis_thre:
                final_results.append(final_poses[i])

        if len(final_results) == 0:
            return []

    return final_results



def dist(p,q):
    return math.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )


def image_process(img):

    count = 0

    total_now = time.time()
    # img = cv2.imread('test.img')
    # img = cv2.imread(os.path.join(src_folder,filename))

    frame_height, frame_width, _ = img.shape
    # if img is not None:
    image_in_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    count += 1
    image_pose = image_in_RGB.copy()

    # Clone 1 image for debugging purpose
    image_debug = img.copy()

    now = time.time()
    pose_preds = get_pose_estimation_prediction(
        cfg, pose_model, image_pose, args.visthre, transforms=pose_transform)
    then = time.time()
    if len(pose_preds) == 0:
        count += 1
        # continue
        print("no pose detected")
    print("Find person pose in: {} sec".format(then - now))                    

    new_csv_coord_row = []
    new_csv_row = [frame_width, frame_height]
    jds_list = []
    correctness = False
    for coords in pose_preds:
        # Draw each point on image
        [nose, left_eye, right_eye, left_ear, right_ear,
            left_shoulder, right_shoulder, left_elbow, right_elbow,
            left_wrist, right_wrist, left_hip, right_hip,
            left_knee, right_knee, left_ankle, right_ankle]=coords

        lh_lk_dist = dist(left_hip, left_knee)
        rh_rk_dist = dist(right_hip, right_knee)
        h_k_condition = (lh_lk_dist/rh_rk_dist < 1.2) and (lh_lk_dist/rh_rk_dist > 0.8)
        lw_le_dist = dist(left_wrist, left_elbow)
        rw_re_dist = dist(right_wrist, right_elbow)
        le_ls_dist = dist(left_elbow, left_shoulder)
        re_rs_dist = dist(right_elbow, right_shoulder)
        k_x_condition = abs(left_knee[0]-right_knee[0]) > 10
        w_e_condition = (lw_le_dist/rw_re_dist < 1.3) and (lw_le_dist/rw_re_dist > 0.6)
        arm_condition = (lw_le_dist/le_ls_dist < 1.3) and (lw_le_dist/le_ls_dist > 0.6) and (rw_re_dist/re_rs_dist < 1.3) and (rw_re_dist/re_rs_dist > 0.6)
        
        ls_rs_dist = dist(left_shoulder, right_shoulder)
        ls_lh_dist = dist(left_shoulder, left_hip)
        rs_rh_dist = dist(right_shoulder, right_hip)
        avg_s_h_dist = (ls_lh_dist + rs_rh_dist) / 2
        s_h_ratio = avg_s_h_dist / ls_rs_dist
        shoulder_condition = (s_h_ratio >= 1.3) #and (s_h_ratio <= 1.6)
        sh_hk_ratio_condition = (ls_lh_dist / lh_lk_dist) < 2 and (rs_rh_dist / rh_rk_dist) <2
        larm_sh_ratio_condition = (ls_lh_dist/lw_le_dist) < 3 and (rs_rh_dist/rw_re_dist) < 3
        uarm_sh_ratio_condition = (ls_lh_dist/le_ls_dist) < 3 and (rs_rh_dist/re_rs_dist) < 3
        lh_rh_dist = dist(left_hip, right_hip)
        should_hip_ratio_condition = (ls_rs_dist / lh_rh_dist) < 1.8 and (ls_rs_dist / lh_rh_dist) > 0.8


        # if filename[:-4] == 'img1311' or filename[:-4] == 'img1207':
        #     print(filename)
        #     print([left_shoulder[1] < left_hip[1], left_hip[1] < left_knee[1], 
        #                     lh_rh_dist > 10 > 10, h_k_condition, 
        #                     w_e_condition, arm_condition, k_x_condition, 
        #                     shoulder_condition, sh_hk_ratio_condition, (ls_rs_dist > 10),
        #                     larm_sh_ratio_condition, uarm_sh_ratio_condition])
        if ((left_shoulder[1] < left_hip[1]) and (left_hip[1] < left_knee[1]) and
                        lh_rh_dist > 10 and h_k_condition and 
                        w_e_condition and arm_condition and k_x_condition and 
                        shoulder_condition and sh_hk_ratio_condition and (ls_rs_dist > 10) and
                        larm_sh_ratio_condition and uarm_sh_ratio_condition
                        ):
            
            correctness = True
        else:
            correctness = False


        for coord_idx, coord in enumerate(coords):
            # print(COCO_KEYPOINT_INDEXES[coord_idx], coord_idx)
            if coord_idx > 4 and coord_idx < 15:
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(image_debug, (x_coord, y_coord), 4, (0, 255, 0), 2)
                cv2.putText(image_debug, COCO_KEYPOINT_INDEXES_SHORT[coord_idx], (x_coord+10, y_coord+10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)
                new_csv_coord_row.extend([x_coord, y_coord])
            
        
        # JOINT CONNECTIONS
        if left_shoulder.any() and right_shoulder.any(): #jd1
            cv2.line(image_debug, (left_shoulder[0], left_shoulder[1]), (right_shoulder[0], right_shoulder[1]), (100, 171, 231), 2)
            jds_list.append( dist(left_shoulder, right_shoulder) )
        if left_shoulder.any() and left_hip.any(): #jd2
            cv2.line(image_debug, (left_shoulder[0], left_shoulder[1]), (left_hip[0], left_hip[1]), (100, 171, 231), 2)
            jds_list.append( dist(left_shoulder, left_hip) )
        if left_shoulder.any() and left_elbow.any(): #jd3
            cv2.line(image_debug, (left_shoulder[0], left_shoulder[1]), (left_elbow[0], left_elbow[1]), (100, 171, 231), 2)
            jds_list.append( dist(left_shoulder, left_elbow) )
        if left_elbow.any() and left_wrist.any(): #jd4
            cv2.line(image_debug, (left_elbow[0], left_elbow[1]), (left_wrist[0], left_wrist[1]), (100, 171, 231), 2)
            jds_list.append( dist(left_elbow, left_wrist) )
        if right_hip.any() and right_shoulder.any(): #jd5
            cv2.line(image_debug, (right_shoulder[0], right_shoulder[1]), (right_hip[0], right_hip[1]), (100, 171, 231), 2)
            jds_list.append( dist(right_hip, right_shoulder) )
        if right_elbow.any() and right_shoulder.any(): #jd6
            cv2.line(image_debug, (right_shoulder[0], right_shoulder[1]), (right_elbow[0], right_elbow[1]), (100, 171, 231), 2)
            jds_list.append( dist(right_elbow, right_shoulder) )
        if right_wrist.any() and right_elbow.any(): #jd7
            cv2.line(image_debug, (right_elbow[0], right_elbow[1]), (right_wrist[0], right_wrist[1]), (100, 171, 231), 2)
            jds_list.append( dist(right_wrist, right_elbow) )
        if right_hip.any() and left_hip.any():   #jd8
            cv2.line(image_debug, (left_hip[0], left_hip[1]), (right_hip[0], right_hip[1]), (100, 171, 231), 2)
            jds_list.append( dist(left_hip, right_hip) )
        if left_hip.any() and left_knee.any(): #jd9
            cv2.line(image_debug, (left_hip[0], left_hip[1]), (left_knee[0], left_knee[1]), (100, 171, 231), 2)
            jds_list.append( dist(left_hip, left_knee) )
        if right_hip.any() and right_knee.any():  #jd10
            cv2.line(image_debug, (right_hip[0], right_hip[1]), (right_knee[0], right_knee[1]), (100, 171, 231), 2)
            jds_list.append( dist(right_hip, right_knee) )


        # total_then = time.time()
        # text = "{:03.2f} sec".format(total_then - total_now)
        # cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 0, 255), 2, cv2.LINE_AA)
        if correctness:
            new_csv_row = new_csv_row + jds_list + new_csv_coord_row

            # csv_output_rows.append(new_csv_row)
            # return image_debug, jds_list, csv_output_rows, correctness
        # save output with each folder
        # cv2.imshow('test', image_debug)
        # cv2.waitKey(0)
        # and finally destroy/close all open windows
        # cv2.destroyAllWindows()

        ##TODO: check this part later
        # isExist = os.path.exists(os.path.join(pose_dir, folder+'_processed'))
        # if not isExist:
        #     # Create a new directory because it does not exist
        #     os.makedirs(os.path.join(pose_dir, folder+'_processed'))

        # isExist2 = os.path.exists(os.path.join(pose_dir, folder+'_failed'))
        # if not isExist2:
        #     # Create a new directory because it does not exist
        #     os.makedirs(os.path.join(pose_dir, folder+'_failed'))
        # if correctness:
        #     img_file = os.path.join(pose_dir, folder+'_processed', 'processed_'+filename)
        # else:
        #     img_file = os.path.join(pose_dir, folder+'_failed', 'failed_'+filename)
        # cv2.imwrite(img_file, image_debug)

    return image_debug, jds_list, new_csv_row, correctness, pose_preds
    # # write csv
    # csv_headers = [ 'w','h', 'jd1', 'jd2','jd3','jd4','jd5','jd6','jd7','jd8','jd9','jd10']
    # if cfg.DATASET.DATASET_TEST == 'coco':
    #     for keypoint in COCO_KEYPOINT_INDEXES.values():
    #         csv_headers.extend([keypoint+'_x', keypoint+'_y'])
    # else:
    #     raise ValueError('Please implement keypoint_index for new dataset: %s.' % cfg.DATASET.DATASET_TEST)

    # csv_output_filename = os.path.join(pose_dir, '_processed'+'.csv') ## TODO: check here
    # with open(csv_output_filename, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(csv_headers)
    #     csvwriter.writerows(csv_output_rows)


class JointAnalysis:
    def __init__(self):
        rospy.init_node('diver_joint_analyzer')
        self.img_counter = 0
        # self.rate = rospy.Rate(50)
        self.jds_msg = Float64MultiArray()
        self.two_joints_msg = Float64MultiArray()
        
        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.imageCallBack, queue_size=3, buff_size=2**24)
        self.image_sub = rospy.Subscriber('/loco_cams/right/image_raw', Image, self.imageCallBack, queue_size=1, buff_size=2**24)
        self.state_sub = rospy.Subscriber('/adroc_diver/state', String, self.stateCallBack, queue_size=1, buff_size=2**24)
        self.label_sub = rospy.Subscriber('/adroc_diver/label', String, self.labelCallBack, queue_size=1, buff_size=2**24)
        self.target_sub = rospy.Subscriber('/adroc_diver/target', String, self.targetCallBack, queue_size=1, buff_size=2**24)
        self.predict_sub = rospy.Subscriber('/adroc_diver/predict', String, self.predictCallBack, queue_size=1, buff_size=2**24)
        self.image_pub = rospy.Publisher('/detection/output_image', Image, queue_size=1)
        self.jds_pub = rospy.Publisher('/detection/jds', Float64MultiArray, queue_size=1)
        self.two_joints_pub = rospy.Publisher('/detected_poses_keypoints', Float64MultiArray, queue_size=1)

        self.correctness = False
        self.state = ""
        self.label = ""
        self.target = ""
        self.predict = ""

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Rospy shutting down.")

    def stateCallBack(self, state_topic):
        self.state = state_topic.data
    
    def labelCallBack(self, label_topic):
        self.label = label_topic.data

    def targetCallBack(self, target_topic):
        self.target = target_topic.data

    def predictCallBack(self, predict_topic):
        self.predict = predict_topic.data

    def imageCallBack(self, img_topic):
        # print("INSIDE CALLBACK")
        try:
            self.img_raw = self.bridge.imgmsg_to_cv2(img_topic, "bgr8")
            # print(self.img_raw)
        except CvBridgeError as e:
            print(e)
            
        if self.img_raw is None:
            print('frame dropped, skipping tracking')
        else:
            ## TODO save csv_output_rows for training?
            image_debug, jds_list, csv_output_rows, correctness, pose_pred = image_process(self.img_raw)
            # print(jds_list)

            # [nose, left_eye, right_eye, left_ear, right_ear,
            # left_shoulder, right_shoulder, left_elbow, right_elbow,
            # left_wrist, right_wrist, left_hip, right_hip,
            # left_knee, right_knee, left_ankle, right_ankle]
            # print(pose_pred)

            frame_height, frame_width, _ = self.img_raw.shape


            self.jds_msg.data = jds_list
            
            if self.state:
                cv2.putText(self.img_raw, "Current State: "+self.state, (frame_width-550, frame_height-30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 191, 255), 2, cv2.LINE_AA)
                cv2.putText(image_debug, "Current State: "+self.state, (frame_width-550, frame_height-30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 191, 255), 2, cv2.LINE_AA)

            if self.label:
                cv2.putText(image_debug, "Assigned Label: "+self.label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2, cv2.LINE_AA)
                self.label = ""
            
            if self.target:
                cv2.putText(image_debug, "Target Diver: "+self.target, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2, cv2.LINE_AA)
                # self.target = ""
            
            if self.predict:
                cv2.putText(image_debug, "Predicted Diver: "+self.predict, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2, cv2.LINE_AA)
                # self.predict = ""
            
            # Added for distance control
            self.two_joints_msg.data = [pose_pred[0][5,0],pose_pred[0][5,1],pose_pred[0][6,0], pose_pred[0][6,1],
                    pose_pred[0][11,0], pose_pred[0][11,1], pose_pred[0][12,0], pose_pred[0][12,1]]
            self.two_joints_pub.publish(self.two_joints_msg)
            
            if correctness:
                # img = cv2.cvtColor(image_debug, cv2.COLOR_RGB2BGR)

                msg_frame = CvBridge().cv2_to_imgmsg(image_debug, encoding="bgr8")
                self.image_pub.publish(msg_frame)
                self.jds_pub.publish(self.jds_msg)
                self.img_counter += 1
            else:
                msg_frame = CvBridge().cv2_to_imgmsg(self.img_raw, encoding="bgr8")
                self.image_pub.publish(msg_frame)


if __name__ == '__main__':
    JointAnalysis()
    # img = cv2.imread('/home/irvlab/diver_id_ws2/src/diver_joint/scripts/DEKR/tools/img1.jpg')
    # image_process(img)