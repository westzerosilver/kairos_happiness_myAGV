#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from std_msgs.msg import String  # Import String type
from geometry_msgs.msg import Twist  # Import Twist type for velocity command
import time
import threading

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.avoid_flag=False
        self.image_sub = rospy.Subscriber('/agv_camera_info/compressed', CompressedImage, self.callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)  # Use Twist type for velocity command
        self.found_person = False
        #self.detect_object() = False
        self.full_body_cascade = cv2.CascadeClassifier('/home/uk/catkin_ws/src/agv_control/scripts/haarcascade_fullbody.xml')
        self.detector = cv2.SIFT_create()  # SIFT detector 사용
        # 특정 이미지에서 특징점과 디스크립터 추출
        self.keypoints_image, self.des1 = self.extract_keypoints_descriptors_from_image()
        
        self.avoid_thread=threading.Thread(target=self.avoid_object,daemon=True)
        self.avoid_thread.start()

    def find_dual_lines_center(self, contours):
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centers.append((cx, cy))
        return centers
    
    def calculate_steering(self, centers, camera_width=190):
        center_threshold = 10 # 카메라 중앙과 중심점의 허용 오차
        # Sort lane centers based on x-coordinate
        sorted_centers = sorted(centers, key=lambda x: x[0])
        mid_index = len(sorted_centers) // 2
        mid_point = sorted_centers[mid_index][0]
        if mid_point < camera_width // 2 + center_threshold :
            return "right"
        elif mid_point > camera_width // 2 - center_threshold :
            return "left"
        else:
            return "straight"

################ tree camera match ##########
    def extract_keypoints_descriptors_from_image(self):
        img_tree = cv2.imread("/home/uk/catkin_ws/src/agv_control/scripts/tree.jpg")
        gray = cv2.cvtColor(img_tree, cv2.COLOR_BGR2GRAY)
        keypoints_image, des = self.detector.detectAndCompute(gray, None)
        return keypoints_image, des
    
    def detect_object(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints_frame, des2 = self.detector.detectAndCompute(gray_frame, None)
        # 특징점 매칭
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des1, des2, k=2)
        # 좋은 매칭점 찾기
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
        #print(len(good_matches))
        if len(good_matches) > 30:  # 매칭점이 일정 개수 이상이면 객체가 감지된 것으로 판단
            return True
        else:
            return False
#####################################################
    def avoid_object(self):
        while True:
            if self.avoid_flag:
                print("장애물을 발견했습니다. 회피 동작을 수행합니다.")
                twist_cmd = Twist()
                twist_cmd.linear.x = 0.0  # 필요한 경우 선속도 조절
                twist_cmd.angular.z = 0.0  # 직진 운동을 위해 각속도를 0으로 설정
                self.cmd_vel_pub.publish(twist_cmd)
                rospy.sleep(1)
                twist_cmd.linear.y = 0.1
                self.cmd_vel_pub.publish(twist_cmd)
                rospy.sleep(3)
                twist_cmd.linear.y = 0.0
                twist_cmd.linear.x = 0.2
                self.cmd_vel_pub.publish(twist_cmd)
                rospy.sleep(3)
                twist_cmd.linear.x = 0.0
                twist_cmd.linear.y = -0.1
                self.cmd_vel_pub.publish(twist_cmd)
                rospy.sleep(3)
                twist_cmd.linear.y = 0.0
                self.cmd_vel_pub.publish(twist_cmd)
                self.avoid_flag=False

    def callback(self, data):
        try:
            # Convert the compressed image message to OpenCV format
            img = self.bridge.compressed_imgmsg_to_cv2(data)
            ############## lane, traffic light #################
            cropped_img_y = img[240:,105:295]
            cropped_img_r = img[150:,:]
            # HSV 색공간으로 변환
            hsv_y = cv2.cvtColor(cropped_img_y, cv2.COLOR_BGR2HSV)
            hsv_r = cv2.cvtColor(cropped_img_r, cv2.COLOR_BGR2HSV)
            # 노란색을 위한 범위 설정
            lower_yellow = np.array([23, 85, 115])
            upper_yellow = np.array([42, 211, 216])
            lower_red = np.array([160, 91, 0])
            upper_red = np.array([207, 255, 255])
            # 노란색 필터링
            mask_yellow = cv2.inRange(hsv_y, lower_yellow, upper_yellow)
            mask_red = cv2.inRange(hsv_r, lower_red, upper_red)
            # 노란색을 제외한 나머지를 검은색으로 채우기
            mask_inv_y = cv2.bitwise_not(mask_yellow)
            mask_inv_r = cv2.bitwise_not(mask_red)
            # 잡음 제거
            mask_inv_y = cv2.erode(mask_inv_y, None, iterations=2)
            mask_inv_y = cv2.dilate(mask_inv_y, None, iterations=2)
            mask_inv_r = cv2.erode(mask_inv_r, None, iterations=2)
            mask_inv_r = cv2.dilate(mask_inv_r, None, iterations=2)
            blur_y = cv2.GaussianBlur(mask_inv_y, (5, 5), 0)
            blur_r = cv2.GaussianBlur(mask_inv_r, (5, 5), 0)
            mask_y = cv2.erode(blur_y, None, iterations=2)
            mask_y = cv2.dilate(mask_y, None, iterations=2)
            mask_r = cv2.erode(blur_r, None, iterations=2)
            mask_r = cv2.dilate(mask_r, None, iterations=2)
            contours_y, _ = cv2.findContours(mask_y.copy(), 1, cv2.CHAIN_APPROX_NONE)
            contours_r, _ = cv2.findContours(mask_r.copy(), 1, cv2.CHAIN_APPROX_NONE)
            max_area_y = 6000
            min_area_y = 1
            # max_area_r = 1345
            # min_area_r = 1280         
            # Filtering contours based on area
            filtered_contours_y = [cnt for cnt in contours_y if cv2.contourArea(cnt) > min_area_y and cv2.contourArea(cnt) < max_area_y]
            #filtered_contours_r = [cnt for cnt in contours_r if cv2.contourArea(cnt) > min_area_r and cv2.contourArea(cnt) < max_area_r]

            cv2.drawContours(img, filtered_contours_y, -1, (0,0,255), 1)

            # total red pixels 
            total_contour_pixels = 0
            for contour in contours_r:
                contour_length = len(contour)
                total_contour_pixels += contour_length
            # 결과 출력
            #print("모든 윤곽선의 픽셀 수 합:", total_contour_pixels)
                            
            ##########################################################

            #################### person #############################
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bodies = self.full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # 사람 감지된 부분에 사각형 그리기
            for (x, y, w, h) in bodies:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # 사람의 너비와 높이가 일정 값보다 클 경우 제어 명령 수행
                if w > 90 and h > 180:
                    self.found_person = True
                    break  # 하나의 사람이라도 발견되면 종료
            ##########################################################
            if self.avoid_flag:
                pass
            
            elif self.detect_object(img):
                self.avoid_flag=True
                    
                # print("장애물을 발견했습니다. 회피 동작을 수행합니다.")
                # twist_cmd = Twist()
                # twist_cmd.linear.x = 0.0  # 필요한 경우 선속도 조절
                # twist_cmd.angular.z = 0.0  # 직진 운동을 위해 각속도를 0으로 설정
                # self.cmd_vel_pub.publish(twist_cmd)
                # rospy.sleep(1)
                # twist_cmd.linear.y = 0.1
                # self.cmd_vel_pub.publish(twist_cmd)
                # rospy.sleep(3)
                # twist_cmd.linear.y = 0.0
                # twist_cmd.linear.x = 0.2
                # self.cmd_vel_pub.publish(twist_cmd)
                # rospy.sleep(3)
                # twist_cmd.linear.x = 0.0
                # twist_cmd.linear.y = -0.1
                # self.cmd_vel_pub.publish(twist_cmd)
                # rospy.sleep(3)
                # twist_cmd.linear.y = 0.0
                # self.cmd_vel_pub.publish(twist_cmd)
                # #self.detect_object() = False
                # # rospy.sleep(3)
                
                


            # 사람이 발견되었을 경우 제어 명령 수행
            elif self.found_person:
                print("전방에 사람을 감지하였습니다. 정지하겠습니다.")
                twist_cmd = Twist()
                twist_cmd.linear.x = 0.0  # 필요한 경우 선속도 조절
                twist_cmd.angular.z = 0.0  # 직진 운동을 위해 각속도를 0으로 설정
                self.cmd_vel_pub.publish(twist_cmd)
                self.found_person = False      
            elif total_contour_pixels >= 1200:
                print("신호등이 빨간 불 입니다. 정지하겠습니다.")
                twist_cmd = Twist()
                twist_cmd.linear.x = 0.0  # Adjust linear velocity as needed
                twist_cmd.angular.z = 0.0  # No angular velocity for straight movement
                self.cmd_vel_pub.publish(twist_cmd)
            else:
                if len(filtered_contours_y) == 0:
                    # Publish linear velocity for straight movement
                    twist_cmd = Twist()
                    twist_cmd.linear.x = 0.05  # Adjust linear velocity as needed
                    twist_cmd.angular.z = 0.0  # No angular velocity for straight movement
                    self.cmd_vel_pub.publish(twist_cmd)
                else:
                    # Finding centers of dual lines
                    centers = self.find_dual_lines_center(filtered_contours_y)
                    # Calculating steering direction
                    steering_direction = self.calculate_steering(centers)
                    
                    # Publish angular velocity for turning
                    twist_cmd = Twist()
                    twist_cmd.linear.x = 0.0  # No linear velocity for turning
                    if steering_direction == "right":
                        twist_cmd.angular.z = -0.5  # Adjust angular velocity as needed
                        print("right")
                    elif steering_direction == "left":
                        twist_cmd.angular.z = 0.5  # Adjust angular velocity as needed
                        print("left")
                    else:
                        twist_cmd.angular.z = 0.0  # No angular velocity for straight movement
                    self.cmd_vel_pub.publish(twist_cmd)
                    
            # Display the image
            cv2.imshow("AGV_lane", mask_y)
            cv2.imshow("AGV_light",mask_r)
            cv2.imshow("detect_object", img)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    image_subscriber = ImageSubscriber()
    rospy.spin()
    cv2.destroyAllWindows()