import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
class CameraSubscriber:
    def __init__(self):
        rospy.init_node('camera_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.callback)
        self.detector = cv2.SIFT_create()  # SIFT detector 사용
        # 특정 이미지에서 특징점과 디스크립터 추출
        self.keypoints_image, self.des1 = self.extract_keypoints_descriptors_from_image()
    
    def extract_keypoints_descriptors_from_image(self):
        img = cv2.imread("/home/lee/client/KakaoTalk_20240501_104427483.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        print(len(good_matches))
        if len(good_matches) > 20:  # 매칭점이 일정 개수 이상이면 객체가 감지된 것으로 판단
            return True
        else:
            return False
    def callback(self, data):
        try:
            # Convert the compressed image message to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
            # Detect object
            if self.detect_object(cv_image):
                print("특정 이미지를 발견했습니다. 멈추는 동작을 수행합니다.")
                # 여기서 AGV를 제어하는 코드를 추가해야 합니다.
            # Display the image with detected bodies
            cv2.imshow("AGV Camera", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)
if __name__ == '__main__':
    camera_subscriber = CameraSubscriber()
    rospy.spin()