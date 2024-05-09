import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
class CameraSubscriber:
    def __init__(self):
        rospy.init_node('camera_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.callback)
        # Load the full body cascade classifier
        self.full_body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
        if self.full_body_cascade.empty():
            rospy.logerr("Failed to load full body cascade classifier.")
            exit()
    def callback(self, data):
        try:
            # Convert the compressed image message to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
            # Convert the image to grayscale for cascade classifier
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # Detect full body in the image
            bodies = self.full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # Draw rectangles around the detected bodies
            for (x, y, w, h) in bodies:
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Display the image with detected bodies
            cv2.imshow("AGV Camera", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)
if __name__ == '__main__':
    camera_subscriber = CameraSubscriber()
    rospy.spin()