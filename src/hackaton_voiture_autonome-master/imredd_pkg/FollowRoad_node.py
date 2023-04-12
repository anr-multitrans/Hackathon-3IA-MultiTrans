#!/usr/bin/env python3

import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
from PIL import Image as Img
import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from model_classif import get_mobilenet_classif_1
from model_lstm_2 import get_classif_lstm_1
#import tensorflow as tf
#import scipy.misc
#!/usr/bin/env python3



class follow_road:
    def __init__(self):
        print('init')
        self.model = get_mobilenet_classif_1().cuda()
        #self.model.fc = torch.nn.Linear(512, 1)
        #self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/test_03_02_2023_preprocessedimg.pth'))
        #self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/weights_model_lstm_3.ckpt'))
        self.model.load_state_dict(torch.load('/root/catkin_ws/src/imredd_pkg/best_model_classif.ckpt'))
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        #self.h0 = torch.rand((1,4,1024))

        self.steering_gain=float(rospy.get_param("steering_gain"))
        self.steering_dgain=float(rospy.get_param("steering_dgain"))
        self.steering_bias=float(rospy.get_param("steering_bias"))
        self.speed=float(-1.7)
        self.speed_factor = float(5)

        self.pub=rospy.Publisher("/car/mux/ackermann_cmd_mux/input/navigation",AckermannDriveStamped, queue_size=30)
        self.rate=rospy.Rate(30)
        
        self.bridge = CvBridge()
        self.angle_last=0.0
	
        self.image_rgb = rospy.Subscriber("/car/camera/color/image_raw", Image, self.callback, callback_args="rgb")
        self.dico = {'0': float(0.7),'1' : float(0.5),'2' : float(0.3),'3' : float(0.1),'4' : float(-0.1),'5' : float(-0.3),'6' : float(-0.5),'7' : float(-0.5)}
        self._run()

    def _run(self):
        rospy.spin()

    def get_image(self,msg):

        # img_size = (240,424,3)
        #print('msg',msg)


        msg_image = msg.data
        #print('msg_image',msg_image)

        msg_image_2 = np.frombuffer(msg_image, dtype=np.uint8)

        
        img_1 = np.reshape(msg_image_2, (240,424,3), order='C')
    
        img_1 = img_1[120:]
        #img_1[:20] = np.zeros((20,424,3))
        #copy_img = img_1 > 150 
        #img_1 = copy_img*255


        
        return img_1


    #def preprocess(self, img_rgb,img_depth):
        #image_rgb = self.bridge.imgmsg_to_cv2(img, "bgr8")
        #image_rgb = self.get_image(img_rgb)
        #image_depth = self.get_depth_image(img_depth)
        #return image_rgb,image_depth

    def preprocess(self, img_rgb):
        if img_rgb is not None:

            image_rgb = torch.tensor(self.get_image(img_rgb))
            #image_rgb = torch.cat([torch.unsqueeze(torch.where(image_rgb[:,:,0] > 160, 255, 0),dim=0), torch.unsqueeze(torch.where(image_rgb[:,:,1] > 160, 255, 0),dim=0), torch.unsqueeze(torch.where(image_rgb[:,:,2] > 160, 255, 0), dim=0)], dim=0).float()

            #image_rgb[:20] = torch.zeros((20,424,3))
        

            image_rgb = torch.swapaxes(image_rgb, -1, 0)
            image_rgb = torch.swapaxes(image_rgb, -1, 1).float()

            #image_rgb = torch.unsqueeze(torch.tensor(image_rgb), dim=0).float()
            image_rgb = torch.unsqueeze(torch.tensor(image_rgb), dim=0).float().cuda()

            

            return image_rgb

    #def callback(self, img_rgb,img_depth):
        #speed, steering = self.model(self.preprocess(img_rgb,img_depth)).detach().float().cpu().numpy().flatten()
    def callback(self, img,args):
        if args == "rgb":
            self.img_rgb = img
        

        

        if self.img_rgb is not None:
            img_rgb = self.img_rgb
            
            input_1 = self.preprocess(img_rgb)

            steering_label = self.model(input_1)
            #steering_label = self.model(input_1)
            
            steering_label = torch.argmax(steering_label, dim = -1)
            steering_label = torch.unsqueeze(steering_label,dim=0).item()
            steering = self.dico[str(steering_label)]+float(0.11)
            speed = self.speed

            
            steering = steering*self.steering_gain
    
        print('steering : ',steering)
        print('speed : ',speed)

        commande = AckermannDriveStamped()
        commande.drive.speed = speed * self.speed_factor
        commande.drive.steering_angle = steering 
        self.pub.publish(commande)
        self.rate.sleep()


if __name__=="__main__":
    try:
        rospy.init_node('FollowRoad_node', anonymous=True)
        p = follow_road()
    except rospy.ROSInterruptException:
        pass





