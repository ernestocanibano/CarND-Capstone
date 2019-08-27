from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import os
import cv2

import visualization_utils as vis_util

import rospy #to remove
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
MIN_SCORE = 0.5

class TLClassifier(object):
	def __init__(self, is_site=False):
		#TODO load classifier
		# Just for debugging
		#self.bridge = CvBridge()
		#self.image_publisher = rospy.Publisher('image_predicted', Image, queue_size=1)
		#self.category_index = {1: {'id': 1, 'name': u'red'}, 2: {'id': 2, 'name': u'yellow'}, 3: {'id': 3, 'name': u'green'}, 4: {'id': 4, 'name': u'off'}}

		self.classes = {1: TrafficLight.RED,
						2: TrafficLight.YELLOW,
						3: TrafficLight.GREEN,
						4: TrafficLight.UNKNOWN}
		
		if is_site:
			self.path_to_ckpt = os.path.dirname(os.path.realpath(__file__))+'/models/ssd_inception_v2_real_graph.pb'
		else:
			self.path_to_ckpt = os.path.dirname(os.path.realpath(__file__))+'/models/ssd_inception_v2_sim_graph.pb'
		
		# optimization options
		config = tf.ConfigProto()
		config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.6
		
		# load the graph
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
		
		# init detection
		self.session = tf.Session(graph=self.detection_graph, config=config)
		self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
	def load_image_into_numpy_array(self, image):
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)    
		
       
	def get_classification(self, image):
		"""Determines the color of the traffic light in the image
		
		Args:
			image (cv::Mat): image containing the traffic light
		
		Returns:
			int: ID of traffic light color (specified in styx_msgs/TrafficLight)
		
		"""
		image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
		image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_np_expanded = np.expand_dims(image_np, axis=0)
		
		# Actual detection.
		(boxes, scores, classes, num_detections) = self.session.run( [self.boxes, self.scores, self.classes, self.num_detections], feed_dict={self.image_tensor: image_np_expanded})
		
		scores = np.squeeze(scores)
		classes = np.squeeze(classes)
		boxes = np.squeeze(boxes)
		
		# Visualization of the results of a detection. Just for debugging
		#vis_util.visualize_boxes_and_labels_on_image_array( image_np, boxes, classes.astype(np.int32), scores, self.category_index, use_normalized_coordinates=True, line_thickness=4)
		#image = self.prediction(image_np)
		#image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		#self.image_publisher.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
		
		num_red = 0
		num_yellow = 0
		num_green = 0
		for i, box in enumerate(boxes):
			if scores[i] > MIN_SCORE:
				if classes[i] == 1:
					num_red = num_red + 1
				elif classes[i] == 3:
					num_green = num_green + 1
				elif classes[i] == 2:
					num_yellow = num_yellow + 1
		
		if(num_red >= num_green and num_red >= num_yellow and num_red > 0):
			light_class = TrafficLight.RED
		elif(num_green > num_red and num_green > num_yellow and num_green > 0):
			light_class = TrafficLight.GREEN	
		elif(num_yellow > num_red and num_yellow >= num_green and num_yellow > 0):
			light_class = TrafficLight.YELLOW
		else:
			light_class = TrafficLight.UNKNOWN

        
		return light_class
