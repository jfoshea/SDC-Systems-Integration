from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime

class TLClassifier(object):
    def __init__(self, is_sim):

        self.graph = tf.Graph()
        self.threshold = .40
        
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile( 'light_classification/ssd_mobilenet_v1/frozen_inference_graph.pb', 'rb' ) as fname:
                graph_def.ParseFromString( fname.read() )
                tf.import_graph_def( graph_def, name='' )

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.session = tf.Session( graph = self.graph )

    def get_classification(self, image):
        with self.graph.as_default():
            img_expand = np.expand_dims( image, axis=0 )
            (boxes, scores, classes, num_detections) = self.session.run( 
                    [ self.boxes, 
                    self.scores, 
                    self.classes, 
                    self.num_detections ],
                feed_dict={ self.image_tensor: img_expand } )

        boxes = np.squeeze( boxes )
        scores = np.squeeze( scores )
        classes = np.squeeze( classes ).astype( np.int32 )

        if scores[0] > self.threshold:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[0] == 3:
                return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN
