import torch
import cv2

class model_detection():

    def __init__(self, model_name, label, threshold):
        ''' 
        Initialization of model object
        
        Parameters:
        -----------
        model_name: str
            Path of the weight to load.
        threshold: float
            Threshold to use for detection in prediction.
        '''
        # Load the model 
        self.model = self.load_model(model_name)
        # Classes of the model
        if label:
            self.classes = label
        else:
            self.classes = self.model.names
        # Threshold
        if threshold:
            self.threshold = threshold
        else:
            self.threshold = 0.8
                    
    
    def load_model(self, model_name):

        '''
        Loads Yolo5 model from pytorch hub. If model_name is given loads a custom model, else loads the pretrained model.
        
        return: 
            Trained Pytorch model.
        '''
        
        if model_name:
            # Use local weight
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name)
        else:
            # Use Yolo5 weight
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        return model

    def cam_detection(self, image):
        '''
        Detection of object in image

        Parameters:
        -----------
        image: Array
            Image/frame used for the detection.

        Return:
        -------
        image: Array
            Image/frame with element detected.
        state: bool
            State of detection, True when all uniform is complete else False is return. 
        '''
        # Model prediction
        results = self.model(image)
        # Image resulted of prediction
        image = results.ims[0]
        # Shape of image
        x_shape, y_shape = image.shape[1], image.shape[0]
        # Labels of prediction
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # Number of labels
        n = len(labels)
        


        for i in range(n):
            # Coordonate and probability of detection
            row = cord[i]

            if row[4] >= self.threshold:
                # Coordonate of detection
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                if (labels[i] == 0) or (labels[i] == 3):
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                # Draw text
                cv2.putText(image, f"{self.classes[int((labels[i]))]} : {row[4]:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        if (1 in labels) or (2 in labels):
            state = False
        else:
            state = True

        return image, state