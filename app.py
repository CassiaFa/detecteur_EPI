from types import NoneType
import streamlit as st
import numpy as np
import time

from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes, VideoProcessorBase, WebRtcMode
import av

from capture_cam import *

@st.cache
def load_model(model_name, label=None, threshold=None):
    """ 
    Load the Yolo model
    
    Parameters:
    -----------
    model_name: str
        Path to the weight of model.
    threshold: float
        Threshol will be use for object detection in image/frame.

    Return:
    -------
    detector: Yolo5 model
        
    """

    # Instanciate the detector
    detector = model_detection(model_name=model_name, label=label, threshold=threshold)

    return detector

class VideoProcessor(VideoProcessorBase):
    
    def __init__(self):
        self.message_state = False

    def recv(self, frame):
        # Get the frame from webcam
        frm = frame.to_ndarray(format="rgb24")
        # Detect on the frame
        frm, self.message_state = detector.cam_detection(frm)
        
        return av.VideoFrame.from_ndarray(frm, format="rgb24")

def main():
    """ Safety equipement app"""

    global detector
    
    labels = ['Casque_OK','Casque_NO','Gilet_NO', 'Gilet_OK']
    # 0 for Casque_OK
    # 1 for Casque_NO
    # 2 for Gilet_NO
    # 3 for Gilet_OK

    CONFIDENCE_THRESHOLD = 0.6

    # Load model
    detector = load_model('Models/last_v2.pt', label=labels, threshold=CONFIDENCE_THRESHOLD)

    # Set a tile in interface
    st.title('Vérificateur de l Uniforme')

    # Set un empty element in interface
    c1 = st.empty()
    c1.subheader("Uniforme non vérifié")
    
    # State for printing message of conformity uniform 
    state = False
    
    # Instanciate the streamer
    stream = webrtc_streamer( key="safety_detector",
                mode=WebRtcMode.SENDRECV, 
                video_processor_factory=VideoProcessor,
                video_html_attrs=VideoHTMLAttributes( autoPlay=True, controls=True, style={"width": "100%"} ),
                media_stream_constraints={"video" : True, "audio" : False} )


    if type(stream.video_processor) != NoneType:
        while True:
            time.sleep(0.10)
            # Check the message state
            if stream.video_processor.message_state:
                # Message to display
                message = "Uniforme vérifié"

                state = True

                if state == True:
                    c1.empty()
                
                c1.subheader(message)
                
            else:
                if state == True:
                    state = False
                    c1.empty()
                c1.subheader("Uniforme non vérifié")


if __name__ == "__main__":
    main()