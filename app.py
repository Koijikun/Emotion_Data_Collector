from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st
import predict_functions as pf
import cv2
import time
import os

#VideoProcessor is needed to handle the frames that are used for prediction / saving data, and acts as a "mirror" for the user
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

def process_screenshot(img):
    #Save screenshot for testing
    current_time = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(f"screenshot_{current_time}.jpg", img)

    result = pf.classify_emotion(img)
    if result is None:
        return None, None
    
    predicted_emotion, probabilities = result
    return predicted_emotion, probabilities

def save_image(img, emotion):
    directory = f'new_data/{emotion}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    current_time = time.strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(directory, f"{current_time}.jpg")
    cv2.imwrite(filepath, img)

#session state had to be added because some radio- & selectboxes were not working properly
def reset_session_state():
    st.session_state.prediction_made = False
    st.session_state.is_correct = None
    st.session_state.probabilities = {}
    st.session_state.image = None

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

if 'is_correct' not in st.session_state:
    st.session_state.is_correct = None

if 'probabilities' not in st.session_state:
    st.session_state.probabilities = {}

if 'image' not in st.session_state:
    st.session_state.image = None

st.title("Emotion Detection - Help me improve!")

#splitting UI in 2 columns because the page was getting too "vertical"
col1, col2 = st.columns([2, 1])
with col1:
    
    st.write('1. Click the "Start" Button')
    st.write('2. Display One of the following emotions: angry, fear, happy, neutral, sad, surprise')
    st.write('3. Click the "Predict Emotion" Button')

    webrtc_ctx = webrtc_streamer(key="randomkey", video_processor_factory=VideoProcessor)


with col2:
    if webrtc_ctx.video_processor:
        if st.button('Predict Emotion'):
            img = webrtc_ctx.video_processor.frame
            if img is not None:
                predicted_emotion, probabilities = process_screenshot(img)

                if predicted_emotion is None or probabilities is None:
                    st.error('Error in predicting your emotion. Please try again and make sure your face is visible and facing the camera.')
                else:
                    if predicted_emotion == 'No Face Detected':
                        st.warning('No face detected')
                    else:
                        st.session_state.prediction_made = True
                        st.session_state.probabilities = probabilities
                        st.session_state.is_correct = None
                        st.session_state.image = img
                        st.success(f'Predicted Emotion: {predicted_emotion}')
                        # uncomment in order to see the specific probabilities
                        # st.write('Probabilities:')
                        # for emotion, probability in probabilities.items():
                        #     st.write(f'{emotion}: {probability:.2f}')

    if st.session_state.prediction_made:
        st.session_state.is_correct = st.radio("Is the prediction correct?", ("Select an option", "Yes", "No"))

        if st.session_state.is_correct == "No":
            correct_emotion = st.selectbox("What emotion were you displaying?", list(st.session_state.probabilities.keys()))
            st.write(f'Thanks! Can I hold on to your picture in order to improve my accuracy?')
            if st.button('Save Corrected Emotion'):
                save_image(st.session_state.image, correct_emotion)
                st.write(f'Image saved as {correct_emotion}')
                reset_session_state()
                st.rerun()
        elif st.session_state.is_correct == "Yes":
            predicted_emotion = max(st.session_state.probabilities, key=st.session_state.probabilities.get)
            save_image(st.session_state.image, predicted_emotion)
            st.write(f'Thanks! Image saved as {predicted_emotion}')
            reset_session_state()
            st.rerun()

    if st.button('Reset'):
        reset_session_state()
        st.rerun()