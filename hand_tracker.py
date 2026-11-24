"""Wrapper MediaPipe Hands : détection + classification du geste."""

from __future__ import annotations

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id='litert-community/MediaPipe-HandLandmarker-Full',
    filename='hand_landmarker.task',
)
model_path2 = hf_hub_download(
    repo_id='litert-community/MediaPipe-PoseLandmarker-Full',
    filename='pose_landmarker_full.task',
)

from config import (
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_COMPLEXITY,
)
from gestures import HandState, classify_gesture
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



class HandTracker:
    def __init__(self) -> None:
        # STEP 2: Create an HandLandmarker object.
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=MAX_HANDS, min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.drawsth=vision.drawing_utils.draw_landmarks
        base_options2 = python.BaseOptions(model_asset_path=model_path2)
        options2 = vision.PoseLandmarkerOptions(base_options=base_options2, num_poses=1, min_pose_detection_confidence=0.7, min_pose_presence_confidence=0.7)

        self.detector2 = vision.PoseLandmarker.create_from_options(options2)
        self.drawsth2=vision.drawing_utils.draw_landmarks
        

        

    def detect(self, frame: np.ndarray, draw: bool = True) -> HandState | None:
        """Analyse une frame BGR. Dessine le squelette de la main si draw=True."""
        # STEP 3: Load the input image.
        #image = mp.Image.create_from_file("image.jpg")

        image=mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )
        #frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        
        # STEP 4: Detect hand landmarks from the input image.
        detection_result = self.detector.detect(image)
        detection_result2 = self.detector2.detect(image)
        state=None



        try:
            if not detection_result.hand_landmarks:
                return None
            try:
                hand_lms = detection_result.hand_landmarks[0]
                state = classify_gesture(hand_lms, h, w)
            except:
                hand_lms=None
            try:
                pose_lms = detection_result2.pose_landmarks[0]
            except:
                pose_lms=None
            #state = classify_gesture(hand_lms, w, h)

            if draw:
                RED_COLOR = (0, 0, 255)
                WHITE_COLOR = (224, 224, 224)
                BLACK_COLOR = (0, 0, 0)
                RED_COLOR = (0, 0, 255)
                GREEN_COLOR = (0, 128, 0)
                BLUE_COLOR = (255, 0, 0)
                if hand_lms:
                    #print(hand_lms)
                    print("see and draw hand")
                    vision.drawing_utils.draw_landmarks(
                        image= frame,
                        landmark_list=(hand_lms),
                        connections=mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.tasks.vision.drawing_utils.DrawingSpec(color=RED_COLOR, thickness=2, circle_radius=3),
                        connection_drawing_spec=vision.drawing_utils.DrawingSpec(color=BLUE_COLOR, thickness=2),
                        is_drawing_landmarks=True
                    )
                if pose_lms:
                    print("draw pose")
                    vision.drawing_utils.draw_landmarks(
                        image= frame,
                        landmark_list=(pose_lms),
                        connections=mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
                        landmark_drawing_spec=mp.tasks.vision.drawing_utils.DrawingSpec(color=GREEN_COLOR, thickness=2, circle_radius=3),
                        connection_drawing_spec=vision.drawing_utils.DrawingSpec(color=WHITE_COLOR, thickness=2),
                        is_drawing_landmarks=True
                    )
        except Exception as e:
            print("e error hands", e)

        return state

    def close(self) -> None:
        print("hahaha")
        self.detector.close()
        self.detector2.close()
