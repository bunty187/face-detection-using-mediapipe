import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

DEMO_VIDEO = '1.mp4'
DEMO_IMAGE = '1.png'

st.title('Face Detection Application using MediaPipe and Streamlit')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Face Detection Application using MediaPipe and Streamlit')

def fancyDraw(img, bbox, l=30, rt=1, t=7):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    cv2.rectangle(img, bbox, (255, 255, 0), rt)

    # Top left x, y
    cv2.line(img, (x, y), (x + l, y), (255, 255, 0), t)
    cv2.line(img, (x, y), (x, y + l), (255, 255, 0), t)

    # Top Right x1, y
    cv2.line(img, (x1, y), (x1 - l, y), (255, 255, 0), t)
    cv2.line(img, (x1, y), (x1, y + l), (255, 255, 0), t)

    # Bottom Left x, y1
    cv2.line(img, (x, y1), (x + l, y1), (255, 255, 0), t)
    cv2.line(img, (x, y1), (x, y1 - l), (255, 255, 0), t)

    # Bottom Right x1, y1
    cv2.line(img, (x1, y1), (x1 - l, y1), (255, 255, 0), t)
    cv2.line(img, (x1, y1), (x1, y1 - l), (255, 255, 0), t)

    return img

@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(h)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

app_mode = st.sidebar.selectbox('Choose the App mode', ['Run on Image', 'Run on Video', 'Run on Webcam'])

if app_mode == 'Run on Image':
    st.sidebar.markdown('---')
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    face_count = 0

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image)
        bboxs = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)

                bboxs.append([bbox, detection.score])
                face_count += 1
                image = fancyDraw(image, bbox)

                cv2.putText(image, f'{int(detection.score[0]*100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

                kpi1_text.write(
                    f"<h3 style='text-align: center; color: red;'>{face_count}</h3>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(image, use_column_width=True)

elif app_mode == 'Run on Video':
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.markdown(' ## Output')

    stframe = st.empty()

    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tfflie.name = DEMO_VIDEO
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc(*'mpv4')
    output_filepath = 'output.mp4'
    out = cv2.VideoWriter(output_filepath, codec, fps_input, (width, height))

    st.sidebar.text('Input Video')

    fps = 0
    i = 0

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        prevTime = 0

        while True:
            i += 1
            ret, frame = vid.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = face_detection.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            face_count = 0
            bboxs = []

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)

                    bboxs.append([bbox, detection.score])

                    face_count += 1
                    frame = fancyDraw(frame, bbox)

                    cv2.putText(frame, f'{int(detection.score[0]*100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            kpi1_text.write(f"<h2 style='text-align: center; color: red;'>{int(fps)}</h2>", unsafe_allow_html=True)
            kpi2_text.write(f"<h2 style='text-align: center; color: red;'>{face_count}</h2>", unsafe_allow_html=True)
            kpi3_text.write(f"<h2 style='text-align: center; color: red;'>{width}</h2>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)

            out.write(frame)

    vid.release()
    out.release()

    if os.path.exists(output_filepath):
        output_video = open(output_filepath, 'rb')
        out_bytes = output_video.read()
        st.video(out_bytes)
    # else:
        # st.error("Error: The output video file was not found.")

elif app_mode == 'Run on Webcam':
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        def transform(self, frame):
            img = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)

            results = self.face_detection.process(img)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)

                    img = fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

            return img

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
