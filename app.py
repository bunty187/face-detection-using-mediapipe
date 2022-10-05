import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

DEMO_VIDEO = '1.mp4'
DEMO_IMAGE = '1.png'

st.title('Face Detection Application using MediaPipe and Streamlit')
# <h3 style='text-align: center; color: red;'>{face_count}</h3>"

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
# st.sidebar.subheader('Parameters')


def fancyDraw(img, bbox, l=30, rt=1, t=7):
    x, y, w, h = bbox
    x1, y1 = x+w, y+h
    cv2.rectangle(img, bbox, (255, 255, 0), rt)

    # Top left x,y
    cv2.line(img, (x, y), (x+l, y), (255, 255, 0), t)
    cv2.line(img, (x, y), (x, y+l), (255, 255, 0), t)

    # Top Right x1,y
    cv2.line(img, (x1, y), (x1-l, y), (255, 255, 0), t)
    cv2.line(img, (x1, y), (x1, y+l), (255, 255, 0), t)

    # Bottom Left x,y1
    cv2.line(img, (x, y1), (x+l, y1), (255, 255, 0), t)
    cv2.line(img, (x, y1), (x, y1-l), (255, 255, 0), t)

    # Bottom Right x1,y1
    cv2.line(img, (x1, y1), (x1-l, y1), (255, 255, 0), t)
    cv2.line(img, (x1, y1), (x1, y1-l), (255, 255, 0), t)

    return img


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['Run on Image', 'Run on Video']
                                )

if app_mode == 'Run on Image':
    # drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

    # max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    # st.sidebar.markdown('---')
    # detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    # st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader(
        "Upload an image", type=["jpg", "jpeg", 'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    face_count = 0
    # Dashboard
    with mp_face_detection.FaceDetection(
        # static_image_mode=True,
        # max_num_faces=max_faces,
            min_detection_confidence=0.5) as face_detection:

        results = face_detection.process(image)
        # out_image = image.copy()
        bboxs = []

        # for face_landmarks in results.multi_face_landmarks:
        #     face_count += 1

        #     #print('face_landmarks:', face_landmarks)

        #     mp_drawing.draw_landmarks(
        #     image=out_image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_detection.FACE_CONNECTIONS,
        #     landmark_drawing_spec=drawing_spec,
        #     connection_drawing_spec=drawing_spec)
        #     kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        # st.subheader('Output Image')
        # st.image(out_image,use_column_width= True)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)

                bboxs.append([bbox, detection.score])
                face_count += 1
                # mp_drawing.draw_detection(image, detection)
                image = fancyDraw(image, bbox)

                cv2.putText(image, f'{int(detection.score[0]*100)}%',
                            (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

                kpi1_text.write(
                    f"<h3 style='text-align: center; color: red;'>{face_count}</h3>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(image, use_column_width=True)


elif app_mode == 'Run on Video':
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    # record = st.sidebar.checkbox("Record Video")
    # if record:
    #     st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # max faces
    # max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    # st.sidebar.markdown('---')
    # detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    # tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    # st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader(
        "Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    # codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    codec=cv2.VideoWriter_fourcc(*'mpv4')
    out = cv2.VideoWriter('output.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    # drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

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

    with mp_face_detection.FaceDetection(
            min_detection_confidence=0.5) as face_detection:
        prevTime = 0

        while vid.isOpened():
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue

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
                    image = fancyDraw(frame, bbox)

                    cv2.putText(image, f'{int(detection.score[0]*100)}%',
                                (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                    # mp_drawing.draw_detection(frame, detection)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            # if record:
                #st.checkbox("Recording", value=True)
                # out.write(frame)
        # Dashboard
            kpi1_text.write(
                f"<h2 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(
                f"<h2 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(
                f"<h2 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')


    output_video = open('output.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out. release()
