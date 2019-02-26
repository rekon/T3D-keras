import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
import random

ROOT_PATH = ''


def get_video_frames(src, fpv, frame_height, frame_width):
    # print('reading video from', src)
    cap = cv2.VideoCapture(src)

    frames = []
    if not cap.isOpened():
        cap.open(src)
    ret = True
    while(True and ret):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()

    # Returning fpv=10 frames
    step = len(frames)//fpv
    avg_frames = frames[::step]
    avg_frames = avg_frames[:fpv]
    avg_resized_frames_2d = []
    avg_resized_frames_3d = []
    for af in avg_frames:
        rsz_f_3d = cv2.resize(af, (frame_width, frame_height))
        rsz_f_2d = cv2.resize(af, (224, 224))
        avg_resized_frames_2d.append(rsz_f_2d)
        avg_resized_frames_3d.append(rsz_f_3d)
    return np.asarray(avg_resized_frames_2d)/255.0, np.asarray(avg_resized_frames_3d)/255.0


def get_video_and_label(index, data, frames_per_video, frame_height, frame_width):
    # Read clip and appropiately send the sports' class
    frames, clip = get_video_frames(os.path.join(
        ROOT_PATH, data['path'].values[index].strip()), frames_per_video, frame_height, frame_width)
    sport_class = data['class'].values[index]

    clip = np.expand_dims(clip, axis=0)

    # print('Frame shape',frame.shape)
    # print('Clip shape',clip.shape)


    return frames, clip, sport_class


def video_gen(data, frames_per_video, frame_height, frame_width, channels, num_classes, batch_size=1):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(data.count()[0])
        for batch in range(0, len(indices_arr), batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]

            # initializing the arrays, x_train and y_train
            input_3d = np.empty([0, frames_per_video, frame_height, frame_width, channels], dtype=np.float32)
            input_2d = np.empty([0, 224, 224, 3], dtype=np.float32)

            y_train = np.empty([0], dtype=np.int32)

            for i in current_batch:
                # get frames and its corresponding color for an traffic light
                frames, single_clip, sport_class = get_video_and_label(
                    i, data, frames_per_video, frame_height, frame_width)

                # Appending them to existing batch
                input_2d = np.append(input_2d, frames, axis=0)
                input_3d = np.append(input_3d, single_clip, axis=0)

                y_train = np.append(y_train, [sport_class])
            y_train = to_categorical(y_train, num_classes=num_classes)

            yield ([input_2d, input_3d], y_train)
