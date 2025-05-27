import os
import cv2
import numpy as np
from PIL import Image

def save_video_as_mp4(video_array, fps, output_path):
    # video_array: TCHW (RGB)
    height, width = video_array.shape[2], video_array.shape[3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for t in range(video_array.shape[0]):
        frame = video_array[t].transpose(1, 2, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # RGB->BGR
        out.write(cv2.convertScaleAbs(frame))
    out.release()

def save_video_as_png(video_array, output_path):
    os.makedirs(output_path, exist_ok=True)
    # video_array: TCHW (RGB)
    for i, sample in enumerate(video_array):
        sample = np.transpose(sample, (1, 2, 0))
        Image.fromarray(sample).save( # HWC
            os.path.join(output_path, f"{i:06d}.png")
        )
