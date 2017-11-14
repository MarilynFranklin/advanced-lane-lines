from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
from window import Window
from pipeline import ProcessImages

Output_video = 'output1_tracked.mp4'
Input_video = 'project_video.mp4'

clip1 = VideoFileClip(Input_video)
process_images = ProcessImages()
video_clip = clip1.fl_image(process_images.run_pipeline)
video_clip.write_videofile(Output_video, audio=False)
