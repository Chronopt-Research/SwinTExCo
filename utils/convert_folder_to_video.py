import cv2
import os
import argparse
from tqdm import tqdm

def convert_frames_to_video(input_folder_path, output_path, fps=24):
    list_frames = sorted(os.listdir(input_folder_path))
    first_frame = cv2.imread(os.path.join(input_folder_path, list_frames[0]))
    height, width, _ = first_frame.shape
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on the file extension
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in tqdm(list_frames):
        frame_path = os.path.join(input_folder_path, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
        
    video_writer.release()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--fps', type=int, default=24)
    args = parser.parse_args()
    
    convert_frames_to_video(args.input_folder_path, args.output_path, args.fps)