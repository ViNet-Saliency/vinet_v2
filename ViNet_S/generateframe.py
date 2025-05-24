import os
import subprocess

input_folder = './video'
output_folders = ['./val', './annotation']

for output_folder in output_folders:
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.endswith('.AVI')]

    for video_file in video_files:
        input_file = os.path.join(input_folder, video_file)
        output_file_prefix = os.path.splitext(video_file)[0]
        output_file_pattern = os.path.join(output_folder, f'0{output_file_prefix}')

        # Run FFmpeg command to extract frames from the current video file
        # ffmpeg -i /content/video' movie(num).name '  /content/' videofolder '/%04d.png
        if not os.path.isdir(output_file_pattern):
            continue
        output_file = os.path.join(output_file_pattern, f'frames')
        os.makedirs(output_file, exist_ok=True)
        command = f'ffmpeg -i {input_file} {output_file}/%04d.png'
        subprocess.call(command, shell=True)
        # print(command)