"""
Author: Sippawit Thammawiset
Date: September 28, 2024.
File: extract_frames_from_video.py
"""

from utils import colors
from tqdm import tqdm
import cv2
import os
import argparse


"""
Raw dataset folder structure:
    raw_dataset/
    ├── EXCLUDE/  --> To exclude any video (e.g., recorded_X.mp4), place it here.
    ├── recorded_1.mp4
    ├── recorded_2.mp4
    ├── recorded_3.mp4
    └── ...
"""

def resize_with_padding(img, target_width=800, target_height=320):
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = target_width - new_w
    pad_h = target_height - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])  # Black border
    return padded

def extract_frame(video_path: str,
                  output_directory: str,
                  hertz: int,
                  video: str) -> int:
    video_filename = os.path.splitext(video)[0]
    os.makedirs(os.path.join(output_directory, video_filename, 'images', 'img'), exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f'{colors.ERROR}'
              f'[ERROR] Could not open video. Skipped.'
              f'{colors.ENDC}')
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_hertz = hertz if hertz > 0 else fps
    frame_interval = int(fps / target_hertz)

    frame_index = 0
    extracted_frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc='Extracting frames', unit='frames'):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            # resized_frame = cv2.resize(frame, (800, 320))  # Resize to 800x320
            resized_frame = resize_with_padding(frame, 1280, 720)
            frame_filename = os.path.join(output_directory, video_filename, 'images', 'img',
                                        f'{video_filename}-{extracted_frame_index:04d}.png')
            cv2.imwrite(frame_filename, resized_frame)
            extracted_frame_index += 1

        frame_index += 1

    cap.release()

    print(f'> Extracted {extracted_frame_index} frames from "{video_path}".')

    return extracted_frame_index


def main() -> None:
    parse = argparse.ArgumentParser(description='Extract frames from a batch of videos.')
    parse.add_argument('-i', '--input-directory', dest='input_directory', type=str, required=True,
                       help='Directory containing the raw dataset.')
    parse.add_argument('-o', '--output-directory', dest='output_directory', type=str, required=True,
                       help='Directory to save the extracted frames.')
    parse.add_argument('-f', '--frequency', dest='frequency', type=int, default=-1, required=False,
                       help='Frequency to extract frames. Default is -1 (extract all frames).')
    args = parse.parse_args()

    if not args.input_directory:
        parse.print_help()
        return

    input_directory: str = args.input_directory
    output_directory: str = args.output_directory
    frequency: int = args.frequency
    os.makedirs(output_directory, exist_ok=True)

    videos = [video for video in os.listdir(input_directory) if video != 'EXCLUDE' and video != '.DS_Store' and
              os.path.isfile(os.path.join(input_directory, video))]
    frame_counts = 0

    for i, video in enumerate(videos):
        print(f'[INFO] [{i + 1}/{len(videos)}] Processing: {video}.')
        video_path = os.path.join(input_directory, video)
        frame_counts += extract_frame(video_path, output_directory, frequency, video)

    print(f'{colors.OKGREEN}[INFO] Extracted {frame_counts} frames to "{output_directory}/".{colors.ENDC}')
    print(f'{colors.OKGREEN}[INFO] Done.{colors.ENDC}')


if __name__ == '__main__':
    main()

# python preprocessing/extract_frames_from_video_resize.py --input "raw_dataset" --output "dataset" --frequency 5