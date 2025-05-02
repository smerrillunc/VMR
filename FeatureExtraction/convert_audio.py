import os
from moviepy import VideoFileClip
from tqdm import tqdm

path = '/work/users/s/m/smerrill/Youtube8m'

video_dir = os.path.join(path, 'video')
audio_dir = os.path.join(path, 'audio')
video_files = os.listdir(video_dir)
audio_files = os.listdir(audio_dir)

# Normalize file names to extract IDs (before any .fxxx or extensions)
def extract_id(filename):
    return filename.split('.')[0].split('.')[0]

# Make a set of IDs that already have audio
audio_ids = set(extract_id(f) for f in audio_files)

# Loop through videos
for video_file in tqdm(video_files):
    video_id = extract_id(video_file)
    
    if video_id in audio_ids:
        print(f'Audio Already Exists for {video_id}, skipping')
        # Audio already exists for this ID, skip
        continue
    else:
        print(f'Processing {video_id}')

    video_path = os.path.join(video_dir, video_file)
    try:
        clip = VideoFileClip(video_path)
        if clip.audio:
            audio_output_path = os.path.join(audio_dir, f"{video_id}.mp3")
            clip.audio.write_audiofile(audio_output_path)
            print("Writing Audio File")
        else:
            print("Audio Doesn't Exist")
            continue
    
        clip.close()
        break
    except Exception as e:
        print(f"Error processing {video_file}: {e}")