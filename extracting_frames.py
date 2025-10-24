import cv2
import os
import csv

def extract_frames(video_path, output_dir, video_id, category, csv_writer):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)

    os.makedirs(output_dir, exist_ok=True)

    for sec in range(duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if ret:
            frame_filename = f"{sec}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            # Log to CSV
            csv_writer.writerow([video_id, category, sec, frame_path])
    cap.release()


# Root directories
FIGHT_DIR = './Video Fights for Learning/fight-detection-surv-dataset-master/fight'
NON_FIGHT_DIR = './Video Fights for Learning/fight-detection-surv-dataset-master/noFight'

# Output root
OUTPUT_ROOT = './Processed_Frames'
OUTPUT_FIGHT = os.path.join(OUTPUT_ROOT, "fight")
OUTPUT_NON_FIGHT = os.path.join(OUTPUT_ROOT, "non_fight")

# CSV setup
csv_file = open("frames_index.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["video_id", "category", "frame_index", "frame_path"])  # header

# Process fight videos
for i, file in enumerate(os.listdir(FIGHT_DIR)):
    if file.lower().endswith((".mp4", ".avi", ".mov")):
        video_path = os.path.join(FIGHT_DIR, file)
        output_dir = os.path.join(OUTPUT_FIGHT, f"video_{i}")
        extract_frames(video_path, output_dir, f"fight_{i}", "fight", csv_writer)

# Process non-fight videos
for i, file in enumerate(os.listdir(NON_FIGHT_DIR)):
    if file.lower().endswith((".mp4", ".avi", ".mov")):
        video_path = os.path.join(NON_FIGHT_DIR, file)
        output_dir = os.path.join(OUTPUT_NON_FIGHT, f"video_{i}")
        extract_frames(video_path, output_dir, f"nonfight_{i}", "non_fight", csv_writer)

csv_file.close()
