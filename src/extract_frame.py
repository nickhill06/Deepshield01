import cv2
import os

def extract_frames(video_path, output_folder, fps_sample=5):
    """
    Breaks a video into individual frame images.
    video_path    → path to your .mp4 video
    output_folder → where to save the frames
    fps_sample    → how many frames to save per second (5 is enough)
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get original FPS of the video (usually 24 or 30)
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # How many frames to skip between each save
    # Example: 30fps video, we want 5fps → save every 6th frame
    frame_interval = max(1, int(original_fps / fps_sample))

    frame_count = 0   # total frames read
    saved_count = 0   # total frames saved

    while True:
        # Read one frame from the video
        # ret = True if successful, False if video ended
        ret, frame = cap.read()

        if not ret:
            break  # video ended, stop

        # Only save every Nth frame
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()  # free memory
    print(f"✅ Extracted {saved_count} frames from {os.path.basename(video_path)}")
    return saved_count


def process_dataset(raw_folder, processed_folder):
    """
    Goes through ALL videos in raw_folder
    and extracts frames for each one
    """
    for root, dirs, files in os.walk(raw_folder):
        for filename in files:
            if filename.endswith(('.mp4', '.avi', '.mov')):

                video_path = os.path.join(root, filename)

                # Mirror folder structure: raw/real/vid1.mp4
                # becomes processed/real/vid1/
                relative_path = os.path.relpath(root, raw_folder)
                video_name = os.path.splitext(filename)[0]
                output_folder = os.path.join(
                    processed_folder, relative_path, video_name
                )

                print(f"📹 Processing: {video_path}")
                extract_frames(video_path, output_folder)


if __name__ == "__main__":
    process_dataset(
        raw_folder="data/raw",
        processed_folder="data/processed"
    )