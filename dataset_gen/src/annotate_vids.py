import cv2
import sys
import os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

# Path to the directory containing videos
target_scene = "test_obj_moving"
video_dir = f"../scene/{target_scene}/output"
video_files = []
for case_dir in os.listdir(video_dir):
    video_files.append(os.path.join(video_dir, f"{case_dir}/render.mkv"))

video_files.sort()

annotations = []

for idx, video_file_path in enumerate(video_files):
    video_path = os.path.abspath(video_file_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_file_path}")
        annotations.append(None)
        continue

    print(f"Showing video {idx + 1}/{len(video_files)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video Annotation Tool", frame)
        
        # Press 'q' to quit early if needed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()

    # Wait for user annotation
    user_input = input(f"Enter annotation:")
    try:
        annotation = float(user_input)  # store as number (float)
    except ValueError:
        print("Invalid input, storing None.")
        annotation = None
    annotations.append(annotation)

cv2.destroyAllWindows()

# Save annotations as numpy array
annotations_array = np.array(annotations, dtype=object)
np.save("annotations.npy", annotations_array)

print("Annotation completed. Saved to annotations.npy")