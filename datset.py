import os
import pandas as pd


# Read keypoint CSV
KEYPOINT_CSV = r'C:\Users\WIN 11\Downloads\yoloposeshopliftingmain\yolo-pose-shoplifting-main\nkeypoint.csv'
df = pd.read_csv(KEYPOINT_CSV)

# Paths for dataset output and class folders
dataset_path = r'C:\Users\WIN 11\Downloads\yoloposeshopliftingmain\yolo-pose-shoplifting-main\dataset_path'
sus_path = os.path.join(dataset_path, 'Suspicious')
normal_path = os.path.join(dataset_path, 'Normal')

# Ensure the dataset folders exist
os.makedirs(sus_path, exist_ok=True)
os.makedirs(normal_path, exist_ok=True)

# Build filename sets for faster lookup
sus_files = set(os.listdir(sus_path)) if os.path.isdir(sus_path) else set()
normal_files = set(os.listdir(normal_path)) if os.path.isdir(normal_path) else set()

def get_label(image_name: str) -> str:
    """Return 'Suspicious' or 'Normal' if the image exists in the corresponding folder, otherwise None."""
    if image_name in sus_files:
        return 'Suspicious'
    if image_name in normal_files:
        return 'Normal'
    return None

# Apply labeling
df['label'] = df['image_name'].astype(str).apply(get_label)

# Save the result (use os.path.join to build path)
out_csv = os.path.join(dataset_path, 'dataset.csv')
df.to_csv(out_csv, index=False)
print(f"Wrote labeled dataset to: {out_csv}")