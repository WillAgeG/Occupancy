import pathlib
import subprocess
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <dataset_root>")
    sys.exit(1)

root = pathlib.Path(sys.argv[1]).resolve()
if not root.is_dir():
    print("Given path is not a directory")
    sys.exit(1)

videos_dir = root / "videos"
videos_dir.mkdir(exist_ok=True)

print("Searching for scenes containing frame_*.png ...")

# Находим все сцены (каталоги верхнего уровня с кадрами)
scenes = [
    p for p in root.iterdir()
    if p.is_dir() and list(p.glob("frame_[0-9][0-9][0-9][0-9][0-9][0-9].png"))
]

scenes = sorted(scenes)
print(f"Found {len(scenes)} scenes.")

# Первым делом: создаём отдельные видео
scene_videos = []

for scene in scenes:
    scene_name = scene.name
    output_path = videos_dir / f"{scene_name}.mp4"

    print(f"\nProcessing: {scene_name}")
    print(f"→ Output: {output_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", "10",
        "-i", "frame_%06d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    subprocess.run(cmd, check=False, cwd=scene)
    scene_videos.append(output_path)

print("\nAll individual scene videos generated.")

# Теперь создаём единый файл со списком видео
list_file = videos_dir / "videos_list.txt"
with pathlib.Path(list_file).open("w") as f:
    f.writelines(f"file '{vid}'\n" for vid in scene_videos)

# Имя финального видео
final_video = videos_dir / "all_scenes.mp4"

print(f"\nConcatenating all videos into: {final_video}")

# Команда для объединения через concat demuxer
concat_cmd = [
    "ffmpeg",
    "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", str(list_file),
    "-c", "copy",
    str(final_video)
]

subprocess.run(concat_cmd, check=False)

print("\nDONE: Final video created at", final_video)
