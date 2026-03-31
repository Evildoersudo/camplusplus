import shutil
from pathlib import Path

import kagglehub


project_root = Path(__file__).resolve().parents[4]
output_dir = project_root / "egs" / "3dspeaker" / "sv-cam++" / "data"
output_dir.mkdir(parents=True, exist_ok=True)

cache_path = Path(kagglehub.dataset_download("pratt3000/vctk-corpus"))
target_dir = output_dir / cache_path.name

if target_dir.exists():
    shutil.rmtree(target_dir)

shutil.copytree(cache_path, target_dir)

print("Downloaded cache path:", cache_path)
print("Copied dataset to:", target_dir)
