#!/usr/bin/env python3
"""Create distributable zip package for the project."""

from pathlib import Path
import zipfile


DEFAULT_FILES = [
    "face_recognition_system.py",
    "system_test_script.py",
    "README.md",
    "requirements.txt",
    "package_project.py",
    "web_interface.py",
]


def create_zip(zip_name="Face-Recognition-System.zip", files=None):
    files = files or DEFAULT_FILES
    zip_path = Path(zip_name)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in files:
            file_path = Path(file_name)
            if file_path.exists():
                archive.write(file_path, arcname=file_path.name)
                print(f"Added: {file_path}")
            else:
                print(f"Skipped missing file: {file_path}")

    print(f"Created archive: {zip_path.resolve()}")
    return zip_path


if __name__ == "__main__":
    create_zip()
