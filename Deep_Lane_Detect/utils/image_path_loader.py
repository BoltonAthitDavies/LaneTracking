"""
Author: Sippawit Thammawiset
Date: September 1, 2024.
File: image_path_loader.py
"""

from typing import List
import re
import os


def load_image_path_from_directory(directory: str) -> List[str]:
    image_path = []

    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else float('inf')

    for file_name in sorted(os.listdir(directory), key=extract_number):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path.append(os.path.join(directory, file_name))

    return image_path
