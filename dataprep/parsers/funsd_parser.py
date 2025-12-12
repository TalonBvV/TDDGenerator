"""
FUNSD Annotation Parser

Parses FUNSD JSON format annotations:
    {
        "form": [
            {"box": [x1,y1,x2,y2], "text": "...", "words": [...]}
        ]
    }
"""

import json
from pathlib import Path
from typing import List

from .base import BaseParser, TextInstance


class FUNSDParser(BaseParser):
    """Parser for FUNSD JSON format."""
    
    def __init__(self, ignore_text: str = "###"):
        super().__init__(ignore_text=ignore_text)
    
    def parse_file(self, ann_path: Path) -> List[TextInstance]:
        """Parse FUNSD annotation file."""
        instances = []
        
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for item in data.get("form", []):
                box = item.get("box", [])
                text = item.get("text", "")
                
                if len(box) != 4:
                    continue
                
                # Convert xyxy to 4-point polygon
                x1, y1, x2, y2 = box
                coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                
                instances.append(TextInstance(
                    polygon=[float(c) for c in coords],
                    text=text,
                    ignore=(text == self.ignore_text or not text.strip()),
                ))
                
        except Exception as e:
            print(f"Warning: Failed to parse {ann_path}: {e}")
        
        return instances
