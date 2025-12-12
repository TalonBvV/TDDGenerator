"""
Wildreceipt Annotation Parser

Parses Wildreceipt line-based format:
    Each line in train.txt/test.txt contains a JSON object with image path and annotations
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseParser, TextInstance


class WildreceiptParser(BaseParser):
    """Parser for Wildreceipt format."""
    
    def __init__(self, ignore_text: str = "###"):
        super().__init__(ignore_text=ignore_text)
        self._annotations_by_image: Dict[str, List[TextInstance]] = {}
        self._loaded = False
        self._ann_path: Optional[Path] = None
    
    def parse_file(self, ann_path: Path) -> List[TextInstance]:
        """Parse Wildreceipt annotation file.
        
        Format: Each line is a JSON object with:
            {
                "file_name": "path/to/image.jpg",
                "annotations": [
                    {"box": [x1,y1,x2,y2,x3,y3,x4,y4], "text": "..."}
                ]
            }
        """
        if self._ann_path != ann_path:
            self._load_annotations(ann_path)
        
        return []
    
    def _load_annotations(self, ann_path: Path) -> None:
        """Load all annotations from Wildreceipt file."""
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        filename = data.get("file_name", "")
                        
                        if not filename:
                            continue
                        
                        instances = []
                        for ann in data.get("annotations", []):
                            box = ann.get("box", [])
                            text = ann.get("text", "")
                            
                            if len(box) < 4:
                                continue
                            
                            # Convert to polygon if needed
                            if len(box) == 4:
                                # xyxy format
                                x1, y1, x2, y2 = box
                                coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                            else:
                                # Already a polygon
                                coords = box
                            
                            instances.append(TextInstance(
                                polygon=[float(c) for c in coords],
                                text=text,
                                ignore=(text == self.ignore_text),
                            ))
                        
                        self._annotations_by_image[filename] = instances
                        
                    except json.JSONDecodeError:
                        continue
            
            self._ann_path = ann_path
            self._loaded = True
            
        except Exception as e:
            print(f"Warning: Failed to parse {ann_path}: {e}")
    
    def get_instances_for_image(self, img_filename: str) -> List[TextInstance]:
        """Get instances for a specific image filename."""
        return self._annotations_by_image.get(img_filename, [])
