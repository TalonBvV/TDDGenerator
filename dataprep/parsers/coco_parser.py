"""
COCO Text Detection Annotation Parser

Parses COCO-style JSON annotations used by:
- TextOCR
- COCOText v2
- Other COCO-format text detection datasets
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseParser, TextInstance


class COCOParser(BaseParser):
    """Parser for COCO-style text detection format."""
    
    def __init__(self, ignore_text: str = "###"):
        super().__init__(ignore_text=ignore_text)
        self._annotations_by_image: Dict[str, List[TextInstance]] = {}
        self._loaded = False
        self._ann_path: Optional[Path] = None
    
    def parse_file(self, ann_path: Path) -> List[TextInstance]:
        """Parse COCO annotation file.
        
        COCO uses a single JSON file for all annotations, so we load once
        and cache by image.
        """
        if self._ann_path != ann_path:
            self._load_annotations(ann_path)
        
        # Return empty - need to call get_instances_for_image
        return []
    
    def _load_annotations(self, ann_path: Path) -> None:
        """Load all annotations from COCO JSON file."""
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Build image ID to filename mapping
            id_to_filename = {}
            for img in data.get("images", data.get("imgs", {})):
                if isinstance(img, dict):
                    img_id = img.get("id", "")
                    filename = img.get("file_name", "")
                    id_to_filename[str(img_id)] = filename
                elif isinstance(data.get("imgs"), dict):
                    # COCOText format: {"imgs": {"id": {info}}}
                    for img_id, info in data.get("imgs", {}).items():
                        filename = info.get("file_name", "")
                        id_to_filename[str(img_id)] = filename
                    break
            
            # Parse annotations
            annotations = data.get("annotations", data.get("anns", {}))
            
            if isinstance(annotations, list):
                # Standard COCO format
                for ann in annotations:
                    self._parse_annotation(ann, id_to_filename)
            elif isinstance(annotations, dict):
                # COCOText/TextOCR format: {"anns": {"id": {ann}}}
                for ann_id, ann in annotations.items():
                    self._parse_annotation(ann, id_to_filename)
            
            self._ann_path = ann_path
            self._loaded = True
            
        except Exception as e:
            print(f"Warning: Failed to parse {ann_path}: {e}")
    
    def _parse_annotation(self, ann: dict, id_to_filename: Dict[str, str]) -> None:
        """Parse a single annotation."""
        img_id = str(ann.get("image_id", ""))
        filename = id_to_filename.get(img_id, "")
        
        if not filename:
            return
        
        # Get polygon or bbox
        polygon = ann.get("segmentation", ann.get("polygon", []))
        bbox = ann.get("bbox", [])
        
        coords = []
        if polygon and isinstance(polygon, list):
            if isinstance(polygon[0], list):
                coords = polygon[0]  # [[x1,y1,x2,y2,...]]
            else:
                coords = polygon
        elif bbox and len(bbox) >= 4:
            # Convert xywh to polygon
            x, y, w, h = bbox[:4]
            coords = [x, y, x + w, y, x + w, y + h, x, y + h]
        
        if not coords or len(coords) < 4:
            return
        
        text = ann.get("text", ann.get("utf8_string", ""))
        
        # Initialize list for this image
        if filename not in self._annotations_by_image:
            self._annotations_by_image[filename] = []
        
        self._annotations_by_image[filename].append(TextInstance(
            polygon=[float(c) for c in coords],
            text=text,
            ignore=(text == self.ignore_text or ann.get("illegibility", False)),
        ))
    
    def get_instances_for_image(self, img_filename: str) -> List[TextInstance]:
        """Get instances for a specific image filename."""
        return self._annotations_by_image.get(img_filename, [])
