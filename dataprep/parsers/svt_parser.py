"""
SVT (Street View Text) Annotation Parser

Parses SVT XML format with structure:
    <imageset>
        <image>
            <imageName>filename.jpg</imageName>
            <taggedRectangles>
                <taggedRectangle x="100" y="200" width="50" height="30">
                    <tag>text</tag>
                </taggedRectangle>
            </taggedRectangles>
        </image>
    </imageset>
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from .base import BaseParser, TextInstance


class SVTParser(BaseParser):
    """Parser for SVT XML format."""
    
    def __init__(self, ignore_text: str = "###"):
        super().__init__(ignore_text=ignore_text)
        self._parsed_data: Dict[str, List[TextInstance]] = {}
    
    def parse_file(self, ann_path: Path) -> List[TextInstance]:
        """Parse SVT annotation file.
        
        Note: SVT uses a single XML file for all images, so we cache results.
        """
        # Check if already parsed
        ann_str = str(ann_path)
        if ann_str in self._parsed_data:
            return self._parsed_data.get(ann_str, [])
        
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            # Parse all images
            all_data = {}
            for image in root.findall(".//image"):
                img_name_elem = image.find("imageName")
                if img_name_elem is None:
                    continue
                img_name = img_name_elem.text
                
                instances = []
                for rect in image.findall(".//taggedRectangle"):
                    x = float(rect.get("x", 0))
                    y = float(rect.get("y", 0))
                    w = float(rect.get("width", 0))
                    h = float(rect.get("height", 0))
                    
                    tag_elem = rect.find("tag")
                    text = tag_elem.text if tag_elem is not None and tag_elem.text else ""
                    
                    # Convert to 4-point polygon
                    coords = [x, y, x + w, y, x + w, y + h, x, y + h]
                    
                    instances.append(TextInstance(
                        polygon=coords,
                        text=text,
                        ignore=(text == self.ignore_text),
                    ))
                
                all_data[img_name] = instances
            
            self._parsed_data = all_data
            
        except Exception as e:
            print(f"Warning: Failed to parse {ann_path}: {e}")
        
        return self._parsed_data.get(ann_str, [])
    
    def get_instances_for_image(self, img_name: str) -> List[TextInstance]:
        """Get instances for a specific image name."""
        return self._parsed_data.get(img_name, [])
