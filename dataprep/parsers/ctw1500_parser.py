"""
CTW1500 Annotation Parsers

CTW1500 uses different formats for train (XML) and test (TXT) splits.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from .base import BaseParser, TextInstance


class CTW1500XMLParser(BaseParser):
    """Parser for CTW1500 XML annotations (train split)."""
    
    def __init__(self, ignore_text: str = "###"):
        """Initialize CTW1500 XML parser."""
        super().__init__(ignore_text=ignore_text)
    
    def parse_file(self, ann_path: Path) -> List[TextInstance]:
        """
        Parse CTW1500 XML annotation file.
        
        XML structure:
            <image>
                <box>
                    <label>transcription</label>
                    <segs>x1,y1,x2,y2,...,x14,y14</segs>
                </box>
            </image>
        
        Args:
            ann_path: Path to annotation file
            
        Returns:
            List of TextInstance objects (28 coords / 14 points each)
        """
        instances = []
        
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            for image in root:
                for box in image:
                    text_elem = box.find("label") 
                    if text_elem is None:
                        text_elem = box[0]
                    segs_elem = box.find("segs")
                    if segs_elem is None:
                        segs_elem = box[1]
                    
                    text = text_elem.text if text_elem.text else ""
                    segs = segs_elem.text if segs_elem.text else ""
                    
                    # Parse 28 coordinates (14 points)
                    pts = segs.strip().split(",")
                    coords = [int(x) for x in pts]
                    
                    if len(coords) == 28:
                        instances.append(TextInstance(
                            polygon=[float(c) for c in coords],
                            text=text,
                            ignore=(text == self.ignore_text),
                        ))
        except Exception as e:
            print(f"Warning: Failed to parse {ann_path}: {e}")
        
        return instances


class CTW1500TxtParser(BaseParser):
    """Parser for CTW1500 TXT annotations (test split)."""
    
    def __init__(self, ignore_text: str = "###"):
        """Initialize CTW1500 TXT parser."""
        super().__init__(ignore_text=ignore_text)
    
    def parse_file(self, ann_path: Path) -> List[TextInstance]:
        """
        Parse CTW1500 TXT annotation file.
        
        Format: 28 coords followed by ####transcription
            695,885,866,888,...,696,1143,####Latin 9
        
        Args:
            ann_path: Path to annotation file
            
        Returns:
            List of TextInstance objects (28 coords / 14 points each)
        """
        instances = []
        
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(",")
                    
                    # First 28 elements are coordinates
                    if len(parts) < 29:
                        continue
                    
                    try:
                        coords = [float(parts[i]) for i in range(28)]
                        
                        # Text starts after #### in position 28
                        text_part = parts[28]
                        if text_part.startswith("####"):
                            text = text_part[4:]
                        else:
                            text = text_part
                        
                        instances.append(TextInstance(
                            polygon=coords,
                            text=text,
                            ignore=(text == self.ignore_text),
                        ))
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"Warning: Failed to parse {ann_path}: {e}")
        
        return instances
