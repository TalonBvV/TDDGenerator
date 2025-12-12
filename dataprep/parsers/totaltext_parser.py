"""
TotalText Annotation Parser

Parses TotalText format annotations:
    x: [[x1 x2 x3 ... xn]], y: [[y1 y2 y3 ... yn]], 
    ornt: [u'c'], transcriptions: [u'transcription']
"""

import re
from pathlib import Path
from typing import List, Tuple

import yaml

from .base import BaseParser, TextInstance


class TotalTextParser(BaseParser):
    """Parser for TotalText annotation format."""
    
    def __init__(self, ignore_text: str = "#"):
        """Initialize TotalText parser."""
        super().__init__(ignore_text=ignore_text)
    
    def parse_file(self, ann_path: Path) -> List[TextInstance]:
        """
        Parse TotalText annotation file.
        
        Args:
            ann_path: Path to annotation file
            
        Returns:
            List of TextInstance objects
        """
        instances = []
        
        try:
            for poly, text in self._load_annotations(ann_path):
                instances.append(TextInstance(
                    polygon=poly,
                    text=text,
                    ignore=(text == self.ignore_text),
                ))
        except Exception as e:
            print(f"Warning: Failed to parse {ann_path}: {e}")
        
        return instances
    
    def _load_annotations(self, file_path: Path) -> List[Tuple[List[float], str]]:
        """
        Load annotations from TotalText file.
        
        TotalText annotations may span multiple lines, so we need special handling.
        """
        results = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        tmp_line = ""
        for idx, line in enumerate(lines):
            line = line.strip()
            if idx == 0:
                tmp_line = line
                continue
            
            if not line.startswith("x:"):
                tmp_line += " " + line
                continue
            
            # Process complete annotation
            result = self._parse_annotation(tmp_line)
            if result:
                results.append(result)
            tmp_line = line
        
        # Process last annotation
        if tmp_line:
            result = self._parse_annotation(tmp_line)
            if result:
                results.append(result)
        
        return results
    
    def _parse_annotation(self, line: str) -> Tuple[List[float], str]:
        """Parse a single TotalText annotation line."""
        try:
            # Convert to dict-like format
            line = "{" + line.replace("[[", "[").replace("]]", "]") + "}"
            
            # Fix spacing issues in coordinates
            ann_dict = re.sub(r"([0-9]) +([0-9])", r"\1,\2", line)
            ann_dict = re.sub(r"([0-9]) +([ 0-9])", r"\1,\2", ann_dict)
            ann_dict = re.sub(r"([0-9]) -([0-9])", r"\1,-\2", ann_dict)
            ann_dict = ann_dict.replace("[u',']", "[u'#']")
            
            # Parse as YAML
            ann_dict = yaml.safe_load(ann_dict)
            
            # Extract polygon
            xs = ann_dict["x"]
            ys = ann_dict["y"]
            
            poly = []
            for x, y in zip(xs, ys):
                poly.append(float(x))
                poly.append(float(y))
            
            # Extract text
            transcriptions = ann_dict.get("transcriptions", [])
            if not transcriptions:
                text = "#"
            else:
                word = transcriptions[0]
                if len(transcriptions) > 1:
                    for ann_word in transcriptions[1:]:
                        word += "," + ann_word
                text = str(eval(word)) if word else "#"
            
            return poly, text
            
        except Exception as e:
            return None
