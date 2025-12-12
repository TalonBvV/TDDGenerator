"""
ICDAR Text Detection Annotation Parser

Parses ICDAR format annotations:
    x1,y1,x2,y2,x3,y3,x4,y4,transcription

Used for: ICDAR2013, ICDAR2015, SROIE
"""

from pathlib import Path
from typing import List

from .base import BaseParser, TextInstance


class ICDARParser(BaseParser):
    """Parser for ICDAR text detection format."""
    
    def __init__(
        self,
        separator: str = ",",
        encoding: str = "utf-8-sig",
        ignore_text: str = "###",
    ):
        """
        Initialize ICDAR parser.
        
        Args:
            separator: Field separator (default: ",")
            encoding: File encoding (default: "utf-8-sig" for BOM handling)
            ignore_text: Text marker for ignored instances
        """
        super().__init__(ignore_text=ignore_text)
        self.separator = separator
        self.encoding = encoding
    
    def parse_file(self, ann_path: Path) -> List[TextInstance]:
        """
        Parse ICDAR annotation file.
        
        Format: x1,y1,x2,y2,x3,y3,x4,y4,transcription
        
        Args:
            ann_path: Path to annotation file
            
        Returns:
            List of TextInstance objects
        """
        instances = []
        
        try:
            with open(ann_path, "r", encoding=self.encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    instance = self._parse_line(line)
                    if instance:
                        instances.append(instance)
        except Exception as e:
            print(f"Warning: Failed to parse {ann_path}: {e}")
        
        return instances
    
    def _parse_line(self, line: str) -> TextInstance:
        """Parse a single annotation line."""
        parts = line.split(self.separator)
        
        if len(parts) < 9:
            # Try to handle cases where transcription contains separator
            # Assume at least 8 coordinates
            if len(parts) < 8:
                return None
        
        try:
            # Extract 8 coordinates (4 points)
            coords = [float(parts[i].strip()) for i in range(8)]
            
            # Remaining parts form the transcription
            text = self.separator.join(parts[8:]).strip()
            
            # Remove quotes if present
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            
            return TextInstance(
                polygon=coords,
                text=text,
                ignore=(text == self.ignore_text or text.startswith("###")),
            )
        except (ValueError, IndexError) as e:
            return None


class ICDARXYXYParser(BaseParser):
    """Parser for ICDAR xyxy format (e.g., ICDAR2013).
    
    Format: x1 y1 x2 y2 transcription (space-separated, bounding box)
    """
    
    def __init__(
        self,
        separator: str = " ",
        encoding: str = "utf-8-sig",
        ignore_text: str = "###",
    ):
        super().__init__(ignore_text=ignore_text)
        self.separator = separator
        self.encoding = encoding
    
    def parse_file(self, ann_path: Path) -> List[TextInstance]:
        """Parse ICDAR xyxy annotation file."""
        instances = []
        
        try:
            with open(ann_path, "r", encoding=self.encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    instance = self._parse_line(line)
                    if instance:
                        instances.append(instance)
        except Exception as e:
            print(f"Warning: Failed to parse {ann_path}: {e}")
        
        return instances
    
    def _parse_line(self, line: str) -> TextInstance:
        """Parse a single annotation line."""
        # Remove quotes
        line = line.replace('"', '').replace(',', '')
        parts = line.split(self.separator)
        
        if len(parts) < 5:
            return None
        
        try:
            # x1, y1, x2, y2 bounding box
            x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            
            # Convert to 4-point polygon (clockwise from top-left)
            coords = [x1, y1, x2, y1, x2, y2, x1, y2]
            
            # Text is remaining parts
            text = self.separator.join(parts[4:]).strip()
            
            return TextInstance(
                polygon=coords,
                text=text,
                ignore=(text == self.ignore_text or text.startswith("###")),
            )
        except (ValueError, IndexError):
            return None
