"""
Base Parser

Abstract base class for annotation parsers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class TextInstance:
    """A single text instance with polygon and transcription."""
    polygon: List[float]  # Flat list of x,y coordinates
    text: str
    ignore: bool = False
    
    @property
    def num_points(self) -> int:
        """Number of polygon points."""
        return len(self.polygon) // 2
    
    def to_normalized(self, width: int, height: int, clamp: bool = True) -> List[Tuple[float, float]]:
        """Convert to normalized (0-1) coordinates."""
        points = []
        for i in range(0, len(self.polygon), 2):
            x = self.polygon[i] / width
            y = self.polygon[i + 1] / height
            if clamp:
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
            points.append((x, y))
        return points


class BaseParser(ABC):
    """Abstract base class for annotation parsers."""
    
    def __init__(self, ignore_text: str = "###"):
        """
        Initialize parser.
        
        Args:
            ignore_text: Text marker for ignored instances (default: "###")
        """
        self.ignore_text = ignore_text
    
    @abstractmethod
    def parse_file(self, ann_path: Path) -> List[TextInstance]:
        """
        Parse a single annotation file.
        
        Args:
            ann_path: Path to annotation file
            
        Returns:
            List of TextInstance objects
        """
        pass
    
    def parse_files(self, ann_paths: List[Path]) -> List[List[TextInstance]]:
        """
        Parse multiple annotation files.
        
        Args:
            ann_paths: List of annotation file paths
            
        Returns:
            List of lists of TextInstance objects
        """
        return [self.parse_file(p) for p in ann_paths]
