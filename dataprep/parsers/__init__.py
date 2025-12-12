"""
Annotation Parsers

Format-specific parsers for reading text detection annotations.
"""

from .base import BaseParser, TextInstance
from .icdar_parser import ICDARParser, ICDARXYXYParser
from .totaltext_parser import TotalTextParser
from .ctw1500_parser import CTW1500XMLParser, CTW1500TxtParser
from .funsd_parser import FUNSDParser
from .svt_parser import SVTParser
from .coco_parser import COCOParser
from .wildreceipt_parser import WildreceiptParser

__all__ = [
    "BaseParser",
    "TextInstance",
    "ICDARParser",
    "ICDARXYXYParser",
    "TotalTextParser",
    "CTW1500XMLParser",
    "CTW1500TxtParser",
    "FUNSDParser",
    "SVTParser",
    "COCOParser",
    "WildreceiptParser",
]

# Parser registry
PARSERS = {
    "icdar": ICDARParser,
    "icdar_xyxy": ICDARXYXYParser,
    "totaltext": TotalTextParser,
    "ctw1500_xml": CTW1500XMLParser,
    "ctw1500_txt": CTW1500TxtParser,
    "funsd": FUNSDParser,
    "svt": SVTParser,
    "coco": COCOParser,
    "wildreceipt": WildreceiptParser,
    "naf": FUNSDParser,  # NAF uses similar JSON format to FUNSD
    "synthtext": COCOParser,  # Placeholder - synthtext uses .mat format
}


def get_parser(parser_type: str) -> type:
    """Get parser class by type name."""
    if parser_type not in PARSERS:
        raise ValueError(f"Unknown parser type: {parser_type}. Available: {list(PARSERS.keys())}")
    return PARSERS[parser_type]
