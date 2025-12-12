"""
Dataset Configuration

Contains download URLs, MD5 checksums, and parsing configuration for each dataset.
Extracted from MMOCR dataset_zoo configs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FileConfig:
    """Configuration for a single downloadable file."""
    url: str
    save_name: str
    md5: Optional[str] = None
    mapping: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class SplitConfig:
    """Configuration for a dataset split (train/test)."""
    files: List[FileConfig]
    img_pattern: str  # Regex to match image files
    ann_pattern: str  # Regex to get annotation from image name
    parser_type: str  # Parser to use


@dataclass
class DatasetConfig:
    """Full configuration for a dataset."""
    name: str
    train: Optional[SplitConfig] = None
    test: Optional[SplitConfig] = None
    size_estimate_gb: float = 1.0


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

ICDAR2013 = DatasetConfig(
    name="icdar2013",
    size_estimate_gb=0.5,
    train=SplitConfig(
        files=[
            FileConfig(
                url="https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task12_Images.zip",
                save_name="ic13_train_images.zip",
                md5="a443b9649fda4229c9bc52751bad08fb",
                mapping=[("Challenge2_Training_Task12_Images", "images/train")],
            ),
            FileConfig(
                url="https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task1_GT.zip",
                save_name="ic13_train_gt.zip",
                md5="f3a425284a66cd67f455d389c972cce4",
                mapping=[("", "annotations/train")],
            ),
        ],
        img_pattern=r"(\w+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"gt_{}.txt",
        parser_type="icdar_xyxy",
    ),
    test=SplitConfig(
        files=[
            FileConfig(
                url="https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task12_Images.zip",
                save_name="ic13_test_images.zip",
                md5="af2e9f070c4c6a1c7bdb7b36bacf23e3",
                mapping=[("Challenge2_Test_Task12_Images", "images/test")],
            ),
            FileConfig(
                url="https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task1_GT.zip",
                save_name="ic13_test_gt.zip",
                md5="3191c34cd6ac28b60f5a7db7030190fb",
                mapping=[("", "annotations/test")],
            ),
        ],
        img_pattern=r"(\w+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"gt_{}.txt",
        parser_type="icdar_xyxy",
    ),
)

ICDAR2015 = DatasetConfig(
    name="icdar2015",
    size_estimate_gb=0.5,
    train=SplitConfig(
        files=[
            FileConfig(
                url="https://rrc.cvc.uab.es/downloads/ch4_training_images.zip",
                save_name="ic15_train_images.zip",
                md5="c51cbace155dcc4d98c8dd19d378f30d",
                mapping=[("ch4_training_images", "images/train")],
            ),
            FileConfig(
                url="https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip",
                save_name="ic15_train_gt.zip",
                md5="3bfaf1988960909014f7987d2343060b",
                mapping=[("", "annotations/train")],
            ),
        ],
        img_pattern=r"img_(\d+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"gt_img_{}.txt",
        parser_type="icdar",
    ),
    test=SplitConfig(
        files=[
            FileConfig(
                url="https://rrc.cvc.uab.es/downloads/ch4_test_images.zip",
                save_name="ic15_test_images.zip",
                md5="97e4c1ddcf074ffcc75feff2b63c35dd",
                mapping=[("ch4_test_images", "images/test")],
            ),
            FileConfig(
                url="https://rrc.cvc.uab.es/downloads/Challenge4_Test_Task4_GT.zip",
                save_name="ic15_test_gt.zip",
                md5="8bce173b06d164b98c357b0eb96ef430",
                mapping=[("", "annotations/test")],
            ),
        ],
        img_pattern=r"img_(\d+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"gt_img_{}.txt",
        parser_type="icdar",
    ),
)

TOTALTEXT = DatasetConfig(
    name="totaltext",
    size_estimate_gb=0.5,
    train=SplitConfig(
        files=[
            FileConfig(
                url="https://universityofadelaide.box.com/shared/static/8xro7hnvb0sqw5e5rxm73tryc59j6s43.zip",
                save_name="totaltext.zip",
                md5="5b56d71a4005a333cf200ff35ce87f75",
                mapping=[("totaltext/Images/Train", "images/train")],
            ),
            FileConfig(
                url="https://universityofadelaide.box.com/shared/static/2vmpvjb48pcrszeegx2eznzc4izan4zf.zip",
                save_name="txt_format.zip",
                md5="53377a83420b4a0244304467512134e8",
                mapping=[("txt_format/Train", "annotations/train")],
            ),
        ],
        img_pattern=r"img(\d+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"poly_gt_img{}.txt",
        parser_type="totaltext",
    ),
    test=SplitConfig(
        files=[
            FileConfig(
                url="https://universityofadelaide.box.com/shared/static/8xro7hnvb0sqw5e5rxm73tryc59j6s43.zip",
                save_name="totaltext.zip",
                md5="5b56d71a4005a333cf200ff35ce87f75",
                mapping=[("totaltext/Images/Test", "images/test")],
            ),
            FileConfig(
                url="https://universityofadelaide.box.com/shared/static/2vmpvjb48pcrszeegx2eznzc4izan4zf.zip",
                save_name="txt_format.zip",
                md5="53377a83420b4a0244304467512134e8",
                mapping=[("txt_format/Test", "annotations/test")],
            ),
        ],
        img_pattern=r"img(\d+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"poly_gt_img{}.txt",
        parser_type="totaltext",
    ),
)

CTW1500 = DatasetConfig(
    name="ctw1500",
    size_estimate_gb=0.8,
    train=SplitConfig(
        files=[
            FileConfig(
                url="https://universityofadelaide.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip",
                save_name="ctw1500_train_images.zip",
                md5="f1453464b764343040644464d5c0c4fa",
                mapping=[("ctw1500_train_images/train_images", "images/train")],
            ),
            FileConfig(
                url="https://universityofadelaide.box.com/shared/static/jikuazluzyj4lq6umzei7m2ppmt3afyw.zip",
                save_name="ctw1500_train_labels.zip",
                md5="d9ba721b25be95c2d78aeb54f812a5b1",
                mapping=[("ctw1500_train_labels/ctw1500_train_labels", "annotations/train")],
            ),
        ],
        img_pattern=r"(\d{4})\.(jpg|JPG|png|PNG)",
        ann_pattern=r"{}.xml",
        parser_type="ctw1500_xml",
    ),
    test=SplitConfig(
        files=[
            FileConfig(
                url="https://universityofadelaide.box.com/shared/static/t4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip",
                save_name="ctw1500_test_images.zip",
                md5="79103fd77dfdd2c70ae6feb3a2fb4530",
                mapping=[("ctw1500_test_images/test_images", "images/test")],
            ),
            FileConfig(
                url="https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/download",
                save_name="ctw1500_test_labels.zip",
                md5="7f650933a30cf1bcdbb7874e4962a52b",
                mapping=[("ctw1500_test_labels", "annotations/test")],
            ),
        ],
        img_pattern=r"(\d{4})\.(jpg|JPG|png|PNG)",
        ann_pattern=r"000{}.txt",
        parser_type="ctw1500_txt",
    ),
)

SROIE = DatasetConfig(
    name="sroie",
    size_estimate_gb=0.1,
    train=SplitConfig(
        files=[
            FileConfig(
                url="https://rrc.cvc.uab.es/downloads/SROIE2019.zip",
                save_name="sroie.zip",
                md5=None,
                mapping=[
                    ("SROIE2019/train/img", "images/train"),
                    ("SROIE2019/train/gt", "annotations/train"),
                ],
            ),
        ],
        img_pattern=r"(.+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"{}.txt",
        parser_type="icdar",
    ),
    test=SplitConfig(
        files=[
            FileConfig(
                url="https://rrc.cvc.uab.es/downloads/SROIE2019.zip",
                save_name="sroie.zip",
                md5=None,
                mapping=[("SROIE2019/test/img", "images/test")],
            ),
        ],
        img_pattern=r"(.+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"{}.txt",
        parser_type="icdar",
    ),
)

FUNSD = DatasetConfig(
    name="funsd",
    size_estimate_gb=0.1,
    train=SplitConfig(
        files=[
            FileConfig(
                url="https://guillaumejaume.github.io/FUNSD/dataset.zip",
                save_name="funsd.zip",
                md5="e05de47de238aa343bf55d8807d659a9",
                mapping=[
                    ("funsd/dataset/training_data/images", "images/train"),
                    ("funsd/dataset/training_data/annotations", "annotations/train"),
                ],
            ),
        ],
        img_pattern=r"(\w+)\.(png|PNG)",
        ann_pattern=r"{}.json",
        parser_type="funsd",
    ),
    test=SplitConfig(
        files=[
            FileConfig(
                url="https://guillaumejaume.github.io/FUNSD/dataset.zip",
                save_name="funsd.zip",
                md5="e05de47de238aa343bf55d8807d659a9",
                mapping=[
                    ("funsd/dataset/testing_data/images", "images/test"),
                    ("funsd/dataset/testing_data/annotations", "annotations/test"),
                ],
            ),
        ],
        img_pattern=r"(\w+)\.(png|PNG)",
        ann_pattern=r"{}.json",
        parser_type="funsd",
    ),
)

SVT = DatasetConfig(
    name="svt",
    size_estimate_gb=0.1,
    train=SplitConfig(
        files=[
            FileConfig(
                url="http://www.iapr-tc11.org/dataset/SVT/svt.zip",
                save_name="svt.zip",
                md5="42d19160010d990ae6223b14f45eff88",
                mapping=[
                    ("svt/svt1/train.xml", "annotations/train.xml"),
                    ("svt/svt1/img", "images/train"),
                ],
            ),
        ],
        img_pattern=r"(\w+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"train.xml",  # Single XML file
        parser_type="svt",
    ),
    test=SplitConfig(
        files=[
            FileConfig(
                url="http://www.iapr-tc11.org/dataset/SVT/svt.zip",
                save_name="svt.zip",
                md5="42d19160010d990ae6223b14f45eff88",
                mapping=[
                    ("svt/svt1/test.xml", "annotations/test.xml"),
                    ("svt/svt1/img", "images/test"),
                ],
            ),
        ],
        img_pattern=r"(\w+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"test.xml",  # Single XML file
        parser_type="svt",
    ),
)

NAF = DatasetConfig(
    name="naf",
    size_estimate_gb=0.3,
    train=SplitConfig(
        files=[
            FileConfig(
                url="https://github.com/herobd/NAF_dataset/releases/download/v1.0/labeled_images.tar.gz",
                save_name="naf_images.tar.gz",
                md5="6521cdc25c313a1f2928a16a77ad8f29",
                mapping=[("naf_images/labeled_images", "images/train")],
            ),
            FileConfig(
                url="https://github.com/herobd/NAF_dataset/archive/refs/heads/master.zip",
                save_name="naf_anno.zip",
                md5="abf5af6266cc527d772231751bc884b3",
                mapping=[("naf_anno/NAF_dataset-master/groups", "annotations/train")],
            ),
        ],
        img_pattern=r"(.+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"{}.json",
        parser_type="naf",
    ),
    test=None,
)

WILDRECEIPT = DatasetConfig(
    name="wildreceipt",
    size_estimate_gb=0.2,
    train=SplitConfig(
        files=[
            FileConfig(
                url="https://download.openmmlab.com/mmocr/data/wildreceipt.tar",
                save_name="wildreceipt.tar",
                md5="2a2c4a1b4777fb4fe185011e17ad46ae",
                mapping=[
                    ("wildreceipt/wildreceipt/image_files", "images/train"),
                    ("wildreceipt/wildreceipt/train.txt", "annotations/train.txt"),
                ],
            ),
        ],
        img_pattern=r"(.+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"train.txt",
        parser_type="wildreceipt",
    ),
    test=SplitConfig(
        files=[
            FileConfig(
                url="https://download.openmmlab.com/mmocr/data/wildreceipt.tar",
                save_name="wildreceipt.tar",
                md5="2a2c4a1b4777fb4fe185011e17ad46ae",
                mapping=[
                    ("wildreceipt/wildreceipt/image_files", "images/test"),
                    ("wildreceipt/wildreceipt/test.txt", "annotations/test.txt"),
                ],
            ),
        ],
        img_pattern=r"(.+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"test.txt",
        parser_type="wildreceipt",
    ),
)

TEXTOCR = DatasetConfig(
    name="textocr",
    size_estimate_gb=7.0,
    train=SplitConfig(
        files=[
            FileConfig(
                url="https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
                save_name="textocr_images.zip",
                md5="d12dd8098899044e4ae1af34db7ecfef",
                mapping=[("textocr_images/train_images", "images/train")],
            ),
            FileConfig(
                url="https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json",
                save_name="textocr_train.json",
                md5="0f8ba1beefd2ca4d08a4f82bcbe6cfb4",
                mapping=[("textocr_train.json", "annotations/train.json")],
            ),
        ],
        img_pattern=r"(.+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"train.json",
        parser_type="coco",
    ),
    test=SplitConfig(
        files=[
            FileConfig(
                url="https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
                save_name="textocr_images.zip",
                md5="d12dd8098899044e4ae1af34db7ecfef",
                mapping=[("textocr_images/train_images", "images/test")],
            ),
            FileConfig(
                url="https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json",
                save_name="textocr_val.json",
                md5="fb151383ea7b3c530cde9ef0d5c08347",
                mapping=[("textocr_val.json", "annotations/test.json")],
            ),
        ],
        img_pattern=r"(.+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"test.json",
        parser_type="coco",
    ),
)

COCOTEXTV2 = DatasetConfig(
    name="cocotextv2",
    size_estimate_gb=13.0,
    train=SplitConfig(
        files=[
            FileConfig(
                url="http://images.cocodataset.org/zips/train2014.zip",
                save_name="coco_train2014.zip",
                md5="0da8c0bd3d6becc4dcb32757491aca88",
                mapping=[("coco_train2014/train2014", "images/train")],
            ),
            FileConfig(
                url="https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip",
                save_name="cocotextv2_anno.zip",
                md5="5e39f7d6f2f11324c6451e63523c440c",
                mapping=[("cocotextv2_anno/cocotext.v2.json", "annotations/train.json")],
            ),
        ],
        img_pattern=r"(.+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"train.json",
        parser_type="coco",
    ),
    test=None,
)

# SynthText uses magnet links - not directly downloadable
SYNTHTEXT = DatasetConfig(
    name="synthtext",
    size_estimate_gb=40.0,
    train=SplitConfig(
        files=[
            FileConfig(
                # Magnet link - requires manual download
                url="https://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip",
                save_name="SynthText.zip",
                md5="8ae0309c80ff882f9d6ba5ea62cdb556",
                mapping=[("SynthText/SynthText", "images/train")],
            ),
        ],
        img_pattern=r"(.+)\.(jpg|JPG|png|PNG)",
        ann_pattern=r"gt.mat",
        parser_type="synthtext",
    ),
    test=None,
)

# Registry of all datasets
DATASETS: Dict[str, DatasetConfig] = {
    "icdar2013": ICDAR2013,
    "icdar2015": ICDAR2015,
    "totaltext": TOTALTEXT,
    "ctw1500": CTW1500,
    "sroie": SROIE,
    "funsd": FUNSD,
    "svt": SVT,
    "naf": NAF,
    "wildreceipt": WILDRECEIPT,
    "textocr": TEXTOCR,
    "cocotextv2": COCOTEXTV2,
    "synthtext": SYNTHTEXT,
}


def get_dataset_config(name: str) -> Optional[DatasetConfig]:
    """Get configuration for a dataset by name."""
    return DATASETS.get(name.lower())


def get_available_datasets() -> List[str]:
    """Get list of available dataset names."""
    return list(DATASETS.keys())
