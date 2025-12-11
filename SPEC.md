Implementation Specification: Text Detection Dataset Generator
Overview
A dataset-agnostic engine that consumes YOLOv8-Seg text detection datasets, extracts/generates text assets, and synthesizes new training data with diverse augmentations.

1. CLI Interface
bash
python -m engine.main \
    --dataset-dir /path/to/datasets/ \
    --wordlist /path/to/words.txt \
    --fonts-dir /path/to/fonts/ \
    --output-dir /path/to/output/ \
    --per-sample 5 \
    --real-ratio 0.5 \
    --seed 42 \
    --preview 10 \
    --config config.yaml
Required Arguments
Argument	Description
--dataset-dir	Root directory containing YOLOv8-Seg dataset subfolders
--wordlist	Text file with one word per line (English, numbers, codes)
--fonts-dir	Directory containing TTF/OTF font files
--output-dir	Output directory for generated dataset
Optional Arguments
Argument	Default	Description
--per-sample	5	Number of alternative images to generate per input image
--real-ratio	0.5	Ratio of real vs synthetic text assets (0.0–1.0)
--seed	None	Random seed for reproducibility
--preview	0	Generate N preview samples to output/preview/ before full run
--config	None	YAML config file (CLI args override YAML)
--batch-size	8	Batch size for LaMa/RMBG inference
2. Directory Structure
Input Structure
/dataset-dir/
    /dataset_A/
        /images/
            img001.jpg
        /labels/
            img001.txt          # YOLOv8-Seg format: 0 x1 y1 x2 y2 ...
    /dataset_B/
        /images/
        /labels/
 
/wordlist.txt                   # One word per line
/fonts/
    font1.ttf
    font2.otf
Engine Working Directory
/engine/
    /config/
        default.yaml
    /modules/
        __init__.py
        ingest.py               # Dataset scanning & validation
        extract_assets.py       # Real word extraction + RMBG
        synth_generator.py      # Synthetic text rendering
        warp_engine.py          # Geometric deformations
        inpaint_lama.py         # LaMa background cleaning
        compose.py              # Text placement & blending
        augment.py              # Global augmentations
        label_writer.py         # YOLOv8-Seg label generation
        collision.py            # Pixel-level collision detection
        split.py                # Train/val splitting
    /models/
        lama_loader.py          # big-lama checkpoint loader
        rmbg_loader.py          # RMBG-2.0 loader
    main.py
    utils.py
Output Structure
/output/
    /assets/                    # Intermediate assets (can be cached)
        /real_text/
            /dataset_A/
                word_0001.png   # RGBA, alpha-matted
                word_0002.png
            /dataset_B/
        /synth_text/
            /dataset_A/         # Grouped by source dataset for balance
                synth_0001.png
        /backgrounds/
            /dataset_A/
                bg_0001.png     # LaMa-cleaned backgrounds
    /preview/                   # Preview samples (if --preview > 0)
        sample_001.png
        sample_001.txt
    /train/
        /images/
        /labels/
    /val/
        /images/
        /labels/
    generation_report.json      # Summary statistics
3. Module Specifications
3.1 ingest.py — Dataset Ingestion
python
class DatasetIngester:
    def __init__(self, dataset_dir: Path)
    def scan() -> List[DatasetInfo]
    def validate() -> ValidationReport
    
@dataclass
class DatasetInfo:
    name: str
    image_dir: Path
    label_dir: Path
    samples: List[SamplePair]  # (image_path, label_path)
    
@dataclass
class SamplePair:
    image_path: Path
    label_path: Path
    polygons: List[Polygon]    # Parsed from label file
Responsibilities:

Recursively scan dataset-dir for valid dataset subfolders
Parse YOLOv8-Seg labels into polygon objects
Validate image-label pairing
Report statistics per dataset (for balancing)
3.2 extract_assets.py — Real Word Extraction
python
class AssetExtractor:
    def __init__(self, rmbg_model: RMBG, batch_size: int)
    def extract_words(sample: SamplePair) -> List[WordAsset]
    def process_dataset(dataset: DatasetInfo) -> AssetLibrary
    
@dataclass
class WordAsset:
    image: np.ndarray          # RGBA, alpha-matted
    source_dataset: str
    source_image: str
    original_polygon: Polygon
    bbox: Tuple[int, int, int, int]
Pipeline:

Load image
For each polygon in label:
Create binary mask from polygon
Crop bounding box region
Run RMBG-2.0 (batched) → alpha matte
Save as RGBA PNG
Return list of WordAsset
3.3 synth_generator.py — Synthetic Text Rendering
python
class SynthGenerator:
    def __init__(self, fonts_dir: Path, wordlist: List[str])
    def generate(count: int, target_dataset: str) -> List[WordAsset]
    
class TextRenderer:
    def render_word(
        word: str,
        font: Font,
        color: RGB,
        stroke: Optional[StrokeConfig],
        effects: List[Effect]
    ) -> np.ndarray  # RGBA
Text Effects (stacked uniformly):

Stroke/Outline: Random width (1-3px), contrasting color
Drop Shadow: Offset (2-5px), blur (1-3px), opacity (30-60%)
Gradient Fill: Linear/radial, 2-color
Emboss: Light direction, depth
Color Selection:

Analyze target background region
Select color with high contrast (LAB distance > threshold)
3.4 warp_engine.py — Geometric Deformations
python
class WarpEngine:
    def warp(
        image: np.ndarray,
        mask: np.ndarray,
        warp_type: WarpType
    ) -> Tuple[np.ndarray, np.ndarray, Polygon]
    
class WarpType(Enum):
    PERSPECTIVE = "perspective"
    CURVE = "curve"
    ARC = "arc"
    SINE_WAVE = "sine_wave"
    CIRCULAR = "circular"
    SPIRAL = "spiral"
    FREEFORM_POLYGON = "freeform_polygon"
Warp Parameters (moderate, not extreme):

Type	Parameter Range
Perspective	Tilt ±15°, keystoning ±10%
Curve	Bezier curvature 0.1–0.3
Arc	Arc angle 10°–45°
Sine Wave	Amplitude 5–15%, frequency 1–3 cycles
Circular	Arc radius 200–800px
Spiral	Turns 0.1–0.5, tightness 0.8–1.2
Freeform	Random control point displacement 5–15%
Output:

Warped RGBA image
Warped binary mask
New polygon coordinates (for label)
3.5 inpaint_lama.py — Background Cleaning
python
class LamaInpainter:
    def __init__(self, checkpoint: str = "big-lama", device: str = "cuda")
    def inpaint_batch(
        images: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> List[np.ndarray]
    
    def clean_background(
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> np.ndarray
Responsibilities:

Load big-lama checkpoint
Create inpainting masks from all text polygons
Batch inference on GPU
Return clean background (text regions removed)
3.6 compose.py — Composition Engine
python
class Composer:
    def __init__(self, collision_detector: CollisionDetector)
    
    def compose(
        background: np.ndarray,
        original_polygons: List[Polygon],
        real_assets: AssetLibrary,
        synth_assets: AssetLibrary,
        real_ratio: float
    ) -> CompositionResult
    
@dataclass
class CompositionResult:
    image: np.ndarray           # Final composed image (1024x1024)
    polygons: List[Polygon]     # All placed text polygons
    placement_log: List[PlacementInfo]
Placement Strategy:

Phase 1: Fill original polygon locations
For each original polygon position, place a text asset (real or synth per ratio)
Phase 2: Fill remaining space
Find random valid positions
Check pixel-level collision
Place until no valid space remains
Blending (Intermediate tier):

Alpha composite
Gaussian blur at edges (1-2px)
Color matching via LAB histogram transfer
Soft drop shadow (offset 3-5px, blur 2-4px, opacity 40%)
3.7 collision.py — Collision Detection
python
class CollisionDetector:
    def __init__(self, canvas_size: Tuple[int, int])
    
    def add_polygon(polygon: Polygon) -> None
    def check_collision(polygon: Polygon) -> bool
    def find_valid_position(
        asset_mask: np.ndarray,
        max_attempts: int = 100
    ) -> Optional[Tuple[int, int]]
    
    # Internal: maintains a binary occupancy mask
    # Uses pixel-level overlap check
3.8 augment.py — Global Augmentations
python
class Augmentor:
    def __init__(self, probability: float = 0.5)
    def augment(
        image: np.ndarray,
        polygons: List[Polygon]
    ) -> Tuple[np.ndarray, List[Polygon]]
Augmentation Categories (50% application probability per image):

Category	Effects
Photometric	Brightness ±20%, Contrast ±20%, Exposure ±0.5EV, Color temp ±1000K, JPEG artifacts (Q=50-90)
Motion/Sensor	Camera shake (blur kernel 3-7px), Gaussian noise (σ=5-20), Lens distortion (k1=±0.1)
Scene	Fog (density 0.1-0.3), Rain streaks, Lens flare, Shadows
Document	Paper fold, Crumple distortion, Bleed-through, Coffee stains
Note: Augmentations that affect geometry (fold, crumple) must also transform polygon coordinates.

3.9 label_writer.py — Label Generation
python
class LabelWriter:
    def write(
        polygons: List[Polygon],
        image_size: Tuple[int, int],
        output_path: Path
    ) -> None
Output Format (YOLOv8-Seg):

0 x1 y1 x2 y2 x3 y3 ... xn yn
0 x1 y1 x2 y2 ...
Class ID always 0
Coordinates normalized to [0, 1]
One polygon per line
3.10 split.py — Train/Val Splitting
python
class DatasetSplitter:
    def split(
        original_samples: List[Path],
        alternative_samples: List[Path],
        val_ratio: float = 0.2
    ) -> Tuple[List[Path], List[Path]]
Split Logic:

Train: 100% alternatives + 80% originals
Val: 20% originals (held out)
No test set
Shuffle with seed for reproducibility
4. Configuration Schema (YAML)
yaml
# config.yaml
input:
  dataset_dir: /path/to/datasets
  wordlist: /path/to/words.txt
  fonts_dir: /path/to/fonts
output:
  output_dir: /path/to/output
  resolution: [1024, 1024]
generation:
  per_sample: 5
  real_ratio: 0.5
  seed: 42
models:
  lama_checkpoint: big-lama
  rmbg_model: briaai/RMBG-2.0
  batch_size: 8
  device: cuda
text:
  scale_min: 0.03          # 3% of image dimension
  scale_max: 0.25          # 25% of image dimension
  scale_bias: medium       # Bias toward 8-15%
warp:
  types: [perspective, curve, arc, sine_wave, circular, spiral, freeform_polygon]
  intensity: moderate      # Not extreme
blending:
  mode: intermediate       # alpha + color_match + shadow
  edge_blur: 1.5
  shadow_opacity: 0.4
augmentation:
  probability: 0.5
preview:
  count: 10
split:
  val_ratio: 0.2
5. Main Pipeline Flow
┌─────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Parse CLI args + load YAML config                            │
│ 2. Set random seed                                              │
│ 3. Load LaMa model (big-lama) → GPU                             │
│ 4. Load RMBG-2.0 → GPU                                          │
│ 5. Scan dataset-dir → List[DatasetInfo]                         │
│ 6. Load wordlist → List[str]                                    │
│ 7. Load fonts → List[Font]                                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ASSET EXTRACTION                           │
├─────────────────────────────────────────────────────────────────┤
│ For each dataset:                                               │
│   For each sample (batched):                                    │
│     1. Extract word crops from polygons                         │
│     2. Run RMBG-2.0 → alpha-matted PNGs                         │
│     3. Save to /assets/real_text/{dataset}/                     │
│     4. Run LaMa on full image → clean background                │
│     5. Save to /assets/backgrounds/{dataset}/                   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SYNTHETIC ASSET GENERATION                    │
├─────────────────────────────────────────────────────────────────┤
│ For each dataset (to maintain balance):                         │
│   Generate synth assets ≈ count of real assets                  │
│   Save to /assets/synth_text/{dataset}/                         │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREVIEW (if enabled)                       │
├─────────────────────────────────────────────────────────────────┤
│ Generate N preview samples → /output/preview/                   │
│ User inspects manually                                          │
│ Continue? (y/n) or auto-continue                                │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      IMAGE GENERATION                           │
├─────────────────────────────────────────────────────────────────┤
│ For each original sample:                                       │
│   For i in range(per_sample):                                   │
│     1. Select background (prioritize underrepresented)          │
│     2. Resize/crop to 1024x1024                                 │
│     3. Initialize collision detector                            │
│     4. Phase 1: Place text at original polygon positions        │
│        - Select real/synth asset per ratio                      │
│        - Apply random warp                                      │
│        - Apply text effects                                     │
│        - Blend onto background                                  │
│        - Record polygon                                         │
│     5. Phase 2: Fill remaining space                            │
│        - Find valid position (collision check)                  │
│        - Place text until full                                  │
│     6. Apply global augmentations (50% prob)                    │
│     7. Write image + label                                      │
│     8. Update progress bar                                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TRAIN/VAL SPLIT                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Collect all generated alternatives                           │
│ 2. Collect all originals (resized to 1024x1024)                 │
│ 3. Split originals: 80% train, 20% val                          │
│ 4. Train = 100% alternatives + 80% originals                    │
│ 5. Val = 20% originals                                          │
│ 6. Write to /output/train/ and /output/val/                     │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FINALIZATION                             │
├─────────────────────────────────────────────────────────────────┤
│ 1. Generate generation_report.json                              │
│ 2. Print summary to console                                     │
└─────────────────────────────────────────────────────────────────┘
6. Dependencies
txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
scipy>=1.11.0
scikit-image>=0.21.0
tqdm>=4.66.0
pyyaml>=6.0
transformers>=4.35.0          # For RMBG-2.0
huggingface-hub>=0.19.0
einops>=0.7.0                 # LaMa dependency
kornia>=0.7.0                 # Geometric transforms
albumentations>=1.3.0         # Augmentations
7. Generation Report Schema
json
{
  "timestamp": "2024-12-11T08:59:00Z",
  "config": { ... },
  "input_stats": {
    "datasets": 3,
    "total_images": 5000,
    "total_polygons": 45000
  },
  "asset_stats": {
    "real_words_extracted": 45000,
    "synth_words_generated": 45000,
    "backgrounds_cleaned": 5000
  },
  "output_stats": {
    "images_generated": 25000,
    "train_images": 24000,
    "val_images": 1000,
    "total_polygons_placed": 312000,
    "avg_polygons_per_image": 12.5
  },
  "performance": {
    "total_time_seconds": 3600,
    "images_per_second": 6.9
  },
  "errors": {
    "failed_images": 12,
    "error_log": "errors.log"
  }
}
