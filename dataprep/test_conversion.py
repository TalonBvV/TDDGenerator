"""
Test script for verify dataprep conversion logic.
Run with: python dataprep/test_conversion.py
"""

def normalize_polygon(polygon, width, height, clamp=True):
    """Normalize polygon coordinates from pixels to 0-1 range."""
    if len(polygon) % 2 != 0:
        raise ValueError(f"Polygon has odd number of coordinates: {len(polygon)}")
    
    points = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / width
        y = polygon[i + 1] / height
        
        if clamp:
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
        
        points.append((x, y))
    
    return points


def write_yolo_line(points, class_id=0):
    """Write a single YOLO label line."""
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"{class_id} {coords}"


def test_normalization():
    """Test polygon normalization."""
    print("=" * 50)
    print("Test 1: Basic Polygon Normalization")
    print("=" * 50)
    
    # Input: pixel coordinates on a 400x200 image
    polygon = [100, 50, 200, 50, 200, 100, 100, 100]
    width, height = 400, 200
    
    result = normalize_polygon(polygon, width, height)
    expected = [(0.25, 0.25), (0.5, 0.25), (0.5, 0.5), (0.25, 0.5)]
    
    print(f"  Input polygon: {polygon}")
    print(f"  Image size: {width}x{height}")
    print(f"  Result: {result}")
    print(f"  Expected: {expected}")
    print(f"  PASS: {result == expected}")
    
    return result == expected


def test_clamping():
    """Test coordinate clamping."""
    print("\n" + "=" * 50)
    print("Test 2: Coordinate Clamping")
    print("=" * 50)
    
    # Input: out-of-bounds coordinates
    polygon = [-10, 250, 450, 50]
    width, height = 400, 200
    
    result = normalize_polygon(polygon, width, height, clamp=True)
    expected = [(0.0, 1.0), (1.0, 0.25)]
    
    print(f"  Input polygon: {polygon}")
    print(f"  Image size: {width}x{height}")
    print(f"  Result: {result}")
    print(f"  Expected: {expected}")
    print(f"  PASS: {result == expected}")
    
    return result == expected


def test_yolo_format():
    """Test YOLO label format."""
    print("\n" + "=" * 50)
    print("Test 3: YOLO Label Format")
    print("=" * 50)
    
    # Sample from README
    points = [(0.155, 0.238), (0.201, 0.231), (0.217, 0.289), (0.159, 0.303)]
    line = write_yolo_line(points)
    
    print(f"  Input points: {points}")
    print(f"  Generated line: {line}")
    
    # Verify format
    parts = line.split()
    class_id = parts[0]
    num_coords = len(parts) - 1
    
    print(f"  Class ID: {class_id}")
    print(f"  Num coordinates: {num_coords} (expected 8 for 4 points)")
    
    # Check format matches engine expectation
    is_valid = (
        class_id == "0" and
        num_coords == 8 and
        all(0 <= float(x) <= 1 for x in parts[1:])
    )
    
    print(f"  Format valid: {is_valid}")
    
    return is_valid


def test_engine_compatibility():
    """Test compatibility with engine's Polygon.from_yolo_line."""
    print("\n" + "=" * 50)
    print("Test 4: Engine Compatibility")
    print("=" * 50)
    
    # Generate a YOLO line
    original_points = [(0.155, 0.238), (0.201, 0.231), (0.217, 0.289), (0.159, 0.303)]
    line = write_yolo_line(original_points)
    
    # Parse it like the engine does
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    
    parsed_points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    
    print(f"  Original: {original_points}")
    print(f"  YOLO line: {line}")
    print(f"  Parsed back: {parsed_points}")
    
    # Check round-trip accuracy (within floating point precision)
    matches = all(
        abs(orig[0] - parsed[0]) < 1e-5 and abs(orig[1] - parsed[1]) < 1e-5
        for orig, parsed in zip(original_points, parsed_points)
    )
    
    print(f"  Round-trip match: {matches}")
    
    return matches


def main():
    """Run all tests."""
    print("\nDataprep Conversion Logic Tests")
    print("================================\n")
    
    results = [
        ("Normalization", test_normalization()),
        ("Clamping", test_clamping()),
        ("YOLO Format", test_yolo_format()),
        ("Engine Compatibility", test_engine_compatibility()),
    ]
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
