import re


def extract_bbox(response: str):
    """Extract bounding box from a BBOX(x1, y1, x2, y2) or [x1, y1, x2, y2] format response.
    Returns [[x1, y1], [x2, y2]] as list of [int, int] pairs.
    """
    # Try to match BBOX(...) format
    bbox_match = re.search(
        r"BBOX\s*\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)",
        response, re.IGNORECASE
    )
    if bbox_match:
        coords = [float(bbox_match.group(i)) for i in range(1, 5)]
        return [[int(coords[0]), int(coords[1])], [int(coords[2]), int(coords[3])]]

    # Try to match [x1, y1, x2, y2] format
    bracket_match = re.search(
        r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]",
        response
    )
    if bracket_match:
        coords = [float(bracket_match.group(i)) for i in range(1, 5)]
        return [[int(coords[0]), int(coords[1])], [int(coords[2]), int(coords[3])]]

    raise ValueError(f"Cannot extract bbox from response: {response}")


def pred_2_point(response: str):
    """Extract a point coordinate from a response that doesn't contain 'box'.
    Returns [x, y] as list of ints.
    """
    # Try to match (x, y) or (x,y) format
    paren_match = re.search(
        r"\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)",
        response
    )
    if paren_match:
        return [int(float(paren_match.group(1))), int(float(paren_match.group(2)))]

    # Try to match [x, y] format
    bracket_match = re.search(
        r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]",
        response
    )
    if bracket_match:
        return [int(float(bracket_match.group(1))), int(float(bracket_match.group(2)))]

    # Try to match plain x, y format
    plain_match = re.search(
        r"(-?\d*\.?\d+)\s*[, ]\s*(-?\d*\.?\d+)",
        response
    )
    if plain_match:
        return [int(float(plain_match.group(1))), int(float(plain_match.group(2)))]

    raise ValueError(f"Cannot extract point from response: {response}")
