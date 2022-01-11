from typing import List, Tuple

def _parse_color(color: str) -> Tuple[int, int, int]:
    if color.startswith("#"):
        color = color[1:]

    r = int(color[:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:], 16)

    return (r, g, b)

def lerp(a: int, b: int, ratio: float) -> int:
    return a + round((b - a) * ratio)

def shade(start: str, stop: str, steps: int) -> List[float]:
    shades = []
    start = _parse_color(start)
    stop = _parse_color(stop)

    for i in range(steps):
        ratio = (1.0 * i) / (steps - 1)
        color = [lerp(a, b, ratio) for a, b in zip(start, stop)]
        shades.append(f"#{color[0]:0>2x}{color[1]:0>2x}{color[2]:0>2x}")

    return shades