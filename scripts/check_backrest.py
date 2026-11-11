"""Focused regression: recline the backrest, never rotate the seat/legs."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import backrest

assert hasattr(backrest, 'recline_rgba'), 'Missing backrest-only deformation'
chair = np.zeros((120, 90, 4), np.uint8)
chair[5:65, 30:60] = [180, 50, 30, 255]
chair[65:80, 15:75] = [40, 180, 50, 255]
chair[80:115, 20:28] = [30, 50, 180, 255]
chair[80:115, 65:73] = [30, 50, 180, 255]
original = chair.copy()
for angle in (90, 135, 180):
    out, info = backrest.recline_rgba(chair, angle, hinge=55, left=25, right=75, direction='right')
    ox, oy = info['offset']
    # Map fixed source lower half into expanded output: these pixels must not move.
    assert np.array_equal(out[80-oy:120-oy, -ox:90-ox], chair[80:120]), angle
    if angle == 90:
        assert np.array_equal(out, chair)
    else:
        assert info['moving_pixels'] > 0
        assert info['M_backrest'][0, 1] != 0
    assert np.array_equal(chair, original), 'Source was mutated'
print('PASS: 90/135/180 backrest poses; unchanged seat/legs; no source mutation.')
