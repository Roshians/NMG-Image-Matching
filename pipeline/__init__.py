"""
NMG Image Matching Pipeline

Classical feature-matching pipeline for stone slab identification.
SIFT/RootSIFT + FLANN + USAC_MAGSAC + confidence scoring.
"""

from .preprocessing import preprocess
from .features import get_features
from .matching import get_good_matches
from .geometric import geometric_verify
from .scoring import score_pair
from .visualization import draw_inlier_matches, save_visualization, print_result_summary
