from .BasicDensepose import DensePose_Preprocessor
from .utils import here
from pathlib import Path
import sys

# sys.path.append(str(Path(here, "custom_densepose").resolve()))

NODE_CLASS_MAPPINGS = {
    "DensePosePreprocessor": DensePose_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DensePosePreprocessor": "DensePose Estimator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']