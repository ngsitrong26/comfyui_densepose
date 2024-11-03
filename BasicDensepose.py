#https://github.com/Fannovel16/comfyui_controlnet_aux
from .utils import common_annotator_call, INPUT, define_preprocessor_inputs
import comfy.model_management as model_management

from .utils import here
from pathlib import Path
import sys

# sys.path.append(str(Path(here, "custom_densepose").resolve()))

class DensePose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            model=INPUT.COMBO(["densepose_r50_fpn_dl.torchscript", "densepose_r101_fpn_dl.torchscript"]),
            cmap=INPUT.COMBO(["Viridis (MagicAnimate)", "Parula (CivitAI)"]),
            resolution=INPUT.RESOLUTION(),
            background=INPUT.BOOLEAN(True),
            torso=INPUT.BOOLEAN(True),
            hand=INPUT.BOOLEAN(False),
            foot=INPUT.BOOLEAN(False),
            leg=INPUT.BOOLEAN(True),
            arm=INPUT.BOOLEAN(True),
            head=INPUT.BOOLEAN(False),
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "Custom/DensePose"

    def execute(self, image, model="densepose_r50_fpn_dl.torchscript", cmap="Viridis (MagicAnimate)", resolution=512,
               background=True, torso=True, hand=False, foot=False, leg=True, arm=True, head=False):
        from .custom_densepose.densepose import DenseposeDetector
        model = DenseposeDetector \
                    .from_pretrained(filename=model) \
                    .to(model_management.get_torch_device())
        return (common_annotator_call(model, image, cmap="viridis" if "Viridis" in cmap else "parula", resolution=resolution, background=background, torso=torso, hand=hand, foot=foot, leg=leg, arm=arm, head=head), )


