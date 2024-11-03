import torchvision # Fix issue Unknown builtin op: torchvision::nms
import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image

from custom_densepose.util import HWC3, resize_image_with_pad, common_input_validate, custom_hf_download, DENSEPOSE_MODEL_NAME
from .densepose import DensePoseMaskedColormapResultsVisualizer, _extract_i_from_iuvarr, densepose_chart_predictor_output_to_result_with_confidences

N_PART_LABELS = 24
BG_INDEXS=[0]
TORSO_INDEXS=[1,2]
HAND_INDEXS=[3,4,5,6]
LEG_INDEXS=[7,8,9,10,11,12,13,14]
ARM_INDEXS=[15,16,17,18,19,20,21,22]
HEAD_INDEXS=[23, 24]

class DenseposeDetector:
    def __init__(self, model):
        self.dense_pose_estimation = model
        self.device = "cpu"
        self.result_visualizer = DensePoseMaskedColormapResultsVisualizer(
            alpha=1, 
            data_extractor=_extract_i_from_iuvarr, 
            segm_extractor=_extract_i_from_iuvarr, 
            val_scale = 255.0 / N_PART_LABELS
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=DENSEPOSE_MODEL_NAME, filename="densepose_r50_fpn_dl.torchscript"):
        torchscript_model_path = custom_hf_download(pretrained_model_or_path, filename)
        densepose = torch.jit.load(torchscript_model_path, map_location="cpu")
        return cls(densepose)

    def to(self, device):
        self.dense_pose_estimation.to(device)
        self.device = device
        return self

    def preprocess_parts(self, extractor, pred_boxes, corase_segm, fine_segm, u,v, **kwargs):
        densepose_results = []
        for i in range(len(pred_boxes)):
            densepose_result=extractor(pred_boxes[i:i+1], corase_segm[i:i+1], fine_segm[i:i+1], u[i:i+1], v[i:i+1])
            
            if kwargs.get('head', False):
                head = np.isin(densepose_result[1], HEAD_INDEXS)
                densepose_result[1][head] = 0
            if kwargs.get('background', False):
                background = np.isin(densepose_result[1], BG_INDEXS)
                densepose_result[1][background] = 0
            if kwargs.get('torso', False):
                torso = np.isin(densepose_result[1], TORSO_INDEXS)
                densepose_result[1][torso] = 0
            if kwargs.get('hand', False):
                hand = np.isin(densepose_result[1], HAND_INDEXS)
                densepose_result[1][hand] = 0
            if kwargs.get('leg', False):
                leg = np.isin(densepose_result[1], LEG_INDEXS)
                densepose_result[1][leg] = 0
            if kwargs.get('arm', False):
                arm = np.isin(densepose_result[1], ARM_INDEXS)
                densepose_result[1][arm] = 0
                
            densepose_results.append(densepose_result)
        return densepose_results
    
    def __call__(self, input_image, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", cmap="viridis", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        H, W  = input_image.shape[:2]

        hint_image_canvas = np.zeros([H, W], dtype=np.uint8)
        hint_image_canvas = np.tile(hint_image_canvas[:, :, np.newaxis], [1, 1, 3])

        input_image = rearrange(torch.from_numpy(input_image).to(self.device), 'h w c -> c h w')

        pred_boxes, corase_segm, fine_segm, u, v = self.dense_pose_estimation(input_image)

        extractor = densepose_chart_predictor_output_to_result_with_confidences
        densepose_results = self.preprocess_parts(extractor, pred_boxes, corase_segm, fine_segm, u,v, **kwargs)
        # densepose_results = [extractor(pred_boxes[i:i+1], corase_segm[i:i+1], fine_segm[i:i+1], u[i:i+1], v[i:i+1]) for i in range(len(pred_boxes))]

        if cmap=="viridis":
            self.result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_VIRIDIS
            hint_image = self.result_visualizer.visualize(hint_image_canvas, densepose_results)
            hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)
            hint_image[:, :, 0][hint_image[:, :, 0] == 0] = 68
            hint_image[:, :, 1][hint_image[:, :, 1] == 0] = 1
            hint_image[:, :, 2][hint_image[:, :, 2] == 0] = 84
        else:
            self.result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_PARULA
            hint_image = self.result_visualizer.visualize(hint_image_canvas, densepose_results)
            hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)

        detected_map = remove_pad(HWC3(hint_image))
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        return detected_map
