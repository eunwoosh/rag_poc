import sys
import os
import time
from typing import Union, List
from pathlib import Path

import torch
import numpy as np
import supervision as sv
import openvino as ov
from PIL import Image

GROUND_DINO_DIR = Path("openvino_notebooks/notebooks/grounded-segment-anything/GroundingDINO")

# append to sys.path so that modules from the repo could be imported
sys.path.append(str(GROUND_DINO_DIR))

from groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util import get_tokenlizer
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import Model

IRS_PATH = Path("openvino_notebooks/notebooks/grounded-segment-anything/openvino_irs")
CKPT_BASE_PATH = Path("openvino_notebooks/notebooks/grounded-segment-anything/checkpoints")
os.makedirs(IRS_PATH, exist_ok=True)
os.makedirs(CKPT_BASE_PATH, exist_ok=True)

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = f"{GROUND_DINO_DIR}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = CKPT_BASE_PATH / "groundingdino_swint_ogc.pth"


ground_dino_img_size = (1024, 1280)
ov_dino_path = IRS_PATH / "openvino_grounding_dino.xml"


class GroundDINO:

    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.25

    def __init__(self):
        self.model_cfg = self._get_model_cfg()
        self.max_text_len = self.model_cfg.max_text_len
        self.dino_tokenizer = get_tokenlizer.get_tokenlizer(self.model_cfg.text_encoder_type)

        if not ov_dino_path.exists():
            self.export()

        core = ov.Core()
        self.model = core.compile_model(core.read_model(ov_dino_path), "CPU")

    def _get_model_cfg(self):
        model_cfg = SLConfig.fromfile(GROUNDING_DINO_CONFIG_PATH)
        # modified config
        model_cfg.use_checkpoint = False
        model_cfg.use_transformer_ckpt = False

        return model_cfg

    def export(self):
        pt_grounding_dino_model = self._get_pt_dino_model()
        tokenized = pt_grounding_dino_model.tokenizer(["the running dog ."], return_tensors="pt")
        input_ids = tokenized["input_ids"]
        token_type_ids = tokenized["token_type_ids"]
        attention_mask = tokenized["attention_mask"]
        position_ids = torch.arange(input_ids.shape[1]).reshape(1, -1)
        text_token_mask = torch.randint(0, 2, (1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.bool)
        img = torch.randn(1, 3, *ground_dino_img_size)

        dummpy_inputs = (
            img,
            input_ids,
            attention_mask,
            position_ids,
            token_type_ids,
            text_token_mask,
        )

        # without disabling gradients trace error occurs: "Cannot insert a Tensor that requires grad as a constant"
        for par in pt_grounding_dino_model.parameters():
            par.requires_grad = False
        # If we don't trace manually ov.convert_model will try to trace it automatically with default check_trace=True, which fails.
        # Therefore we trace manually with check_trace=False, despite there are warnings after tracing and conversion to OpenVINO IR
        # output boxes are correct.
        traced_model = torch.jit.trace(
            pt_grounding_dino_model,
            example_inputs=dummpy_inputs,
            strict=False,
            check_trace=False,
        )

        ov_dino_model = ov.convert_model(traced_model, example_input=dummpy_inputs)
        ov.save_model(ov_dino_model, ov_dino_path)

    def _get_pt_dino_model(self):
        model = build_model(self.model_cfg)
        checkpoint = torch.load(GROUNDING_DINO_CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()

        return model

    def infer(
        self,
        img_path: Path,
        caption: Union[str, List[str]],
    ):
        pil_image = Image.open(img_path)

        #  for text prompt pre-processing we reuse existing routines from GroundignDINO repo
        if isinstance(caption, list):
            caption = ". ".join(caption)
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        captions = [caption]

        tokenized = self.dino_tokenizer(captions, padding="longest", return_tensors="pt")
        specical_tokens = self.dino_tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(tokenized, specical_tokens, self.dino_tokenizer)

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :self.max_text_len, :self.max_text_len]

            position_ids = position_ids[:, :self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, :self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :self.max_text_len]

        # inputs dictionary which will be fed into the ov.CompiledModel for inference
        inputs = {}
        inputs["attention_mask.1"] = tokenized["attention_mask"]
        inputs["text_self_attention_masks"] = text_self_attention_masks
        inputs["input_ids"] = tokenized["input_ids"]
        inputs["position_ids"] = position_ids
        inputs["token_type_ids"] = tokenized["token_type_ids"]

        # GroundingDINO fails to run with input shapes different than one used for conversion.
        # As a workaround we resize input_image to the size used for conversion. Model does not rely
        # on image resolution to know object sizes therefore no need to resize box_predictions
        from torchvision.transforms.functional import resize, InterpolationMode

        input_img = resize(
            self.transform_image(pil_image),
            ground_dino_img_size,
            interpolation=InterpolationMode.BICUBIC,
        )[None, ...]
        inputs["samples"] = input_img

        # OpenVINO inference
        request = self.model.create_infer_request()
        request.start_async(inputs, share_inputs=False)
        request.wait()

        def sig(x):
            return 1 / (1 + np.exp(-x))

        logits = torch.from_numpy(sig(np.squeeze(request.get_tensor("pred_logits").data, 0)))
        boxes = torch.from_numpy(np.squeeze(request.get_tensor("pred_boxes").data, 0))

        # filter output
        filt_mask = logits.max(dim=1)[0] > self.BOX_THRESHOLD
        logits, boxes = logits[filt_mask], boxes[filt_mask]

        # get phrase and build predictions
        tokenized = self.dino_tokenizer(caption)
        pred_phrases = []
        for logit in logits:
            pred_phrase = get_phrases_from_posmap(logit > self.TEXT_THRESHOLD, tokenized, self.dino_tokenizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

        source_w, source_h = pil_image.size
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits.max(dim=1)[0]
        )

        class_id = Model.phrases2classes(phrases=pred_phrases, classes=list(map(str.lower, classes_prompt)))
        detections.class_id = class_id

        return detections

    @staticmethod
    def transform_image(pil_image: Image.Image) -> torch.Tensor:
        import groundingdino.datasets.transforms as T

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(pil_image, None)  # 3, h, w
        return image

    def infer_multiple_image(self, img_dir: Path, task_prompt: str):
        score_arr = []
        for img_file in img_dir.iterdir():
            pred = self.infer(img_file, task_prompt)
            if pred.confidence.size != 0:
                score_arr.append((img_file, pred.confidence.max().item()))
        
        score_arr = sorted(score_arr, key=lambda x: x[1], reverse=True)
        for val in score_arr:
            print(val)

    def draw_image(self, pil_image, detections):
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [f"{classes_prompt[class_id] if class_id is not None else 'None'} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
        annotated_frame = box_annotator.annotate(scene=np.array(pil_image).copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame.copy(), detections=detections, labels=labels)

        return Image.fromarray(annotated_frame)


if __name__ == "__main__":
    img_path = Path("conveyorpotato-660x330.jpg")
    classes_prompt = ["potato"]

    model = GroundDINO()
    # model.infer_multiple_image(Path("op_trailer/frames"), classes_prompt)

    start_time = time.perf_counter()
    output = model.infer(img_path, classes_prompt)
    print(f"elpased time : {time.perf_counter() - start_time}")
    print(output)
    # breakpoint()
