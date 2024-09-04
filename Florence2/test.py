import time
from pathlib import Path

import torch
import supervision as sv
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
from optimum.exporters.openvino import export_from_model


bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
mask_annotator = sv.MaskAnnotator()


class Florence2:
    def __init__(self, model_id: str = 'microsoft/Florence-2-large'):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        export_from_model(self.model, output="ov_model", task="object-detection")

    def infer_image(self, img_path: Path, task_prompt: str, text_input=None) -> dict:
        """
        Return format:
            {'<OD>': {'bboxes': [[115.75, 3.19599986076355, 498.75, 373.5559997558594]], 'labels': ['dog']}}
        """
        image = Image.open(img_path)

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )

        return parsed_answer

    def infer_multiple_image(self, img_dir: Path, task_prompt: str, text_input=None):
        score_arr = []
        for img_file in img_dir.iterdir():
            pred = self.infer_image(img_file, task_prompt, text_input)
            score_arr.append((img_file, pred[task_prompt]["labels"]))

    def draw_image(self, input_image: Path, bboxes: np.ndarray, label: np.ndarray):
        detections = sv.Detections(
            xyxy=bboxes,
            class_id=label,
            # confidence=pred_instances['scores']
        )

        # label ids with confidence scores
        labels = [
            f"{class_id} {confidence:0.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        # draw bounding box with label
        image = Image.open(input_image)
        svimage = np.array(image)
        svimage = bounding_box_annotator.annotate(svimage, detections)
        svimage = label_annotator.annotate(svimage, detections, labels)

        return svimage


if __name__ == "__main__":
    img_path = Path("dog.jpg")
    model = Florence2()

    text_input = None
    task_prompt = '<OD>'
    # task_prompt = '<DENSE_REGION_CAPTION>'
    # task_prompt = '<OPEN_VOCABULARY_DETECTION>'

    start_time = time.perf_counter()
    results = model.infer_image(img_path, task_prompt, text_input)
    elapsed_time = time.perf_counter() - start_time

    print(results)
    print(f"elapsed time : {elapsed_time}")
