import torch
from PIL import Image
from pathlib import Path

from lavis.models import load_model_and_preprocess


class Blip2ImageRetriever:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        # self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=self.device, is_eval=True)
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=self.device, is_eval=True)

    def get_single_img_score(self, img_path: Path, caption: str, verbose: bool = False):
        raw_image = Image.open(str(img_path)).convert("RGB")
        img = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](caption)

        itm_output = self.model({"image": img, "text_input": txt}, match_head="itm")
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        score = itm_scores[:, 1].item()

        if verbose:
            print(f'The {img_path.name} and {caption} are matched with a probability of {score:.3%}')
            itc_score = self.model({"image": img, "text_input": txt}, match_head='itc')
            print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

        return score

    def get_multiple_img_score(self, img_dir: Path, caption: str, top_k: int = 20):
        score_arr = []
        for img_file in img_dir.iterdir():
            score = self.get_single_img_score(img_file, caption)
            score_arr.append((img_file, score))

        score_arr = sorted(score_arr, key=lambda x: x[1], reverse=True)
        for i in range(top_k):
            print(score_arr[i])


if __name__ == "__main__":
    model = Blip2ImageRetriever()

    img_path = Path("dog.jpg")
    model.get_single_img_score(img_path, "A picture of dog", verbose=True)
