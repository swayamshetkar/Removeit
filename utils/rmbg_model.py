import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation
from PIL import Image
from skimage import io
import io as sysio


class RMBGRemover:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-1.4", trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def preprocess(self, im: np.ndarray, model_input_size: list) -> torch.Tensor:
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(im_tensor.unsqueeze(0), size=model_input_size, mode="bilinear")
        image = im_tensor / 255.0
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return image

    def postprocess(self, result: torch.Tensor, im_size: list) -> np.ndarray:
        result = F.interpolate(result, size=im_size, mode="bilinear").squeeze(0)
        result = (result - result.min()) / (result.max() - result.min())
        im_array = (result * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return np.squeeze(im_array)

    def remove_background(self, image_bytes: bytes) -> bytes:
        im = Image.open(sysio.BytesIO(image_bytes)).convert("RGB")
        np_im = np.array(im)
        im_size = np_im.shape[0:2]

        model_input_size = [1024, 1024]
        image = self.preprocess(np_im, model_input_size).to(self.device)

        with torch.no_grad():
            result = self.model(image)

        mask = self.postprocess(result[0][0], im_size)
        pil_mask = Image.fromarray(mask)
        im.putalpha(pil_mask)

        out_bytes = sysio.BytesIO()
        im.save(out_bytes, format="PNG")
        out_bytes.seek(0)
        return out_bytes
