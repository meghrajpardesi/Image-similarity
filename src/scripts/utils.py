"""Utility functions for image similarity module."""
from io import BytesIO
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve
import base64
import hashlib

from PIL import Image
from tqdm import tqdm
import numpy as np
import smart_open
from PIL import Image
import torch
from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.Resize(440),
    transforms.CenterCrop(416),
    transforms.ToTensor()
])

def image_loader(image_name :str) -> torch.Tensor:
    """image loader reads image from given path and return Tensor image."""
    image = Image.open(image_name)
    image = data_transforms(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def read_image_pil(image_uri: Union[Path, str], grayscale=False) -> Image:
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_image_pil_file(image_file, grayscale=False) -> Image:
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
            image = data_transforms(image).float()
            image = torch.tensor(image, requires_grad=True)
            image = image.unsqueeze(0)
        return image


# Hide lines below until Lab 9
def read_b64_image(b64_string, grayscale=False):  # pylint: disable=unused-argument
    """Load base64-encoded images."""
    try:
        _, b64_data = b64_string.split(",")  # pylint: disable=unused-variable
        image_file = BytesIO(base64.b64decode(b64_data))
        return read_image_pil_file(image_file, grayscale)
    except Exception as exception:
        raise ValueError("Could not load image from b64 {}: {}".format(b64_string, exception)) from exception


# Hide lines above until Lab 9


def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec
        
 
def post_processing(cosinesimilarity :torch.Tensor) -> dict:
    """ post_processing checks return True if images have more than 0.95 similarity.
    
    
    Args:
        cosinesimilarity (torch.Tensor): cosine simlarity value calculated from two image embeddings

    Returns:
        dict: response in True/False
    """
    result = {}
    result['simlarity_score'] = round(float(cosinesimilarity),4)
    if cosinesimilarity > 0.95:
        result['description'] = "image are similar"
    else:
        result['description'] = "image are not similar"
        
    return result

