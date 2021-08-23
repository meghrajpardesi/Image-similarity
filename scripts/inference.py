import torch
from PIL import Image
from utils import image_loader
import sys

import importlib
sys.path.append("/home/saktiman/Dev-ai/image_similarity/models")
import warnings
warnings.filterwarnings("ignore")
from reduction_resnet import ReductionResNet

class ImageSimilarity:
    
    
    def __init__(self):
        self.model = ReductionResNet()
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.result = {}
        
        
    
    def  _post_processing(self, cosinesimilarity :torch.Tensor) -> dict:
        """ _post_processing checks return True if images have more than 0.95 similarity.
        
        Args:
            cosinesimilarity (torch.Tensor): cosine simlarity value calculated from two image embeddings

        Returns:
            dict: response in True/False
        """
        self.result['simlarity_score'] = round(float(cosinesimilarity),4)
        if cosinesimilarity > 0.95:
            self.result['description'] = "image are similar"
        else:
            self.result['description'] = "image are not similar"
            
        return self.result
    
    
    def check_similarity(self, img1: str, img2:str)->dict:
        """[summary]

        Args:
            img1 (str): path of image to be check for similarity
            img2 (str): path of image to be check for similarity

        Returns:
            dict: [description]
        """
        img1 = image_loader(img1)
        img2 = image_loader(img2)
        ebd1 = self.model.forward(img1)
        ebd2 = self.model.forward(img2)
        cossim = self.cos(ebd1, ebd2)
        self.result = self._post_processing(cossim)
        
        return self.result
    

if __name__=='__main__':
    path1 = "/home/saktiman/Dev-ai/image_similarity/data/1.jpeg"
    path2 ="/home/saktiman/Dev-ai/image_similarity/data/2.jpeg"
    path3 ="/home/saktiman/Dev-ai/image_similarity/data/3.jpg"
    path4 ="/home/saktiman/Dev-ai/image_similarity/data/4.jpg"
    path5 ="/home/saktiman/Dev-ai/image_similarity/data/5.jpeg"
    path6 ="/home/saktiman/Dev-ai/image_similarity/data/6.jpeg"
    ims = ImageSimilarity()
    res = ims.check_similarity(path3, path6)
    print(res)