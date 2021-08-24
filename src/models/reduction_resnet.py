"""Author :Meghraj Pardesi
Licence: MIT
The torch model file implements a cosine similarity algorithm

"""
import torch
import torchvision
import torchextractor as tx
from torch.onnx import register_custom_op_symbolic


def register_custom_op() -> None:
    torch.ops.load_library(
        "/home/saktiman/Dev-ai/image_similarity/scripts/build/lib.linux-x86_64-3.9/reduction_op.cpython-39-x86_64-linux-gnu.so"
    )
    """register_custom_op method registers the custom reduction operation on torch.onnx"""

    def my_reuduction(
        g: object,
        layer_one: torch.Tensor,
        layer_two: torch.Tensor,
        layer_three: torch.Tensor,
        layer_four: torch.Tensor,
    ) -> object:
        return g.op(
            "mydomain::reduction", layer_one, layer_two, layer_three, layer_four
        )

    register_custom_op_symbolic("mynamespace::reduction", my_reuduction, 9)


class ReductionResNet(torch.nn.Module):
    """Custom reduction res net for get calculating embeddings for the


    Args:
        torch ([type]): [description]
    """

    def __init__(self) -> None:
        """method setup all the necessary variable and object to create a model."""

        super(ReductionResNet, self).__init__()
        register_custom_op()
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.model = tx.Extractor(
            self.base_model, ["layer1", "layer2", "layer3", "layer4"]
        )
        self.model.eval()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """forward methods takes a input tensor as [B, C, H, W] pass it
        throught the model and returns the generated embedding

        Args:
            input (torch.Tensor):

        Returns:
            embedding: 512 size embedding consisting of image features.
        """
        _, features = self.model(input)
        embedding = torch.ops.mynamespace.reduction(
            features.get("layer1"),
            features.get("layer2"),
            features.get("layer3"),
            features.get("layer4"),
        )
        return embedding


if __name__ == "__main__":
    input = torch.randn(1, 3, 416, 416)
    net = ReductionResNet()
    res = net.forward(input)
    print(res)
