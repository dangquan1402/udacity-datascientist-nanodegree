import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as v_transforms
from PIL import Image

validation_transform = v_transforms.Compose(
    [
        v_transforms.Resize((64, 64)),
        v_transforms.ToTensor(),
    ]
)


class SamplePretrainedModel(nn.Module):
    def __init__(self, num_classes):
        super(SamplePretrainedModel, self).__init__()
        self.num_classes = num_classes
        self.base_model = torchvision.models.alexnet(pretrained=False).features
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


GENDER_MODEL = SamplePretrainedModel(num_classes=2)
GENDER_MODEL.load_state_dict(
    torch.load("weights/best_weight.pt", map_location=torch.device("cpu"))
)


def get_input(path):
    image = Image.open(path)
    image_tensor = validation_transform(image)
    return torch.unsqueeze(image_tensor, 0)


def get_prediction(path):
    image_tensor = get_input(path)
    with torch.no_grad():
        output = GENDER_MODEL(image_tensor)
        output = F.softmax(output[0], dim=0).numpy()
    output = list(output)
    output = [float(f) for f in output]

    return dict(zip(["female", "male"], output))


def main():
    path = "sample_image.jpg"
    result = get_prediction(path)
    print(result)


if __name__ == "__main__":
    main()
