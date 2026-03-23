import torch
from torchvision.io.image import read_image
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
from driving_model.model_car import Resnet101Steer, Resnet101Speed, Vgg16Steer, Vgg16Speed
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask


if __name__ == "__main__":
    # model = Resnet101Speed()
    # model.load_state_dict(torch.load("/home/weizi/workspace/misbehavior_prediction/src/driving_model/model/resnet101/speed.pt"))
    img_transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor()
    ])
    model = resnet18(pretrained=True)
    model.eval()
    img = Image.open("/home/weizi/workspace/misbehavior_prediction/src/30103824.png").convert("RGB")
    input_tensor = img_transform(img)
    # input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    with SmoothGradCAMpp(model) as cam_extractor:
        out = model(input_tensor.unsqueeze(0))
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        print(activation_map)
        result = overlay_mask(img, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.)
        plt.imshow(result)
        plt.axis('off')
        plt.tight_layout()
        plt.show()