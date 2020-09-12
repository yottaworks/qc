# https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

import numpy as np
import torch
import torch.onnx
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


# Using our model to predict the label
def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)

    # Reverse the log function in our output
    output = torch.exp(output)

    # Get the top predicted class, and the output percentage for
    # that class
    probabilities, classes = output.topk(1, dim=1)

    return probabilities.item(), classes.item()


def main():
    # Make sure the transform matches what is used to train the model
    test_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('qc_model.pth')
    model.eval()
    to_pil = transforms.ToPILImage()

    test_images = ['data/train/Not Shipping Container/4222682943_674cbe55fa.jpg',
                   'captured_images/BMOU6297182-3B4F26D0-B8DB-456C-A823-861DAE02EC3A.jpg',
                   'captured_images/00LU7778568-35138729-7021-4D0E-9D22-6E6699BBFE52.jpg',
                   'captured_images/BMOU5172977-97B25EE6-C8A3-4D5E-A7A3-A8ABF492AB9E.jpg',
                   'captured_images/BNOU4899259-CAE9E383-1186-4BA2-A594-A475815FE8C9.jpg',
                   'captured_images/BSAJ2918899-F4EA99DD-D77F-4484-96B4-FEF0056CB1A6.jpg',
                   'captured_images/BSIU2458453-E53143EE-464D-4AC4-B048-9E14B5783198.jpg',
                   'captured_images/BSIU2918899-4713734C-C6DD-45B8-8373-C317FC39E4C2.jpg',
                   'captured_images/CAIU4018381-5E64DD6E-68B2-4563-95DC-86DF81DC12E9.jpg',
                   'captured_images/62242-470B3488-5004-47D8-94D1-0352B5E669DE.jpg',
                   'data/train/Shipping Container Front/8ft-shipping-container.jpg',
                   'data/train/Shipping Container Front/10ft-container-01.jpg',
                   'data/train/Shipping Container Front/10ft-green-shipping-container-for-sale-large.jpg',
                   'data/train/Shipping Container Front/20-standard.jpg']
    labels = ['Not a shipping container', 'shipping container', 'shipping container front']

    for ii in test_images:
        print('Processing: %s' % ii)
        img = np.array(Image.open(ii))
        image = to_pil(img)
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        predict_input = Variable(image_tensor)
        predict_input = predict_input.to(device)
        output = model(predict_input)

        output = torch.exp(output)
        probabilities, classes = output.topk(1, dim=1)

        message = 'Model is {:01.1f}% certain image is {}'.format(probabilities.item() * 100, labels[classes.item()])
        print(message)


if __name__ == '__main__':
    main()

