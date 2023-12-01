from GA import GA
from utils import *
from models.cifar10.resnet_model import *
import config as flags
import numpy as np

val_loader = get_imagenet_val_loader(flags.imagenet12_valk, 1, image_size=299)

target_model = init_model('inceptionv3')

m = resnet50(pretrained=True)
m.eval()
counter = 0


for i, (data, label) in enumerate(val_loader):
    if (i == 200):
        break
    image = data.squeeze().numpy()
    tt, eval_num, save = GA(20, 500, 299*299*3, target_model, m, image, label.numpy(), 0.05, len(data), None, True)
    x_adv = np.clip(image + np.reshape(tt, image.shape) * 0.05, 0, 1)
    pred = np.squeeze(target_model(torch.tensor(x_adv)[None, ...].float().to(flags.device)).detach().cpu().numpy())
    predict = np.argmax(pred, axis=-1)
    print(f'{i}/{len(val_loader)}')
    if (predict != label.item()):
        print('Success')
        counter += 1
print('Finished')
print(counter/len(val_loader))
