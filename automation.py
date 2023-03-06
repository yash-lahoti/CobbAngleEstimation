import argparse
import json
import os
import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader
from torchvision import models
#from Code.NN_Module import NNModel
import cv2



from model_utils import SegmentationDataset, DiceLoss

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required args
    parser.add_argument('-p', '--params',
                        required=True,
                        help='Json file containing NN training params')
    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output directory')

    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    cwd = os.getcwd()
    os.makedirs(args.output, exist_ok=True)

    with open(args.params, 'r') as handle:
        model_params = json.load(handle)

    print('Building Dataset')
    train_transform = A.Compose(
        [
            A.Resize(model_params['img_width'] * model_params['height_ratio'], model_params['img_width']),
            A.transforms.CLAHE(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate((-30, 30), border_mode=cv2.BORDER_CONSTANT),
            ToTensorV2()
        ]
    )

    image_datasets = {
        x: SegmentationDataset(root=model_params['data_root'],
                               image_folder=model_params['image_folder'],
                               mask_folder=model_params['mask_folder'],
                               seed=model_params['seed'],
                               fraction=model_params['test_split'],
                               subset=x,
                               transforms=train_transform)
        for x in ['Train', 'Test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=4,
                      shuffle=True)
        for x in ['Train', 'Test']
    }

    print('Configuring Model')

    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    SegModel = NNModel(model, model_params)

    print('Training Model')
    l = SegModel.evaluate(predict_generator=dataloaders['Test'])
    SegModel.fit(dataloaders['Train'], dataloaders['Test'], args.output)

    print('Evaluate Performance')
    SegModel.evaluate(predict_generator=dataloaders['Test'], analysis_mode=True)
    l2 = SegModel.evaluate(predict_generator=dataloaders['Test'])
    print("done!")


if __name__ == '__main__':
    main(sys.argv[1:])






