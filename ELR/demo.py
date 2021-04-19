import argparse
import os
import shutil
import torch
import tqdm
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.SelfDataset import SelfDataset_val
from data_loader.augmentation import get_augmentation
from torch.utils.data import DataLoader

def main(config, args):
    logger = config.get_logger('inference')

    label_list = config['data_loader']['args']['label_name']
    for one_label in label_list:
        os.makedirs('%s/%s'%(args.demo_save, one_label))
    
    #load dataset
    val_transforms = get_augmentation((config['data_loader']['args']['image_size'], config['data_loader']['args']['image_size']),
                                        train_flag=False, 
                                        normalize_flag=config['data_loader']['args']['normalize_flag'])
    dataset_infer = SelfDataset_val(args.demo_folder, transforms=val_transforms)
    dataloader_infer = DataLoader(dataset=dataset_infer, batch_size=config['data_loader']['args']['batch_size'],
                                num_workers=config['data_loader']['args']['num_workers'], shuffle=False)
    
    # build model architecture
    model = config.initialize('arch', module_arch)
    
    # load pretrained model
    logger.info('Loading checkpoint: {} ...'.format(args.resume))
    checkpoint = torch.load(args.resume,map_location='cpu')
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    image_index = 0
    with torch.no_grad():
        for images in tqdm.tqdm(dataloader_infer):
            outputs = torch.nn.Softmax(dim=1)(model(images.to(device)))
            for output in outputs:
                predict_y = int(output.data.argmax(axis=0))
                image_path = dataset_infer.image_pathes[image_index]
                base_name = os.path.split(image_path)[1]
                shutil.copy(image_path, '%s/%s/%s'%(args.demo_save, label_list[predict_y], base_name))
                image_index += 1

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default=None, type=str,
                    help='config file path (default: None)')
    args.add_argument('--demo-folder', default='DemoImage',
                    help='folder where the demo images are saved')
    args.add_argument('--demo-save', default='DemoResults',
                    help='folder where to save demo results')
    args.add_argument('--resume', help='model for inference')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.get_instance(args, '')
    #config = ConfigParser(args)
    main(config, args.parse_args())