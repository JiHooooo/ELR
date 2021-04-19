import albumentations as A
from albumentations.pytorch import ToTensorV2, ToTensor

def get_augmentation(input_size, train_flag=True, normalize_flag=True):
    aug_list = []
    ###基本的なサイズ変更関数
    aug_list.append(A.Resize(height=input_size[0], width=input_size[1], p=1))
    ##学習用の水増し方法
    if train_flag:
        aug_list.extend([
            A.Flip(),
            A.ShiftScaleRotate(shift_limit=(-0.02, 0.02), scale_limit=(-0.05,0.05),
                                        rotate_limit=30, border_mode=0 ,value=[0,0,0],p=0.5),
            #色変更
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5,
                                      brightness_by_max=False, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10,sat_shift_limit=10, val_shift_limit=10, p=0.5),
            #画像品質変換
            
            A.OneOf([
                        A.OneOf([
                            A.Blur(blur_limit=5, p=1),
                            A.GaussianBlur(blur_limit=5, p=1),
                                ], p=1),
                        A.GaussNoise(var_limit=(10, 80), p=1),
                        A.Downscale(scale_min=0.5, scale_max=0.5, p=1),
                    ], p=0.4),

            A.CoarseDropout(max_holes=4, max_height=int(input_size[0]/8), max_width=int(input_size[0]/8), 
                                        min_holes=1, min_height=int(input_size[0]/10), min_width=int(input_size[0]/10), 
                                        fill_value=(255,255,255), p=0.3),
            

        ])
    if normalize_flag:
        aug_list.extend([
            A.Normalize(
                p=1.0),
            ToTensorV2(p=1.0)
        ])
    else:
        aug_list.extend([
            ToTensor(),
        ])

    return A.Compose(aug_list)