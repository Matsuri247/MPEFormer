import torch
from .MPEFormer import MPEFormer

def model_generator(method, msfa_size, pretrained_model_path=None):
    # TODO
    if method == 'MPEFormer':
        model = MPEFormer(stage=2, msfa_size=msfa_size).cuda()

    else:
        print(f'Method {method} is not defined !!!!')

    # 读取.pth文件
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
    return model
