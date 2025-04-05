import gc
import math
import os.path

import cv2
import insightface
import torch
import torch.nn as nn
import numpy as np
# from basicsr.utils import img2tensor, tensor2img
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
from torchvision.utils import make_grid

from ..eva_clip import create_model_and_transforms
from ..eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .encoders_flux import IDFormer, PerceiverAttentionCA
import folder_paths

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def img2tensor(imgs, bgr2rgb=True, float32=True):
    
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)



class PuLIDPipeline(nn.Module):
    def __init__(self, dit, device,clip_vision_path, weight_dtype=torch.bfloat16,*args, **kwargs):
        super().__init__()
        self.device = device
        self.weight_dtype = weight_dtype
        self.clip_vision_path=clip_vision_path
        double_interval = 2
        single_interval = 4

        #init encoder
        self.pulid_encoder = IDFormer().to(self.device, self.weight_dtype)

        num_ca = 19 // double_interval + 38 // single_interval
        if 19 % double_interval != 0:
            num_ca += 1
        if 38 % single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA().to(self.device, self.weight_dtype) for _ in range(num_ca)
        ])

        dit.pulid_ca = self.pulid_ca
        dit.pulid_double_interval = double_interval
        dit.pulid_single_interval = single_interval

        # preprocessors
        # face align and parsing
        # self.face_helper = FaceRestoreHelper(
        #     upscale_factor=1,
        #     face_size=512,
        #     crop_ratio=(1, 1),
        #     det_model='retinaface_resnet50',
        #     save_ext='png',
        #     device=self.device,
        # )
        # self.face_helper.face_parse = None
        # self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)
        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', self.clip_vision_path, force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device, dtype=self.weight_dtype)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std
        # antelopev2
        
        # snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
            
        # providers = ['CPUExecutionProvider'] if onnx_provider == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # self.app = FaceAnalysis(name='antelopev2', root='.', providers=providers)
        # self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx',providers=providers)
        # self.handler_ante.prepare(ctx_id=0)

        gc.collect()
        torch.cuda.empty_cache()

        # self.load_pretrain()

        # other configs
        self.debug_img_list = []
    
    def components_to_device(self, device):
        # everything but pulid_ca
        # self.face_helper.face_det = self.face_helper.face_det.to(device)
        # self.face_helper.face_parse = self.face_helper.face_parse.to(device)
        self.clip_vision_model = self.clip_vision_model.to(device)
    #     self.pulid_encoder = self.pulid_encoder.to(device)
    def load_pretrain(self, pretrain_path=None):
        
        if pretrain_path is not None:
            ckpt_path = pretrain_path
        else:
            hf_hub_download('guozinan/PuLID', 'pulid_flux_v0.9.0.safetensors', local_dir='models')
            ckpt_path = 'models/pulid_flux_v0.9.0.safetensors'
        state_dict = load_file(ckpt_path)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

        del state_dict
        del state_dict_dict

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    @torch.no_grad()
    def get_id_embedding(self, image, cal_uncond=False):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """
        self.face_helper.clean_all()
        self.debug_img_list = []
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                image[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.weight_dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image.to(self.weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

        id_embedding = self.pulid_encoder(id_cond, id_vit_hidden)

        if not cal_uncond:
            return id_embedding, None

        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))
        uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)

        return id_embedding, uncond_id_embedding


    @torch.no_grad()
    def get_id_embedding_(self, image,face_helper,app,handler_ante, cal_uncond=False):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """
        self.face_helper=face_helper
        self.app=app
        self.handler_ante=handler_ante

        self.face_helper.clean_all()
        self.debug_img_list = []
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                image[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.weight_dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image.to(self.weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

        id_embedding = self.pulid_encoder(id_cond, id_vit_hidden)

        if not cal_uncond:
            return id_embedding, None

        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))
        uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)

        return id_embedding, uncond_id_embedding