from safetensors.torch import load_file
import torch
from tqdm import tqdm

__all__ = [
    'flux_load_lora'
]


def is_int(d):
    try:
        d = int(d)
        return True
    except Exception as e:
        return False


def flux_load_lora(self, lora_file, lora_weight=1.0):
    device = self.transformer.device

    # DiT 部分
    state_dict, network_alphas = self.lora_state_dict(lora_file, return_alphas=True)
    state_dict = {k:v.to(device) for k,v in state_dict.items()}
    
    model = self.transformer
    keys = list(state_dict.keys())
    keys = [k for k in keys if k.startswith('transformer.')]

    for k_lora in tqdm(keys, total=len(keys), desc=f"loading lora in transformer ..."):
        v_lora = state_dict[k_lora]

        # 非 up 的都跳过
        if '.lora_A.weight' in k_lora:
            continue
        if '.alpha' in k_lora:
            continue

        k_lora_name = k_lora.replace("transformer.", "")
        k_lora_name = k_lora_name.replace(".lora_B.weight", "")
        attr_name_list = k_lora_name.split('.')

        cur_attr = model
        latest_attr_name = ''
        for idx in range(0, len(attr_name_list)):
            attr_name = attr_name_list[idx]
            if is_int(attr_name):
                cur_attr = cur_attr[int(attr_name)]
                latest_attr_name = ''
            else:
                try:
                    if latest_attr_name != '':
                        cur_attr = cur_attr.__getattr__(f"{latest_attr_name}.{attr_name}")
                    else:
                        cur_attr = cur_attr.__getattr__(attr_name)
                    latest_attr_name = ''
                except Exception as e:
                    if latest_attr_name != '':
                        latest_attr_name = f"{latest_attr_name}.{attr_name}"
                    else:
                        latest_attr_name = attr_name

        up_w = v_lora
        down_w = state_dict[k_lora.replace('.lora_B.weight', '.lora_A.weight')]

        # 赋值
        einsum_a = f"ijabcdefg"
        einsum_b = f"jkabcdefg"
        einsum_res = f"ikabcdefg"
        length_shape = len(up_w.shape)
        einsum_str = f"{einsum_a[:length_shape]},{einsum_b[:length_shape]}->{einsum_res[:length_shape]}"
        dtype = cur_attr.weight.data.dtype
        d_w = torch.einsum(einsum_str, up_w.to(torch.float32), down_w.to(torch.float32)).to(dtype)
        cur_attr.weight.data = cur_attr.weight.data + d_w * lora_weight



    # text encoder 部分
    raw_state_dict = load_file(lora_file)
    raw_state_dict = {k:v.to(device) for k,v in raw_state_dict.items()}

    # text encoder
    state_dict = {k:v for k,v in raw_state_dict.items() if 'lora_te1_' in k}
    model = self.text_encoder
    keys = list(state_dict.keys())
    keys = [k for k in keys if k.startswith('lora_te1_')]

    for k_lora in tqdm(keys, total=len(keys), desc=f"loading lora in text_encoder ..."):
        v_lora = state_dict[k_lora]

        # 非 up 的都跳过
        if '.lora_down.weight' in k_lora:
            continue
        if '.alpha' in k_lora:
            continue

        k_lora_name = k_lora.replace("lora_te1_", "")
        k_lora_name = k_lora_name.replace(".lora_up.weight", "")
        attr_name_list = k_lora_name.split('_')

        cur_attr = model
        latest_attr_name = ''
        for idx in range(0, len(attr_name_list)):
            attr_name = attr_name_list[idx]
            if is_int(attr_name):
                cur_attr = cur_attr[int(attr_name)]
                latest_attr_name = ''
            else:
                try:
                    if latest_attr_name != '':
                        cur_attr = cur_attr.__getattr__(f"{latest_attr_name}_{attr_name}")
                    else:
                        cur_attr = cur_attr.__getattr__(attr_name)
                    latest_attr_name = ''
                except Exception as e:
                    if latest_attr_name != '':
                        latest_attr_name = f"{latest_attr_name}_{attr_name}"
                    else:
                        latest_attr_name = attr_name

        up_w = v_lora
        down_w = state_dict[k_lora.replace('.lora_up.weight', '.lora_down.weight')]
        
        alpha = state_dict.get(k_lora.replace('.lora_up.weight', '.alpha'), None)
        if alpha is None:
            lora_scale = 1
        else:
            rank = up_w.shape[1]
            lora_scale = alpha / rank
        
        # 赋值
        einsum_a = f"ijabcdefg"
        einsum_b = f"jkabcdefg"
        einsum_res = f"ikabcdefg"
        length_shape = len(up_w.shape)
        einsum_str = f"{einsum_a[:length_shape]},{einsum_b[:length_shape]}->{einsum_res[:length_shape]}"
        dtype = cur_attr.weight.data.dtype
        d_w = torch.einsum(einsum_str, up_w.to(torch.float32), down_w.to(torch.float32)).to(dtype)
        cur_attr.weight.data = cur_attr.weight.data + d_w * lora_scale * lora_weight

