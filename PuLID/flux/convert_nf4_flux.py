"""
Utilities adapted from

* https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/quantizer_bnb_4bit.py
* https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/bitsandbytes.py
"""

import torch
import bitsandbytes as bnb
from transformers.quantizers.quantizers_utils import get_module_from_name
import torch.nn as nn
from accelerate import init_empty_weights


def _replace_with_bnb_linear(
    model,
    method="nf4",
    has_been_replaced=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            with init_empty_weights():
                in_features = module.in_features
                out_features = module.out_features

                if method == "llm_int8":
                    model._modules[name] = bnb.nn.Linear8bitLt(
                        in_features,
                        out_features,
                        module.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0,
                    )
                    has_been_replaced = True
                else:
                    model._modules[name] = bnb.nn.Linear4bit(
                        in_features,
                        out_features,
                        module.bias is not None,
                        compute_dtype=torch.bfloat16,
                        compress_statistics=False,
                        quant_type="nf4",
                    )
                    has_been_replaced = True
                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
    return model, has_been_replaced


def check_quantized_param(
    model,
    param_name: str,
) -> bool:
    module, tensor_name = get_module_from_name(model, param_name)
    if isinstance(module._parameters.get(tensor_name, None), bnb.nn.Params4bit):
        # Add here check for loaded components' dtypes once serialization is implemented
        return True
    elif isinstance(module, bnb.nn.Linear4bit) and tensor_name == "bias":
        # bias could be loaded by regular set_module_tensor_to_device() from accelerate,
        # but it would wrongly use uninitialized weight there.
        return True
    else:
        return False


def create_quantized_param(
    model,
    param_value: "torch.Tensor",
    param_name: str,
    target_device: "torch.device",
    state_dict=None,
    unexpected_keys=None,
    pre_quantized=False
):
    module, tensor_name = get_module_from_name(model, param_name)

    if tensor_name not in module._parameters:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

    old_value = getattr(module, tensor_name)

    if tensor_name == "bias":
        if param_value is None:
            new_value = old_value.to(target_device)
        else:
            new_value = param_value.to(target_device)

        new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad)
        module._parameters[tensor_name] = new_value
        return

    if not isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
        raise ValueError("this function only loads `Linear4bit components`")
    if (
        old_value.device == torch.device("meta")
        and target_device not in ["meta", torch.device("meta")]
        and param_value is None
    ):
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

    if pre_quantized:
        if (param_name + ".quant_state.bitsandbytes__fp4" not in state_dict) and (
                param_name + ".quant_state.bitsandbytes__nf4" not in state_dict
            ):
                raise ValueError(
                    f"Supplied state dict for {param_name} does not contain `bitsandbytes__*` and possibly other `quantized_stats` components."
                )

        quantized_stats = {}
        for k, v in state_dict.items():
            # `startswith` to counter for edge cases where `param_name`
            # substring can be present in multiple places in the `state_dict`
            if param_name + "." in k and k.startswith(param_name):
                quantized_stats[k] = v
                if unexpected_keys is not None and k in unexpected_keys:
                    unexpected_keys.remove(k)

        new_value = bnb.nn.Params4bit.from_prequantized(
            data=param_value,
            quantized_stats=quantized_stats,
            requires_grad=False,
            device=target_device,
        )

    else:
        new_value = param_value.to("cpu")
        kwargs = old_value.__dict__
        new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)

    module._parameters[tensor_name] = new_value