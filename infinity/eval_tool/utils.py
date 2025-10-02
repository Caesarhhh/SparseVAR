import torch
from peft.tuners import lora

#from vars import LORA_PATHS, SVDQ_LORA_PATHS


def hash_str_to_int(s: str) -> int:
    """Hash a string to an integer."""
    modulus = 10**9 + 7  # Large prime modulus
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def get_pipeline(
    model_name: str,
    precision: str,
    use_qencoder: bool = False,
    lora_name: str = "None",
    lora_weight: float = 1,
    device: str = "cuda",
):
    if model_name == "schnell":
        if precision == "int4":
            assert torch.device(device).type == "cuda", "int4 only supported on CUDA devices"
            pipeline = nunchaku_flux.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.bfloat16,
                qmodel_path="mit-han-lab/svdquant-models/svdq-int4-flux.1-schnell.safetensors",
                qencoder_path="mit-han-lab/svdquant-models/svdq-w4a16-t5.pt" if use_qencoder else None,
                qmodel_device=device,
            )
        else:
            assert precision == "bf16"
            pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    elif model_name == "dev":
        if precision == "int4":
            pipeline = nunchaku_flux.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                qmodel_path="mit-han-lab/svdquant-models/svdq-int4-flux.1-dev.safetensors",
                qencoder_path="mit-han-lab/svdquant-models/svdq-w4a16-t5.pt" if use_qencoder else None,
                qmodel_device=device,
            )
            if lora_name not in ["All", "None"]:
                pipeline.transformer.nunchaku_update_params(SVDQ_LORA_PATHS[lora_name])
                pipeline.transformer.nunchaku_set_lora_scale(lora_weight)
        else:
            assert precision == "bf16"
            pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
            if lora_name == "All":
                # Pre-load all the LoRA weights for demo use
                for name, path in LORA_PATHS.items():
                    pipeline.load_lora_weights(path["name_or_path"], weight_name=path["weight_name"], adapter_name=name)
                for m in pipeline.transformer.modules():
                    if isinstance(m, lora.LoraLayer):
                        m.set_adapter(m.scaling.keys())
                        for name in m.scaling.keys():
                            m.scaling[name] = 0
            elif lora_name != "None":
                path = LORA_PATHS[lora_name]
                pipeline.load_lora_weights(
                    path["name_or_path"], weight_name=path["weight_name"], adapter_name=lora_name
                )
                for m in pipeline.transformer.modules():
                    if isinstance(m, lora.LoraLayer):
                        for name in m.scaling.keys():
                            m.scaling[name] = lora_weight
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    pipeline = pipeline.to(device)

    return pipeline
