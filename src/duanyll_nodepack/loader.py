import os.path
import json
import torch
import huggingface_hub

import comfy.utils
import comfy.sd
import folder_paths


def get_checkpoint_file_from_hf(
    repo_id: str, subfolder: str = None, filename: str = None
) -> str:
    if subfolder is None:
        subfolder = ""
    if filename is not None:
        return os.path.join(subfolder, filename)

    # List files in the subfolder of the Hugging Face repository
    files = [
        f
        for f in huggingface_hub.list_repo_tree(repo_id, path_in_repo=subfolder)
        if isinstance(f, huggingface_hub.hf_api.RepoFile)
    ]

    suffixes = [".index.json", ".safetensors", ".sft", ".bin", ".pt"]

    for suffix in suffixes:
        matching_files = [f.rfilename for f in files if f.rfilename.endswith(suffix)]
        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple files found with suffix '{suffix}' in subfolder '{subfolder}' of repository '{repo_id}'. Please specify a filename."
            )
        elif len(matching_files) == 1:
            return matching_files[0]

    raise ValueError(
        f"No files found with the expected suffixes in subfolder '{subfolder}' of repository. Please specify a filename."
    )


def load_checkpoint_from_hf(
    repo_id: str, subfolder: str = None, filename: str = None, return_metadata=False
):
    checkpoint_file = get_checkpoint_file_from_hf(repo_id, subfolder, filename)

    if checkpoint_file.endswith(".index.json"):
        # Load the index file to get the actual checkpoint file
        index_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=checkpoint_file
        )
        with open(index_path, "r") as f:
            index_data = json.load(f)
        # Actual checkpoint files are unique values in 'weight_map'
        checkpoint_files = set(index_data["weight_map"].values())
        state_dicts = []
        for file in checkpoint_files:
            full_filename = os.path.join(os.path.dirname(checkpoint_file), file)
            file_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id, filename=full_filename
            )
            state_dicts.append(comfy.utils.load_torch_file(file_path))
        # Merge all state dicts
        merged_state_dict = {}
        for state_dict in state_dicts:
            for key, value in state_dict.items():
                if key in merged_state_dict:
                    merged_state_dict[key] += value
                else:
                    merged_state_dict[key] = value
        if return_metadata:
            return merged_state_dict, None
        else:
            return merged_state_dict
    else:
        # Directly load the checkpoint file
        file_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=checkpoint_file
        )
        state_dict, metadata = comfy.utils.load_torch_file(
            file_path, return_metadata=True
        )
        if return_metadata:
            return state_dict, metadata
        else:
            return state_dict


REPO_TOOLTIP = "The Hugging Face repository ID of the model."
SUBFOLDER_TOOLTIP = "The subfolder in the repository where the model is located. Will look for the checkpoint file in this subfolder."
FILENAME_TOOLTIP = "The name of the checkpoint file to load. If not specified, will try to find a suitable file in the subfolder. To specify sharded checkpoint files, enter the filename of the index file (e.g., 'model.index.json')."
DESCRIPTION = "Downloads a model checkpoint from Hugging Face with `huggingface_hub` and loads it in native ComfyUI format. Will utilize global cache (`~/.cache/huggingface/hub` by default) to avoid re-downloading the same file multiple times."


class HfCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "", "tooltip": REPO_TOOLTIP}),
            },
            "optional": {
                "subfolder": ("STRING", {"default": "", "tooltip": SUBFOLDER_TOOLTIP}),
                "filename": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "duanyll/huggingface"
    DESCRIPTION = DESCRIPTION

    def load_checkpoint(self, repo_id, subfolder, filename):
        if not repo_id:
            raise ValueError("Repository ID cannot be empty.")

        if not subfolder:
            subfolder = None

        if not filename:
            filename = None

        state_dict, metadata = load_checkpoint_from_hf(
            repo_id, subfolder, filename, return_metadata=True
        )

        out = comfy.sd.load_state_dict_guess_config(
            state_dict,
            output_model=True,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            metadata=metadata,
        )

        return out[:3]


class HfDiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                ),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                ),
            },
            "optional": {
                "subfolder": (
                    "STRING",
                    {"default": "transformer", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "filename": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "duanyll/huggingface"
    DESCRIPTION = DESCRIPTION

    def load_model(self, repo_id, weight_dtype, subfolder, filename):
        if not repo_id:
            raise ValueError("Repository ID cannot be empty.")

        if not subfolder:
            subfolder = None

        if not filename:
            filename = None

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        state_dict, metadata = load_checkpoint_from_hf(
            repo_id, subfolder, filename, return_metadata=True
        )

        model = comfy.sd.load_diffusion_model_state_dict(
            state_dict, model_options=model_options
        )
        if model is None:
            raise ValueError(
                "Failed to load the diffusion model from the provided checkpoint."
            )
        return (model,)


class HfVaeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                ),
            },
            "optional": {
                "subfolder": ("STRING", {"default": "", "tooltip": SUBFOLDER_TOOLTIP}),
                "filename": (
                    "STRING",
                    {"default": "ae.safetensors", "tooltip": FILENAME_TOOLTIP},
                ),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "duanyll/huggingface"
    DESCRIPTION = DESCRIPTION

    def load_vae(self, repo_id, subfolder, filename):
        if not repo_id:
            raise ValueError("Repository ID cannot be empty.")

        if not subfolder:
            subfolder = None

        if not filename:
            filename = None

        state_dict, metadata = load_checkpoint_from_hf(
            repo_id, subfolder, filename, return_metadata=True
        )

        vae = comfy.sd.VAE(sd=state_dict)
        vae.throw_exception_if_invalid()

        return (vae,)


class HfLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model the LoRA will be applied to."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "The CLIP model the LoRA will be applied to."},
                ),
                "repo_id": ("STRING", {"default": "", "tooltip": REPO_TOOLTIP}),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
                "strength_clip": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the CLIP model. This value can be negative.",
                    },
                ),
            },
            "optional": {
                "subfolder": ("STRING", {"default": "", "tooltip": SUBFOLDER_TOOLTIP}),
                "filename": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "duanyll/huggingface"
    DESCRIPTION = DESCRIPTION

    def load_lora(
        self, model, clip, repo_id, strength_model, strength_clip, subfolder, filename
    ):
        if not repo_id:
            raise ValueError("Repository ID cannot be empty.")

        if not subfolder:
            subfolder = None

        if not filename:
            filename = None

        state_dict, metadata = load_checkpoint_from_hf(
            repo_id, subfolder, filename, return_metadata=True
        )

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, state_dict, strength_model, strength_clip
        )
        return (model_lora, clip_lora)


class HfLoraLoaderModelOnly:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model the LoRA will be applied to."},
                ),
                "repo_id": ("STRING", {"default": "", "tooltip": REPO_TOOLTIP}),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            },
            "optional": {
                "subfolder": ("STRING", {"default": "", "tooltip": SUBFOLDER_TOOLTIP}),
                "filename": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "duanyll/huggingface"
    DESCRIPTION = DESCRIPTION

    def load_lora(self, model, repo_id, strength_model, subfolder, filename):
        if not repo_id:
            raise ValueError("Repository ID cannot be empty.")

        if not subfolder:
            subfolder = None

        if not filename:
            filename = None

        state_dict, metadata = load_checkpoint_from_hf(
            repo_id, subfolder, filename, return_metadata=True
        )

        model_lora = comfy.sd.load_lora_for_models(
            model, None, state_dict, strength_model, 0.0
        )[0]
        return (model_lora,)


class HfDualClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id_1": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                ),
                "repo_id_2": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                ),
                "type": (
                    ["sdxl", "sd3", "flux", "hunyuan_video", "hidream"],
                    {"default": "flux"},
                ),
            },
            "optional": {
                "subfolder_1": (
                    "STRING",
                    {"default": "text_encoder", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "subfolder_2": (
                    "STRING",
                    {"default": "text_encoder_2", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "filename_1": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
                "filename_2": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "duanyll/huggingface"

    def load_clip(
        self,
        repo_id_1,
        repo_id_2,
        type,
        subfolder_1,
        subfolder_2,
        filename_1,
        filename_2,
    ):
        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )

        if not repo_id_1 or not repo_id_2:
            raise ValueError("Both repository IDs cannot be empty.")

        if not subfolder_1:
            subfolder_1 = None

        if not subfolder_2:
            subfolder_2 = None

        if not filename_1:
            filename_1 = None

        if not filename_2:
            filename_2 = None

        state_dict_1, metadata_1 = load_checkpoint_from_hf(
            repo_id_1, subfolder_1, filename_1, return_metadata=True
        )
        state_dict_2, metadata_2 = load_checkpoint_from_hf(
            repo_id_2, subfolder_2, filename_2, return_metadata=True
        )

        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[state_dict_1, state_dict_2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )

        return (clip,)


class HfTripleClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id_1": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                ),
                "repo_id_2": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                ),
                "repo_id_3": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                )
            },
            "optional": {
                "subfolder_1": (
                    "STRING",
                    {"default": "text_encoder", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "subfolder_2": (
                    "STRING",
                    {"default": "text_encoder_2", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "subfolder_3": (
                    "STRING",
                    {"default": "text_encoder_3", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "filename_1": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
                "filename_2": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
                "filename_3": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "duanyll/huggingface"

    def load_clip(
        self,
        repo_id_1,
        repo_id_2,
        repo_id_3,
        subfolder_1,
        subfolder_2,
        subfolder_3,
        filename_1,
        filename_2,
        filename_3,
    ):
        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )

        if not repo_id_1 or not repo_id_2 or not repo_id_3:
            raise ValueError("All three repository IDs cannot be empty.")
        

        if not subfolder_1:
            subfolder_1 = None
        if not subfolder_2:
            subfolder_2 = None
        if not subfolder_3:
            subfolder_3 = None
        
        if not filename_1:
            filename_1 = None
        if not filename_2:
            filename_2 = None
        if not filename_3:
            filename_3 = None

        state_dict_1, metadata_1 = load_checkpoint_from_hf(
            repo_id_1, subfolder_1, filename_1, return_metadata=True
        )
        state_dict_2, metadata_2 = load_checkpoint_from_hf(
            repo_id_2, subfolder_2, filename_2, return_metadata=True
        )
        state_dict_3, metadata_3 = load_checkpoint_from_hf(
            repo_id_3, subfolder_3, filename_3, return_metadata=True
        )

        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[state_dict_1, state_dict_2, state_dict_3],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        return (clip,)
    
    
class HfQuadrupleClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id_1": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                ),
                "repo_id_2": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                ),
                "repo_id_3": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                ),
                "repo_id_4": (
                    "STRING",
                    {
                        "default": "black-forest-labs/FLUX.1-dev",
                        "tooltip": REPO_TOOLTIP,
                    },
                )
            },
            "optional": {
                "subfolder_1": (
                    "STRING",
                    {"default": "text_encoder", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "subfolder_2": (
                    "STRING",
                    {"default": "text_encoder_2", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "subfolder_3": (
                    "STRING",
                    {"default": "text_encoder_3", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "subfolder_4": (
                    "STRING",
                    {"default": "text_encoder_4", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "filename_1": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
                "filename_2": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
                "filename_3": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
                "filename_4": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "duanyll/huggingface"

    def load_clip(
        self,
        repo_id_1,
        repo_id_2,
        repo_id_3,
        repo_id_4,
        subfolder_1,
        subfolder_2,
        subfolder_3,
        subfolder_4,
        filename_1,
        filename_2,
        filename_3,
        filename_4,
    ):
        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )

        if not repo_id_1 or not repo_id_2 or not repo_id_3 or not repo_id_4:
            raise ValueError("All three repository IDs cannot be empty.")
        

        if not subfolder_1:
            subfolder_1 = None
        if not subfolder_2:
            subfolder_2 = None
        if not subfolder_3:
            subfolder_3 = None
        if not subfolder_4:
            subfolder_4 = None
        
        if not filename_1:
            filename_1 = None
        if not filename_2:
            filename_2 = None
        if not filename_3:
            filename_3 = None
        if not filename_4:
            filename_4 = None

        state_dict_1, metadata_1 = load_checkpoint_from_hf(
            repo_id_1, subfolder_1, filename_1, return_metadata=True
        )
        state_dict_2, metadata_2 = load_checkpoint_from_hf(
            repo_id_2, subfolder_2, filename_2, return_metadata=True
        )
        state_dict_3, metadata_3 = load_checkpoint_from_hf(
            repo_id_3, subfolder_3, filename_3, return_metadata=True
        )
        state_dict_4, metadata_4 = load_checkpoint_from_hf(
            repo_id_4, subfolder_4, filename_4, return_metadata=True
        )

        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[state_dict_1, state_dict_2, state_dict_3, state_dict_4],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        return (clip,)