import os
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
    """Finds the correct checkpoint file from a Hugging Face repository."""
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


def get_checkpoint_file_from_local(
    base_path: str, subfolder: str = None, filename: str = None
) -> str:
    """Finds the correct checkpoint file from a local filesystem path."""
    search_path = base_path
    if subfolder:
        search_path = os.path.join(base_path, subfolder)

    if not os.path.isdir(search_path):
        raise FileNotFoundError(f"Directory not found: {search_path}")

    if filename is not None:
        return os.path.join(search_path, filename)

    files = os.listdir(search_path)
    suffixes = [".index.json", ".safetensors", ".sft", ".bin", ".pt"]

    for suffix in suffixes:
        matching_files = [f for f in files if f.endswith(suffix)]
        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple files found with suffix '{suffix}' in local directory '{search_path}'. Please specify a filename."
            )
        elif len(matching_files) == 1:
            return os.path.join(search_path, matching_files[0])

    raise ValueError(
        f"No files found with the expected suffixes in local directory '{search_path}'. Please specify a filename."
    )


def load_checkpoint_from_hf(
    repo_id: str, subfolder: str = None, filename: str = None, return_metadata=False
):
    """Loads a checkpoint from Hugging Face, handling sharded and single files."""
    checkpoint_file = get_checkpoint_file_from_hf(repo_id, subfolder, filename)

    if checkpoint_file.endswith(".index.json"):
        # Download and load the index file to find all checkpoint shards
        index_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=checkpoint_file
        )
        with open(index_path, "r") as f:
            index_data = json.load(f)
        
        # Download and load each shard
        checkpoint_files = set(index_data["weight_map"].values())
        merged_state_dict = {}
        for file in checkpoint_files:
            full_filename = os.path.join(os.path.dirname(checkpoint_file), file)
            file_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id, filename=full_filename
            )
            state_dict = comfy.utils.load_torch_file(file_path)
            merged_state_dict.update(state_dict) # Use update for correct merging
        
        return (merged_state_dict, None) if return_metadata else merged_state_dict
    else:
        # Directly download and load the single checkpoint file
        file_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=checkpoint_file
        )
        state_dict, metadata = comfy.utils.load_torch_file(
            file_path, return_metadata=True
        )
        return (state_dict, metadata) if return_metadata else state_dict


def load_checkpoint_from_local(
    base_path: str, subfolder: str = None, filename: str = None, return_metadata=False
):
    """Loads a checkpoint from the local filesystem, handling sharded and single files."""
    checkpoint_path = get_checkpoint_file_from_local(base_path, subfolder, filename)

    if checkpoint_path.endswith(".index.json"):
        # Load the index file to find all checkpoint shards
        with open(checkpoint_path, "r") as f:
            index_data = json.load(f)

        # Load each shard from the local filesystem
        checkpoint_files = set(index_data["weight_map"].values())
        merged_state_dict = {}
        index_dir = os.path.dirname(checkpoint_path)
        for file in checkpoint_files:
            shard_path = os.path.join(index_dir, file)
            state_dict = comfy.utils.load_torch_file(shard_path)
            merged_state_dict.update(state_dict) # Use update for correct merging
        
        return (merged_state_dict, None) if return_metadata else merged_state_dict
    else:
        # Directly load the single checkpoint file
        state_dict, metadata = comfy.utils.load_torch_file(
            checkpoint_path, return_metadata=True
        )
        return (state_dict, metadata) if return_metadata else state_dict

def load_checkpoint_from_hf_or_local(
    repo_id: str, subfolder: str = None, filename: str = None, return_metadata=False
):

    # Determine if repo_id is a local path or HF repo
    is_local = repo_id.startswith("./") or os.path.isabs(repo_id)

    if is_local:
        # Resolve relative path from ComfyUI root
        if repo_id.startswith("./"):
            base_path = os.path.abspath(os.path.join(folder_paths.base_path, repo_id))
        else:
            base_path = repo_id
        
        return load_checkpoint_from_local(
            base_path, subfolder, filename, return_metadata=return_metadata
        )
    else:
        return load_checkpoint_from_hf(
            repo_id, subfolder, filename, return_metadata=return_metadata
        )

REPO_TOOLTIP = "The Hugging Face repo ID, an absolute path (/path/to/model), or a relative path (./models/...) from the ComfyUI root."
SUBFOLDER_TOOLTIP = "The subfolder where the model is located. Applies to both Hugging Face repos and local paths."
FILENAME_TOOLTIP = "The checkpoint file to load. If not specified, will auto-detect. For sharded checkpoints, use the index file (e.g., 'model.index.json')."
DESCRIPTION = "Downloads a model from Hugging Face or loads it from a local path. Supports single-file and sharded checkpoints."


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
    CATEGORY = "duanyll/loaders"
    DESCRIPTION = DESCRIPTION

    def load_checkpoint(self, repo_id, subfolder, filename):
        if not repo_id:
            raise ValueError("Repository ID cannot be empty.")

        if not subfolder:
            subfolder = None

        if not filename:
            filename = None

        state_dict, metadata = load_checkpoint_from_hf_or_local(
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
    CATEGORY = "duanyll/loaders"
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

        state_dict, metadata = load_checkpoint_from_hf_or_local(
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
    CATEGORY = "duanyll/loaders"
    DESCRIPTION = DESCRIPTION

    def load_vae(self, repo_id, subfolder, filename):
        if not repo_id:
            raise ValueError("Repository ID cannot be empty.")

        if not subfolder:
            subfolder = None

        if not filename:
            filename = None

        state_dict, metadata = load_checkpoint_from_hf_or_local(
            repo_id, subfolder, filename, return_metadata=True
        )

        vae = comfy.sd.VAE(sd=state_dict)
        vae.throw_exception_if_invalid()

        return (vae,)


class HfClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": ("STRING", ),
                "type": ("STRING", ),
            },
            "optional": {
                "subfolder": (
                    "STRING",
                    {"default": "text_encoder", "tooltip": SUBFOLDER_TOOLTIP},
                ),
                "filename": ("STRING", {"default": "", "tooltip": FILENAME_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "duanyll/loaders"

    DESCRIPTION = DESCRIPTION

    def load_clip(self, repo_id, type, subfolder, filename):
        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )

        if not repo_id:
            raise ValueError("Repository ID cannot be empty.")

        if not subfolder:
            subfolder = None

        if not filename:
            filename = None

        state_dict, metadata = load_checkpoint_from_hf_or_local(
            repo_id, subfolder, filename, return_metadata=True
        )

        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[state_dict],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )
        return (clip,)


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
    CATEGORY = "duanyll/loaders"
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

        state_dict, metadata = load_checkpoint_from_hf_or_local(
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
    CATEGORY = "duanyll/loaders"
    DESCRIPTION = DESCRIPTION

    def load_lora(self, model, repo_id, strength_model, subfolder, filename):
        if not repo_id:
            raise ValueError("Repository ID cannot be empty.")

        if not subfolder:
            subfolder = None

        if not filename:
            filename = None

        state_dict, metadata = load_checkpoint_from_hf_or_local(
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
    CATEGORY = "duanyll/loaders"

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

        state_dict_1, metadata_1 = load_checkpoint_from_hf_or_local(
            repo_id_1, subfolder_1, filename_1, return_metadata=True
        )
        state_dict_2, metadata_2 = load_checkpoint_from_hf_or_local(
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
    CATEGORY = "duanyll/loaders"

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

        state_dict_1, metadata_1 = load_checkpoint_from_hf_or_local(
            repo_id_1, subfolder_1, filename_1, return_metadata=True
        )
        state_dict_2, metadata_2 = load_checkpoint_from_hf_or_local(
            repo_id_2, subfolder_2, filename_2, return_metadata=True
        )
        state_dict_3, metadata_3 = load_checkpoint_from_hf_or_local(
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
    CATEGORY = "duanyll/loaders"

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

        state_dict_1, metadata_1 = load_checkpoint_from_hf_or_local(
            repo_id_1, subfolder_1, filename_1, return_metadata=True
        )
        state_dict_2, metadata_2 = load_checkpoint_from_hf_or_local(
            repo_id_2, subfolder_2, filename_2, return_metadata=True
        )
        state_dict_3, metadata_3 = load_checkpoint_from_hf_or_local(
            repo_id_3, subfolder_3, filename_3, return_metadata=True
        )
        state_dict_4, metadata_4 = load_checkpoint_from_hf_or_local(
            repo_id_4, subfolder_4, filename_4, return_metadata=True
        )

        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[state_dict_1, state_dict_2, state_dict_3, state_dict_4],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        return (clip,)