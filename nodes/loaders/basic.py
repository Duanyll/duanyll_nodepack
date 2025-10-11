import os

import folder_paths
    
class ReadTextFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dir": ("STRING", {"default": "input", "multiline": False}),
                "file": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "encoding": ("STRING", {"default": "utf-8", "multiline": False}),
                "ignore_errors": ("BOOLEAN", {"default": False})
            }
        }
        
    RETURN_TYPES = ("STRING", )
    FUNCTION = "read_text_file"
    CATEGORY = "duanyll/loaders"

    def read_text_file(self, dir: str, file: str, encoding, ignore_errors) -> str:
        fullpath = os.path.join(folder_paths.base_path, dir, file)
        try:
            with open(fullpath, "r", encoding=encoding, errors="ignore" if ignore_errors else "strict") as f:
                return (f.read(), )
        except:
            if not ignore_errors:
                raise