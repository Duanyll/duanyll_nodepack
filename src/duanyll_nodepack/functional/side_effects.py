from .utils import AnyType

reap_storage = []

class Sow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (AnyType("*"),),
                "value": (AnyType("*"),),
            },
        }
        
    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/side_effects"
    
    @classmethod
    def IS_CHANGED(cls, signal, value):
        return float("NaN")
    
    def run(self, signal, value):
        global reap_storage
        reap_storage.append(value)
        return (signal,)
    

class Reap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (AnyType("*"),),
            },
        }
        
    RETURN_TYPES = ("LIST",)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/side_effects"
    
    @classmethod
    def IS_CHANGED(cls, signal):
        return float("NaN")
    
    def run(self, signal):
        global reap_storage
        values = reap_storage
        reap_storage = []
        return (values,)
    
    
def reset_reap_storage():
    global reap_storage
    reap_storage = []
    
    
NODE_CLASS_MAPPINGS = {
    "Sow": Sow,
    "Reap": Reap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sow": "Sow",
    "Reap": "Reap",
}