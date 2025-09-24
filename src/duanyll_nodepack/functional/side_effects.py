import time

from .utils import AnyType

reap_storage = {}

class Sow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (AnyType("*"),),
                "value": (AnyType("*"),),
                "tag": ("STRING", {"default": "default", "multiline": False}),
            },
        }
        
    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/side_effects"
    
    @classmethod
    def IS_CHANGED(cls, signal, value, tag):
        return float("NaN")
    
    def run(self, signal, value, tag):
        global reap_storage
        if not tag in reap_storage:
            reap_storage[tag] = []
        reap_storage[tag].append(value)
        return (signal,)
    

class Reap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (AnyType("*"),),
                "tag": ("STRING", {"default": "default", "multiline": False}),
            },
        }
        
    RETURN_TYPES = ("LIST",)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/side_effects"
    
    @classmethod
    def IS_CHANGED(cls, signal, tag):
        return float("NaN")
    
    def run(self, signal, tag):
        global reap_storage
        if tag in reap_storage:
            values = reap_storage[tag]
            reap_storage[tag] = []
        else:
            values = []
        return (values,)
    
    
def reset_reap_storage():
    global reap_storage
    reap_storage = {}
    
    
class Latch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (AnyType("*"),),
                "value": (AnyType("*"),),
            },
        }
        
    RETURN_TYPES = (AnyType("*"), )
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/side_effects"
    
    def run(self, signal, value):
        return (value, )
    
    
class Sleep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (AnyType("*"),),
                "seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            },
        }
        
    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/side_effects"
    
    def run(self, signal, seconds):
        time.sleep(seconds)
        return (signal,)
    
    
NODE_CLASS_MAPPINGS = {
    "Sow": Sow,
    "Reap": Reap,
    "Latch": Latch,
    "Sleep": Sleep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sow": "Sow",
    "Reap": "Reap",
    "Latch": "Latch",
    "Sleep": "Sleep",
}