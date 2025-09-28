import time
import json

from comfy_execution.graph_utils import ExecutionBlocker

from .utils import AnyType, Closure

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
    
    
class Inspect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (AnyType("*"),),
                "value": (AnyType("*"),),
            }
        }
        
    RETURN_TYPES = (AnyType("*"), AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/side_effects"
    
    def run(self, signal, value):
        return (signal, ExecutionBlocker("You should attach an output node with exactly one input to this node."))
    
    
class InspectPassthru:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (AnyType("*"),),
                "value": (AnyType("*"),),
            }
        }
        
    RETURN_TYPES = (AnyType("*"), AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/internal"
    
    def run(self, signal, value):
        return (signal[0], value[0])
    
    
class InspectImpl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (AnyType("*"),),
                "value": (AnyType("*"),),
                "body": ("STRING", {"multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
        
    RETURN_TYPES = (AnyType("*"), AnyType("*"))
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/internal"
    
    def run(self, signal, value, body, unique_id):
        body = json.loads(body)
        graph = {}
        passthru_id = f"{unique_id}_passthru"
        graph[passthru_id] = {
            "inputs": {
                "signal": (signal,),
                "value": (value,),
            },
            "class_type": "__InspectPassthru__",
        }
        for node_id, node_data in body.items():
            inputs = node_data["inputs"]
            for key in inputs.keys():
                spec = inputs[key]
                if isinstance(spec, list) and spec[0] == "__value":
                    inputs[key] = [passthru_id, 1]
            node_data["override_display_id"] = node_id
            graph[f"{unique_id}_{node_id}"] = node_data
        return {
            "result": ([passthru_id, 0], [passthru_id, 1]),
            "expand": graph,
        }
        
    
NODE_CLASS_MAPPINGS = {
    "__Sow__": Sow,
    "__Reap__": Reap,
    "__Inspect__": Inspect,
    "__InspectPassthru__": InspectPassthru,
    "__InspectImpl__": InspectImpl,
    "Latch": Latch,
    "Sleep": Sleep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "__Sow__": "Sow",
    "__Reap__": "Reap",
    "__Inspect__": "Inspect",
    "__InspectPassthru__": "Inspect (Passthru)",
    "__InspectImpl__": "Inspect (Impl)",
    "Latch": "Latch",
    "Sleep": "Sleep",
}

SIDE_EFFECT_NODES = [
    "__Sow__", "__Reap__"
]