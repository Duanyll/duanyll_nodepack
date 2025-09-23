import json

from comfy_execution.graph_utils import ExecutionBlocker

from .utils import AnyType, ContainsDynamicDict, Closure

class FunctionParam:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional"
    
    def run(self, index):
        return (ExecutionBlocker("This node is a placeholder and should never be executed. This error indicates that the `FunctionEnd` node is missing or not properly connected."),)

class FunctionEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "return_value": (AnyType("*"),),
            },
        }
        
    RETURN_TYPES = ("CLOSURE", )
    FUNCTION = "run"
    CATEGORY = "duanyll/functional"
    
    def run(self, return_value):
        raise NotImplementedError("This node is a placeholder and should never be executed. This error indicates that the graph is not properly connected.")

class CreateClosure:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body": ("STRING", {"multiline": True} ),
            },
            "optional": ContainsDynamicDict(
                {
                    "capture_0": (AnyType("*"), {"_dynamic": "number"}),
                }
            ),
        }

    RETURN_TYPES = ("CLOSURE", )
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/internal"

    def run(self, body, **kwargs):
        # kwargs: capture_0, capture_1, ...
        captures = []
        for i in range(len(kwargs)):
            captures.append(kwargs[f"capture_{i}"])
        body = json.loads(body)
        return (Closure(body, captures), )

class CallClosure:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "closure": ("CLOSURE",),
            },
            "optional": ContainsDynamicDict(
                {
                    "param_0": (AnyType("*"), {"_dynamic": "number"}),
                }
            ),
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = (AnyType("*"), )
    FUNCTION = "run"
    CATEGORY = "duanyll/functional"

    def run(self, closure, unique_id, **kwargs):
        # kwargs: param_0, param_1, ...
        params = []
        for i in range(len(kwargs)):
            params.append(kwargs[f"param_{i}"])
        graph, output = closure.create_graph(params, caller_unique_id=unique_id)
        return {
            "result": (output, ),
            "expand": graph
        }


NODE_CLASS_MAPPINGS = {
    "__FunctionParam__": FunctionParam,
    "__FunctionEnd__": FunctionEnd,
    "__CreateClosure__": CreateClosure,
    "CallClosure": CallClosure,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "__FunctionParam__": "Function Parameter",
    "__FunctionEnd__": "Function End",
    "__CreateClosure__": "Create Closure",
    "CallClosure": "Call Closure",
}
