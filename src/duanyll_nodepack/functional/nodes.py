import json

from comfy_execution.graph_utils import ExecutionBlocker

from .utils import AnyType, ContainsAnyDict, Closure

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
                "body": ("STRING", ),
            },
            "optional": ContainsAnyDict(),
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
                "closure": ("CLOSURE", ),
            },
            "optional": ContainsAnyDict(),
        }

    RETURN_TYPES = (AnyType("*"), )
    FUNCTION = "run"
    CATEGORY = "duanyll/functional"

    def run(self, closure, **kwargs):
        # kwargs: param_0, param_1, ...
        params = []
        for i in range(len(kwargs)):
            params.append(kwargs[f"param_{i}"])
        graph, output = closure.create_graph(params)
        return {
            "result": (output, ),
            "expand": graph
        }
