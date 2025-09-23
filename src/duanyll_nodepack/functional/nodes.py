import json

from comfy_execution.graph_utils import ExecutionBlocker

from .utils import AnyType, ContainsDynamicDict, Closure

RECURISON_LIMIT = 50
COROUTINE_LIMIT = 100

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
        if len(unique_id) > RECURISON_LIMIT:
            raise RecursionError("Recursion limit exceeded. Possible infinite recursion detected.")
        # kwargs: param_0, param_1, ...
        params = []
        for i in range(len(kwargs)):
            params.append(kwargs[f"param_{i}"])
        print(f"[FUNCTIONAL] Calling closure {closure} with params: {params}")
        graph, output = closure.create_graph(params, caller_unique_id=unique_id)
        return {
            "result": (output, ),
            "expand": graph
        }
        
          
class IntermidiateCoroutine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "return_value": (AnyType("*"),),
                "coroutine": (AnyType("*"),),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }
    
    RETURN_TYPES = (AnyType("*"), )
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/internal"
    
    def run(self, return_value, coroutine, unique_id):
        try:
            (closure, params) = coroutine.send(return_value)
        except StopIteration as e:
            return (e.value, )
        
        (base_id, index) = unique_id.rsplit("_", 1)
        index = int(index) + 1
        next_node_id = f"{base_id}_{index}"
        
        if index > COROUTINE_LIMIT:
            raise RecursionError("Coroutine recursion limit exceeded. Possible infinite recursion detected.")
        
        graph, output = closure.create_graph(params, caller_unique_id=unique_id)
        
        return {
            "result": ([next_node_id, 0], ),
            "expand": {
                **graph,
                next_node_id: {
                    "inputs": {
                        "return_value": output,
                        "coroutine": coroutine,
                    },
                    "class_type": "__IntermidiateCoroutine__",
                }
            }
        }
                
            
class CoroutineNodeBase:
    @classmethod
    def INPUT_TYPES(cls):
        raise NotImplementedError(
            "This is an abstract base class and should not be instantiated directly."
        )
        
    RETURN_TYPES = (AnyType("*"), )
    FUNCTION = "run"
    
    def run_coroutine(self, **kwargs):
        raise NotImplementedError(
            "Subclasses must implement the run_coroutine method."
        )
        yield # This is just to make this function a generator
        
    def run(self, unique_id, **kwargs):
        if isinstance(unique_id, list):
            unique_id = unique_id[0]
        coroutine = self.run_coroutine(**kwargs)
        return {
            "result": ([f"{unique_id}_0", 0], ),
            "expand": {
                f"{unique_id}_0": {
                    "inputs": {
                        "return_value": None,
                        "coroutine": coroutine,
                    },
                    "class_type": "__IntermidiateCoroutine__",
                }
            }
        }
        
        
class Map(CoroutineNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "function": ("CLOSURE",),
                "items": ("LIST", ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }
        
    CATEGORY = "duanyll/functional/high_order"
    RETURN_TYPES = ("LIST", )
    
    def run_coroutine(self, function, items):
        results = []
        for item in items:
            val = yield (function, [item])
            results.append(val)
        return results
    

NODE_CLASS_MAPPINGS = {
    "__FunctionParam__": FunctionParam,
    "__FunctionEnd__": FunctionEnd,
    "__CreateClosure__": CreateClosure,
    "CallClosure": CallClosure,
    "__IntermidiateCoroutine__": IntermidiateCoroutine,
    "Map": Map,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "__FunctionParam__": "Function Parameter",
    "__FunctionEnd__": "Function End",
    "__CreateClosure__": "Create Closure",
    "CallClosure": "Call Closure",
    "__IntermidiateCoroutine__": "Intermidiate Coroutine",
    "Map": "Map",
}
