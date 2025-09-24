from .core import CoroutineNodeBase
from .utils import AnyType


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
    
    
class MapIndexed(CoroutineNodeBase):
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
        for index, item in enumerate(items):
            val = yield (function, [item, index])
            results.append(val)
        return results
    
    
class Comap(CoroutineNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "functions": ("LIST", ),
                "x": (AnyType("*"), ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }
        
    CATEGORY = "duanyll/functional/high_order"
    RETURN_TYPES = ("LIST", )
    
    def run_coroutine(self, functions, x):
        results = []
        for function in functions:
            val = yield (function, [x])
            results.append(val)
        return results
    
    
class Nest(CoroutineNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "function": ("CLOSURE",),
                "x": (AnyType("*"), ),
                "depth": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }
        
    CATEGORY = "duanyll/functional/high_order"
    RETURN_TYPES = (AnyType("*"), )
    
    def run_coroutine(self, function, x, depth):
        for _ in range(depth):
            val = yield (function, [x])
            x = val
        return val
    
    
class NestWhile(CoroutineNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "function": ("CLOSURE",),
                "x": (AnyType("*"), ),
                "predicate": ("CLOSURE",),
                "max_depth": ("INT", {"default": 10, "min": 1, "max": 100}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
        
    CATEGORY = "duanyll/functional/high_order"
    RETURN_TYPES = (AnyType("*"), )
    
    def run_coroutine(self, function, x, predicate, max_depth):
        for _ in range(max_depth):
            should_continue = yield (predicate, [x])
            if not should_continue:
                break
            x = yield (function, [x])
        return x
    
    
class Fold(CoroutineNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "function": ("CLOSURE",),
                "initial": (AnyType("*"), ),
                "items": ("LIST", ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }
        
    CATEGORY = "duanyll/functional/high_order"
    RETURN_TYPES = (AnyType("*"), )
    
    def run_coroutine(self, function, initial, items):
        accumulator = initial
        for item in items:
            accumulator = yield (function, [accumulator, item])
        return accumulator
    
    
class TakeWhile(CoroutineNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": ("LIST", ),
                "predicate": ("CLOSURE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }
        
    CATEGORY = "duanyll/functional/high_order"
    RETURN_TYPES = ("LIST", )
    
    def run_coroutine(self, items, predicate):
        results = []
        for item in items:
            should_take = yield (predicate, [item])
            if not should_take:
                break
            results.append(item)
        return results
    
    
class Select(CoroutineNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": ("LIST", ),
                "predicate": ("CLOSURE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }
        
    CATEGORY = "duanyll/functional/high_order"
    RETURN_TYPES = ("LIST", )
    
    def run_coroutine(self, items, predicate):
        results = []
        for item in items:
            should_select = yield (predicate, [item])
            if should_select:
                results.append(item)
        return results
    
    
NODE_CLASS_MAPPINGS = {
    "HighLevelMap": Map,
    "HighLevelMapIndexed": MapIndexed,
    "HighLevelComap": Comap,
    "HighLevelNest": Nest,
    "HighLevelNestWhile": NestWhile,
    "HighLevelFold": Fold,
    "HighLevelTakeWhile": TakeWhile,
    "HighLevelSelect": Select,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HighLevelMap": "Map",
    "HighLevelMapIndexed": "MapIndexed",
    "HighLevelComap": "Comap",
    "HighLevelNest": "Nest",
    "HighLevelNestWhile": "NestWhile",
    "HighLevelFold": "Fold",
    "HighLevelTakeWhile": "TakeWhile",
    "HighLevelSelect": "Select",
}