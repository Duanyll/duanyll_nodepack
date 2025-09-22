import copy

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


class ContainsAnyDict(dict):
    def __contains__(self, key):
        return True


class Closure:
    def __init__(self, body, captures):
        self.body = body
        self.captures = captures
        
    def create_graph(self, params):
        body = copy.deepcopy(self.body)
        graph = {}
        output = None
        for node_id, node_data in body.items():
            if node_data["class_type"] == "__FunctionEnd__":
                output = node_data["inputs"]["return_value"]
            else:
                inputs = node_data["inputs"]
                for key in inputs.keys():
                    spec = inputs[key]
                    if isinstance(spec, list):
                        if spec[0] == "__capture":
                            inputs[key] = self.captures[spec[1]]
                        elif spec[0] == "__param":
                            if spec[1] >= len(params):
                                raise ValueError(f"Parameter index {spec[1]} out of range for provided params.")
                            inputs[key] = params[spec[1]]
                graph[node_id] = node_data
        return graph, output
