import copy

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


class ContainsDynamicDict(dict):
    """
    A custom dictionary that dynamically returns values for keys based on a pattern.
    - If a key in the passed dictionary has a value with `{"_dynamic": "number"}` in the tuple's second position,
      then any other key starting with the same string and ending with a number will return that value.
    - For other keys, normal dictionary lookup behavior applies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store prefixes associated with `_dynamic` values for efficient lookup
        self._dynamic_prefixes = {
            key.rstrip("0123456789"): value
            for key, value in self.items()
            if isinstance(value, tuple)
            and len(value) > 1
            and value[1].get("_dynamic") == "number"
        }

    def __contains__(self, key):
        # Check if key matches a dynamically handled prefix or exists normally
        return any(
            key.startswith(prefix) and key[len(prefix) :].isdigit()
            for prefix in self._dynamic_prefixes
        ) or super().__contains__(key)

    def __getitem__(self, key):
        # Dynamically return the value for keys matching a `prefix<number>` pattern
        for prefix, value in self._dynamic_prefixes.items():
            if key.startswith(prefix) and key[len(prefix) :].isdigit():
                return value
        # Fallback to normal dictionary behavior for other keys
        return super().__getitem__(key)


class Closure:
    def __init__(self, body, output, captures):
        self.body = body
        self.output = output
        self.captures = captures

    def create_graph(self, params, caller_unique_id=None):
        body = copy.deepcopy(self.body)
        graph = {}

        def add_recover_node(spec):
            prefix = f"{caller_unique_id}_" if caller_unique_id is not None else ""
            recover_id = f"{prefix}{spec[0]}_{spec[1]}"
            if recover_id in graph:
                return [recover_id, 0]
            value = self.captures[spec[1]] if spec[0] == "__capture" else params[spec[1]]
            graph[recover_id] = {
                "inputs": {
                    # Wrap the list in a tuple so it's treated as a literal, not a link
                    "values": (value,),
                },
                "class_type": "__RecoverList__",
            }
            return [recover_id, 0]
        
        for node_id, node_data in body.items():
            inputs = node_data["inputs"]
            for key in inputs.keys():
                spec = inputs[key]
                if isinstance(spec, list):
                    if spec[0] == "__capture":
                        value = self.captures[spec[1]]
                        if isinstance(value, list):
                            inputs[key] = add_recover_node(spec)
                        else:
                            inputs[key] = value
                    elif spec[0] == "__param":
                        if spec[1] >= len(params):
                            raise ValueError(f"Parameter index {spec[1]} out of range for provided {len(params)} params.")
                        value = params[spec[1]]
                        if isinstance(value, list):
                            inputs[key] = add_recover_node(spec)
                        else:
                            inputs[key] = value
                    else:
                        spec[0] = f"{caller_unique_id}_{spec[0]}"
            node_data["override_display_id"] = node_id
            graph[f"{caller_unique_id}_{node_id}"] = node_data
        spec = self.output
        if spec[0] == "__capture":
            value = self.captures[spec[1]]
            if isinstance(value, list):
                return graph, add_recover_node(spec)
            return graph, value
        elif spec[0] == "__param":
            if spec[1] >= len(params):
                raise ValueError(f"Parameter index {spec[1]} out of range for provided {len(params)} params.")
            value = params[spec[1]]
            if isinstance(value, list):
                return graph, add_recover_node(spec)
            return graph, value
        else:
            return graph, [f"{caller_unique_id}_{spec[0]}", spec[1]]
