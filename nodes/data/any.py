class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


class AsAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (AnyType("*"),),
            }
        }

    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/data"

    def run(self, any):
        return (any,)