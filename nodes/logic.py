class LogicAnd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input1": ("BOOLEAN", ),
                "input2": ("BOOLEAN", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "logical_and"
    CATEGORY = "duanyll/logic"
    
    def check_lazy_status(self, input1, input2):
        if input1 is True:
            return ["input2"]
        else:
            return []

    def logical_and(self, input1, input2):
        if input1 is True:
            return (input2 is True, )
        else:
            return (False, )
    
    
class LogicOr:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input1": ("BOOLEAN",),
                "input2": ("BOOLEAN", {"lazy": True}),
            }
        }
        
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "logical_or"
    CATEGORY = "duanyll/logic"

    def check_lazy_status(self, input1, input2):
        if input1 is False:
            return ["input2"]
        else:
            return []

    def logical_or(self, input1, input2):
        if input1 is False:
            return (input2 is True,)
        else:
            return (True,)