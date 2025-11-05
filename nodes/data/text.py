from .any import AnyType, ContainsDynamicDict

class TextContainsChinese:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", ),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "text_contains_chinese"
    CATEGORY = "duanyll/data"
    DESCRIPTION = "Checks if the input text contains any Chinese characters."

    def text_contains_chinese(self, text):
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return (True,)
        return (False,)
    
    
class StringFormat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {"multiline": True}),
            },
            "optional": ContainsDynamicDict(
                {
                    "arg_0": (AnyType("*"), {"_dynamic": "number"}),
                }
            ),
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "string_format"
    CATEGORY = "duanyll/data"
    DESCRIPTION = "Formats a string using the provided template and arguments."

    def string_format(self, template, **kwargs):
        args = []
        for i in range(len(kwargs)):
            args.append(kwargs[f"arg_{i}"])
        formatted_string = template.format(*args)
        return (formatted_string,)