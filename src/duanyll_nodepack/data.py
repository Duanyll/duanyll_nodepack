import copy
import re
import json
import json5
import jsonpath_ng


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


def _extract_last_json_block(text: str) -> str:
    """
    从文本中提取最后一个被 markdown 包围的 JSON 代码块。
    支持 ```json ... ``` 和 ``` ... ``` 格式。
    """
    # 正则表达式查找 markdown 代码块
    # (?:json)? 表示 "json" 这个词是可选的
    # \s* 匹配任何空白字符（包括换行符）
    # ([\s\S]*?) 非贪婪地匹配所有字符，直到下一个 ```
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text)

    if matches:
        # 如果找到，返回最后一个匹配项
        return matches[-1]

    # 如果没有找到 markdown 块，则返回原始文本
    return text


def _find_last_json_object(text: str) -> str:
    """
    从字符串末尾开始查找第一个完整匹配的 JSON 对象或数组。
    """
    text = text.strip()

    # 查找最后一个 '}' 或 ']'
    last_brace = text.rfind("}")
    last_bracket = text.rfind("]")

    if last_brace == -1 and last_bracket == -1:
        return ""

    last_end_char_index = max(last_brace, last_bracket)
    end_char = text[last_end_char_index]
    start_char = "{" if end_char == "}" else "["

    level = 0
    # 从后向前遍历，寻找匹配的起始符号
    for i in range(last_end_char_index, -1, -1):
        char = text[i]
        if char == end_char:
            level += 1
        elif char == start_char:
            level -= 1
            if level == 0:
                # 找到了匹配的起始位置
                return text[i : last_end_char_index + 1]

    return ""  # 没有找到完整的 JSON 对象


def parse_llm_json_output(llm_output: str) -> dict | list | None:
    """
    从 LLM 的输出中稳健地解析最后一个 JSON 对象。

    处理逻辑:
    1. 优先尝试从 Markdown 代码块 (```json ... ```) 中提取 JSON。
    2. 如果没有代码块，则从整个字符串的末尾寻找最后一个完整的 JSON 对象/数组。
    3. 使用更宽松的 `json5` 库进行解析，以处理注释、末尾逗号等情况。
    4. 如果解析失败，则返回 None。

    :param llm_output: LLM 返回的可能包含 JSON 的字符串。
    :return: 解析后的 Python 字典或列表，如果失败则返回 None。
    """
    # 1. 优先从 markdown 块提取
    potential_json_str = _extract_last_json_block(llm_output)

    # 2. 从提取的字符串（或原始字符串）中定位最后一个 JSON 对象
    json_str = _find_last_json_object(potential_json_str)

    if not json_str:
        # 如果 markdown 提取和定位都失败了，最后尝试对原始文本进行一次定位
        json_str = _find_last_json_object(llm_output)
        if not json_str:
            return None

    # 3. 使用 json5 进行解析
    try:
        # 使用 json5.loads，它可以处理注释、末尾逗号等
        return json5.loads(json_str)
    except Exception as e:
        print(f"Failed to parse JSON with json5. Error: {e}")
        # 可以选择在这里增加一个使用标准库 json 的回退尝试
        try:
            # 去掉常见的注释（一个简单的实现）
            no_comments = re.sub(r"//.*", "", json_str)
            no_comments = re.sub(r"/\*[\s\S]*?\*/", "", no_comments, flags=re.MULTILINE)
            return json.loads(no_comments)
        except json.JSONDecodeError as final_e:
            print(
                f"Failed to parse with standard json library after cleaning. Error: {final_e}"
            )
            return None


class ParseLlmJsonOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_output": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/data"

    def run(self, llm_output):
        return (parse_llm_json_output(llm_output),)


class JsonPathQuery:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": (AnyType("*"),),
                "query": ("STRING", {"default": ""}),
            },
            "optional": {
                "raise_errors": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/data"
    OUTPUT_IS_LIST = (True, )

    def run(self, json_data, query, raise_errors=False):
        expr = jsonpath_ng.parse(query)
        result = []
        try:
            result = [match.value for match in expr.find(json_data)]
        except:
            if raise_errors:
                raise
        return (result,)
    
    
class JsonPathQuerySingle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": (AnyType("*"),),
                "query": ("STRING", {"default": ""}),
            },
            "optional": {
                "raise_errors": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/data"

    def run(self, json_data, query, raise_errors=False):
        expr = jsonpath_ng.parse(query)
        result = None
        try:
            result = expr.find(json_data)[0].value
        except:
            if raise_errors:
                raise
        return (result,)


class JsonPathUpdate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": (AnyType("*"),),
                "value": (AnyType("*"),),
                "query": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/data"

    def run(self, json_data, value, query):
        expr = jsonpath_ng.parse(query)
        # Deep copy json_data before update
        data_copy = copy.deepcopy(json_data)
        expr.update(data_copy, value)
        return (data_copy,)