import json
import copy
from collections import deque

from .side_effects import SIDE_EFFECT_NODES


def transform_workflow(workflow: dict) -> tuple[dict, list]:
    """
    根据指定算法变换 ComfyUI 工作流，将由 __FunctionParam__ 和 __FunctionEnd__
    定义的函数子图转换为一个 __CreateClosure__ 节点。

    Args:
        workflow: 代表 ComfyUI 工作流的字典对象。

    Returns:
        一个元组，包含：
        - 变换后的工作流字典。
        - 一个包含警告信息的列表（如果有）。
    """
    warnings = []
    workflow_og = workflow
    # 创建工作流的深拷贝以避免修改原始输入对象
    workflow = copy.deepcopy(workflow)

    # 步骤 1: 对工作流进行拓扑排序并找到所有叶子节点
    adj = {node_id: [] for node_id in workflow}
    rev_adj = {node_id: [] for node_id in workflow}
    in_degree = {node_id: 0 for node_id in workflow}
    out_degree = {node_id: 0 for node_id in workflow}

    for node_id, node_data in workflow.items():
        for input_value in node_data.get("inputs", {}).values():
            # 检查输入是否为一个有效的链接 [source_id, source_output_index]
            if isinstance(input_value, list) and len(input_value) == 2:
                source_id = str(input_value[0])
                dest_id = node_id
                if source_id in workflow:
                    adj.setdefault(source_id, []).append(dest_id)
                    rev_adj.setdefault(dest_id, []).append(source_id)
                    out_degree[source_id] = out_degree.get(source_id, 0) + 1
                    in_degree[dest_id] = in_degree.get(dest_id, 0) + 1

    leaf_nodes = {node_id for node_id, degree in out_degree.items() if degree == 0}

    # 使用 Kahn 算法进行拓扑排序
    queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
    sorted_nodes = []
    while queue:
        u = queue.popleft()
        sorted_nodes.append(u)
        for v in adj.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(sorted_nodes) != len(workflow):
        warnings.append("Warning: The workflow contains cycles and cannot be fully sorted topologically.")
        return workflow_og, warnings 

    # 步骤 2: 按拓扑顺序迭代，查找 "__FunctionEnd__" 节点
    for end_node_id in sorted_nodes:
        if workflow[end_node_id].get("class_type") != "__FunctionEnd__":
            continue

        end_node = workflow[end_node_id]

        # 步骤 3: 识别子图并将其划分为 'param' 和 'capture' 两部分

        # 从 "__FunctionEnd__" 节点开始后向遍历，找到所有依赖项（即整个函数子图）
        subgraph_nodes = set()
        q = deque([end_node_id])
        visited_subgraph = {end_node_id}
        while q:
            curr_id = q.popleft()
            subgraph_nodes.add(curr_id)
            for pred_id in rev_adj.get(curr_id, []):
                if pred_id not in visited_subgraph:
                    visited_subgraph.add(pred_id)
                    q.append(pred_id)

        # 从子图内的所有 "__FunctionParam__" 节点开始前向遍历，找到 'param' 部分
        param_source_nodes = {
            n
            for n in subgraph_nodes
            if workflow[n]["class_type"] == "__FunctionParam__"
        }

        param_nodes = set()
        q = deque(list(param_source_nodes))
        visited_param = set(param_source_nodes)
        while q:
            curr_id = q.popleft()
            param_nodes.add(curr_id)
            for succ_id in adj.get(curr_id, []):
                # 后继节点必须在子图范围内且未被访问过
                if succ_id in subgraph_nodes and succ_id not in visited_param:
                    visited_param.add(succ_id)
                    q.append(succ_id)

        # "__FunctionEnd__" 节点本身始终属于 'param' 部分
        param_nodes.add(end_node_id)
        capture_nodes = subgraph_nodes - param_nodes

        # 步骤 4: 将 "__FunctionEnd__" 节点类型更改为 "__CreateClosure__"
        create_capture_node = copy.deepcopy(end_node)
        create_capture_node["class_type"] = "__CreateClosure__"
        create_capture_node["_meta"]["title"] = "Create Closure"
        workflow[end_node_id] = create_capture_node

        # 步骤 5: 识别捕获边，并重连接 'param' 子图的副本
        has_side_effects = False
        param_subgraph_copy = {}
        # 映射 (source_id, source_idx) -> "capture_N"，以避免重复捕获
        capture_edge_map = {}
        capture_counter = 0

        for p_node_id in param_nodes:
            original_node_data = workflow[p_node_id]
            if original_node_data.get("class_type") == "__FunctionParam__":
                continue
            if original_node_data.get("class_type") in SIDE_EFFECT_NODES:
                has_side_effects = True
            # 创建深拷贝以在 'body' 中进行修改
            node_copy = copy.deepcopy(original_node_data)

            for input_name, input_link in node_copy.get("inputs", {}).items():
                if isinstance(input_link, list):
                    if str(input_link[0]) in capture_nodes:
                        source_id, source_idx = str(input_link[0]), input_link[1]
                        edge_tuple = (source_id, source_idx)

                        if edge_tuple not in capture_edge_map:
                            capture_edge_map[edge_tuple] = capture_counter
                            # 为 "__CreateClosure__" 节点添加入边
                            create_capture_node["inputs"][f"capture_{capture_counter}"] = [
                                source_id,
                                source_idx,
                            ]
                            capture_counter += 1

                        capture_id = capture_edge_map[edge_tuple]

                        # 在副本中将输入边重定向到占位符节点
                        node_copy["inputs"][input_name] = ["__capture", capture_id]
                    elif workflow.get(str(input_link[0]), {}).get("class_type") == "__FunctionParam__":
                        param_node = workflow[str(input_link[0])]
                        param_index = param_node["inputs"].get("index", 0)
                        node_copy["inputs"][input_name] = ["__param", param_index]

            param_subgraph_copy[p_node_id] = node_copy

        body_json_string = json.dumps(param_subgraph_copy)
        create_capture_node["inputs"]["body"] = body_json_string
        if has_side_effects:
            create_capture_node["inputs"]["side_effects"] = float("NaN")
        else:
            create_capture_node["inputs"]["side_effects"] = 0.0

        # 移除原有的 "return_value" 输入
        if "return_value" in create_capture_node["inputs"]:
            del create_capture_node["inputs"]["return_value"]

    # 步骤 7: 剪枝，删除不再被任何叶子节点依赖的节点
    reachable_nodes = set()
    q = deque(list(leaf_nodes))
    visited = set(leaf_nodes)
    while q:
        curr_id = q.popleft()
        reachable_nodes.add(curr_id)

        node_data = workflow.get(curr_id)
        if node_data:
            for input_value in node_data.get("inputs", {}).values():
                # 确保只处理链接，而不是像 'body' 这样的静态值
                if isinstance(input_value, list) and len(input_value) == 2:
                    pred_id = str(input_value[0])
                    if pred_id in workflow and pred_id not in visited:
                        visited.add(pred_id)
                        q.append(pred_id)

    pruned_workflow = {
        node_id: node_data
        for node_id, node_data in workflow.items()
        if node_id in reachable_nodes
    }

    # 步骤 8: 最终检查，确保图中不包含 "__FunctionParam__" 节点
    for node_id, node_data in pruned_workflow.items():
        if node_data.get("class_type") == "__FunctionParam__":
            warnings.append(
                f"Warning: __FunctionParam__ node '{node_id}' remains in the workflow after transformation. This indicates incompelete function definations."
            )

    return pruned_workflow, warnings


if __name__ == "__main__":
    # 示例 ComfyUI 工作流 JSON 对象 (结构简化)
    # A -> B, A -> C, B -> C, C -> D, D -> E
    # C 是 __FunctionParam__, E 是 __FunctionEnd__
    # A 是捕获变量, B/C/D/E 是函数体
    example_workflow = {
        "1": {  # A: Load Checkpoint (capture)
            "inputs": {},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"},
        },
        "2": {  # B: CLIP Text Encode (param)
            "inputs": {"clip": ["1", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Prompt"},
        },
        "3": {  # C: __FunctionParam__ (param source)
            "inputs": {"index": 0},
            "class_type": "__FunctionParam__",
            "_meta": {"title": "Function Param: latent_image"},
        },
        "4": {  # D: KSampler (param)
            "inputs": {
                "model": ["1", 0],  # Input from capture node "1"
                "positive": ["2", 0],
                "latent_image": ["3", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        "5": {  # E: __FunctionEnd__ -> __CreateClosure__
            "inputs": {"return_value": ["4", 0]},
            "class_type": "__FunctionEnd__",
            "_meta": {"title": "Function End"},
        },
        "6": {  # F: Final Output Node
            "inputs": {"images": ["5", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
    }

    # 调用变换函数
    transformed_workflow, warnings_list = transform_workflow(example_workflow)

    # 打印结果
    print("--- Transformed Workflow ---")
    print(json.dumps(transformed_workflow, indent=2))

    if warnings_list:
        print("\n--- Warnings ---")
        for warning in warnings_list:
            print(warning)
