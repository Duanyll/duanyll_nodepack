import json
import copy
from collections import deque
from typing import Dict, List, Set, Tuple


from .side_effects import SIDE_EFFECT_NODES
# SIDE_EFFECT_NODES = {}


class WorkflowGraph:
    """一个封装了工作流图结构和操作的类。"""

    def __init__(self, workflow: Dict):
        self.workflow = workflow
        self.adj: Dict[str, List[str]] = {node_id: [] for node_id in workflow}
        self.rev_adj: Dict[str, List[str]] = {node_id: [] for node_id in workflow}
        self.in_degree: Dict[str, int] = {node_id: 0 for node_id in workflow}
        self.out_degree: Dict[str, int] = {node_id: 0 for node_id in workflow}
        self._build()

    def _build(self):
        """从工作流字典构建图的邻接表和度信息。"""
        for node_id, node_data in self.workflow.items():
            for input_value in node_data.get("inputs", {}).values():
                if isinstance(input_value, list) and len(input_value) == 2:
                    source_id = str(input_value[0])
                    if source_id in self.workflow:
                        self.add_edge(source_id, node_id)

    def add_edge(self, source_id: str, dest_id: str):
        """向图中添加一条边并更新所有相关结构。"""
        if dest_id not in self.adj.get(source_id, []):
            self.adj.setdefault(source_id, []).append(dest_id)
            self.rev_adj.setdefault(dest_id, []).append(source_id)
            self.out_degree[source_id] = self.out_degree.get(source_id, 0) + 1
            self.in_degree[dest_id] = self.in_degree.get(dest_id, 0) + 1

    def remove_edge(self, source_id: str, dest_id: str):
        """从图中移除一条边并更新所有相关结构。"""
        if source_id in self.adj and dest_id in self.adj[source_id]:
            self.adj[source_id].remove(dest_id)
            self.out_degree[source_id] -= 1
        if dest_id in self.rev_adj and source_id in self.rev_adj[dest_id]:
            self.rev_adj[dest_id].remove(source_id)
            self.in_degree[dest_id] -= 1

    def topological_sort(self) -> Tuple[List[str], bool]:
        """使用 Kahn 算法进行拓扑排序。返回排序后的节点列表和是否成功的标志。"""
        sorted_nodes = []
        q = deque([node_id for node_id, degree in self.in_degree.items() if degree == 0])
        
        temp_in_degree = self.in_degree.copy()

        while q:
            u = q.popleft()
            sorted_nodes.append(u)
            for v in self.adj.get(u, []):
                temp_in_degree[v] -= 1
                if temp_in_degree[v] == 0:
                    q.append(v)
        
        is_valid = len(sorted_nodes) == len(self.workflow)
        return sorted_nodes, is_valid

    def get_leaf_nodes(self) -> Set[str]:
        """获取图中所有的叶子节点（出度为0）。"""
        return {node_id for node_id, degree in self.out_degree.items() if degree == 0}


def _find_function_subgraph(end_node_id: str, graph: WorkflowGraph) -> Set[str]:
    """从 __FunctionEnd__ 节点开始后向遍历，找到完整的函数子图。"""
    subgraph_nodes = set()
    q = deque([end_node_id])
    visited = {end_node_id}
    while q:
        curr_id = q.popleft()
        subgraph_nodes.add(curr_id)
        for pred_id in graph.rev_adj.get(curr_id, []):
            if pred_id not in visited:
                visited.add(pred_id)
                q.append(pred_id)
    return subgraph_nodes


def _find_param_nodes(
    subgraph_nodes: Set[str], workflow: Dict, graph: WorkflowGraph
) -> Tuple[Set[str], Set[str]]:
    param_source_nodes = {
        n for n in subgraph_nodes if workflow[n].get("class_type") == "__FunctionParam__"
    }

    param_nodes = set()
    q = deque(list(param_source_nodes))
    visited = set(param_source_nodes)
    while q:
        curr_id = q.popleft()
        param_nodes.add(curr_id)
        for succ_id in graph.adj.get(curr_id, []):
            if succ_id in subgraph_nodes and succ_id not in visited:
                visited.add(succ_id)
                q.append(succ_id)

    return param_nodes


def _transform_single_function(end_node_id: str, workflow: Dict, graph: WorkflowGraph):
    """转换单个 __FunctionEnd__ 节点及其关联的子图。"""
    subgraph_nodes = _find_function_subgraph(end_node_id, graph)
    original_end_node = workflow[end_node_id]
    
    param_nodes = _find_param_nodes(subgraph_nodes, workflow, graph)
    param_nodes.add(end_node_id)
    capture_nodes = subgraph_nodes - param_nodes

    # 恢复原始节点类型以便深拷贝
    workflow[end_node_id]["class_type"] = original_end_node["class_type"]

    # 步骤 4: 创建 __CreateClosure__ 节点
    closure_node = copy.deepcopy(original_end_node)
    closure_node["class_type"] = "__CreateClosure__"
    closure_node["_meta"] = {"title": "Create Closure"}
    workflow[end_node_id] = closure_node

    # 步骤 5 & 6: 处理捕获边和函数体
    has_side_effects = False
    param_subgraph_copy = {}
    capture_edge_map = {}
    capture_counter = 0

    for p_node_id in param_nodes:
        if workflow[p_node_id]["class_type"] == "__FunctionParam__":
            continue
        if workflow[p_node_id]["class_type"] in SIDE_EFFECT_NODES:
            has_side_effects = True
        
        node_copy = copy.deepcopy(workflow[p_node_id])
        
        for input_name, input_link in list(node_copy.get("inputs", {}).items()):
            if not (isinstance(input_link, list) and len(input_link) == 2):
                continue

            source_id_str = str(input_link[0])
            
            if source_id_str in capture_nodes:
                source_id, source_idx = source_id_str, input_link[1]
                edge_tuple = (source_id, source_idx)

                if edge_tuple not in capture_edge_map:
                    capture_edge_map[edge_tuple] = capture_counter
                    closure_node["inputs"][f"capture_{capture_counter}"] = [source_id, source_idx]
                    graph.add_edge(source_id, end_node_id)
                    capture_counter += 1
                
                capture_id = capture_edge_map[edge_tuple]
                node_copy["inputs"][input_name] = ["__capture", capture_id]

            elif workflow.get(source_id_str, {}).get("class_type") == "__FunctionParam__":
                param_node = workflow[source_id_str]
                param_index = param_node["inputs"].get("index", 0)
                node_copy["inputs"][input_name] = ["__param", param_index]
        
        param_subgraph_copy[p_node_id] = node_copy

    output_edge = param_subgraph_copy[end_node_id]["inputs"].get("return_value")
    closure_node["inputs"]["output"] = json.dumps(output_edge) 
    del param_subgraph_copy[end_node_id]
    closure_node["inputs"]["body"] = json.dumps(param_subgraph_copy)
    closure_node["inputs"]["side_effects"] = float("NaN") if has_side_effects else 0.0

    if "return_value" in closure_node["inputs"]:
        return_link = closure_node["inputs"]["return_value"]
        if isinstance(return_link, list) and len(return_link) == 2:
            source_id = str(return_link[0])
            graph.remove_edge(source_id, end_node_id)
        del closure_node["inputs"]["return_value"]

def _prune_unreachable_nodes(workflow: Dict, leaf_nodes: Set[str]) -> Dict:
    """从叶子节点回溯，移除所有不可达的节点。"""
    reachable_nodes = set()
    q = deque(list(leaf_nodes))
    visited = set(leaf_nodes)
    while q:
        curr_id = q.popleft()
        reachable_nodes.add(curr_id)

        node_data = workflow.get(curr_id)
        if node_data:
            for input_value in node_data.get("inputs", {}).values():
                if isinstance(input_value, list) and len(input_value) == 2:
                    pred_id = str(input_value[0])
                    if pred_id in workflow and pred_id not in visited:
                        visited.add(pred_id)
                        q.append(pred_id)

    return {
        node_id: node_data
        for node_id, node_data in workflow.items()
        if node_id in reachable_nodes
    }

def _preprocess_inspect_nodes(workflow: Dict, execute_outputs: list[str]):
    """
    预处理 __Inspect__ 节点。
    查找所有仅由 __Inspect__ 节点连接的叶子节点，将其"吸收"到 __Inspect__ 节点的 body 中，
    然后将 __Inspect__ 节点转换为 __InspectImpl__ 节点。
    此函数会直接在传入的 workflow 字典上进行修改。
    """
    temp_graph = WorkflowGraph(workflow)
    leaf_nodes = temp_graph.get_leaf_nodes()
    
    nodes_to_remove = set()
    modified_inspect_nodes: Dict[str, Dict] = {}

    for leaf_id in leaf_nodes:
        parents = temp_graph.rev_adj.get(leaf_id, [])
        # 条件：叶子节点，且只有一个父节点，且父节点类型为 __Inspect__
        if len(parents) == 1:
            parent_id = parents[0]
            parent_node = workflow.get(parent_id)
            
            if not (parent_node and parent_node.get("class_type") == "__Inspect__"):
                continue

            # 新增检查：确认叶子节点连接的是 __Inspect__ 的 value 输出（索引 1）
            is_connected_to_value_output = False
            leaf_node = workflow[leaf_id]
            for input_link in leaf_node.get("inputs", {}).values():
                if (isinstance(input_link, list) and 
                        str(input_link[0]) == parent_id and 
                        input_link[1] == 1):
                    is_connected_to_value_output = True
                    break
            
            # 如果不是连接到 value 输出，则跳过此叶子节点
            if not is_connected_to_value_output:
                continue

            # --- 确认无误后，开始转换逻辑 ---
            if parent_id not in modified_inspect_nodes:
                modified_inspect_nodes[parent_id] = {}
            
            body = modified_inspect_nodes[parent_id]
            leaf_node_copy = copy.deepcopy(leaf_node)
            
            # 查找并替换来自父节点的输入边
            for input_name, input_link in leaf_node_copy.get("inputs", {}).items():
                if (isinstance(input_link, list) and 
                        str(input_link[0]) == parent_id and 
                        input_link[1] == 1):
                    leaf_node_copy["inputs"][input_name] = ["__value", 0]
                    break
            
            body[leaf_id] = leaf_node_copy
            nodes_to_remove.add(leaf_id)


    # 将更改应用到工作流
    # 1. 更新被修改的 __Inspect__ 节点
    for inspect_id, body_dict in modified_inspect_nodes.items():
        inspect_node = workflow[inspect_id]
        inspect_node["class_type"] = "__InspectImpl__"
        inspect_node["inputs"]["body"] = json.dumps(body_dict)

    # 2. 移除被吸收的叶子节点
    for node_id in nodes_to_remove:
        if node_id in workflow:
            del workflow[node_id]
        # Remove from execute_outputs if present
        if node_id in execute_outputs:
            execute_outputs.remove(node_id)
            

def transform_workflow(workflow: dict, execute_outputs: list[str]) -> tuple[dict, list]:
    """
    根据指定算法变换 ComfyUI 工作流。
    此算法首先预处理 __Inspect__ 节点以支持动态输出，然后将由 
    __FunctionParam__ 和 __FunctionEnd__ 定义的函数子图转换为一个 __CreateClosure__ 节点。

    Args:
        workflow: 代表 ComfyUI 工作流的字典对象。

    Returns:
        一个元组，包含：
        - 变换后的工作流字典。
        - 一个包含警告信息的列表（如果有）。
    """
    warnings = []
    workflow_og = workflow
    workflow = copy.deepcopy(workflow_og)
    
    # 步骤 2: 预处理 __Inspect__ 节点
    _preprocess_inspect_nodes(workflow, execute_outputs)
    
    # 步骤 3: 从可能已修改的工作流重建图结构以进行后续操作
    graph = WorkflowGraph(workflow)
    sorted_nodes, is_valid_dag = graph.topological_sort()
    if not is_valid_dag:
        warnings.append("Warning: The workflow contains cycles and cannot be fully sorted topologically.")
        return workflow_og, warnings
    
    # 在函数变换之前，确定最终用于剪枝的叶子节点集合
    leaf_nodes_for_pruning = graph.get_leaf_nodes()

    # 步骤 4: 按拓扑顺序迭代，转换所有函数定义
    for node_id in sorted_nodes:
        if workflow[node_id].get("class_type") == "__FunctionEnd__":
            _transform_single_function(node_id, workflow, graph)

    # 步骤 5: 剪枝，删除不再被任何叶子节点依赖的节点
    pruned_workflow = _prune_unreachable_nodes(workflow, leaf_nodes_for_pruning)

    # 步骤 6: 最终检查，确保图中不包含 __FunctionParam__ 节点
    for node_id, node_data in pruned_workflow.items():
        if node_data.get("class_type") == "__FunctionParam__":
            warnings.append(
                f"Warning: __FunctionParam__ node '{node_id}' remains in the workflow. "
                "This indicates incomplete function definitions."
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
            "class_type": "CheckpointLoaderSimple·",
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
