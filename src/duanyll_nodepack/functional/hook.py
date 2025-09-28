from .transform import transform_workflow
from .side_effects import reset_reap_storage

def hook_comfyui_execution():
    try:
        from execution import PromptExecutor
        original_execute = PromptExecutor.execute
        def hooked_execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
            reset_reap_storage()
            prompt, warnings = transform_workflow(prompt, execute_outputs)
            if warnings:
                print("[FUNCTIONAL] Warnings during transformation:")
                for warning in warnings:
                    print(f" - {warning}")
            return original_execute(self, prompt, prompt_id, extra_data, execute_outputs)
        PromptExecutor.execute = hooked_execute
    except Exception as e:
        print(f"[FUNCTIONAL] Failed to hook into ComfyUI execution: {e}")