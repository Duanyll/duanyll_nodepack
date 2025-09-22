from .transform import transform_workflow

def hook_comfyui_execution():
    from execution import PromptExecutor
    original_execute = PromptExecutor.execute
    def hooked_execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        prompt = transform_workflow(prompt)
        return original_execute(self, prompt, prompt_id, extra_data, execute_outputs)
    PromptExecutor.execute = hooked_execute