import os
from jinja2 import Environment, FileSystemLoader

class PromptManager:
    def __init__(self, prompt_dir="app/prompts/scripts"):
        self.prompt_dir = prompt_dir
        self.env = Environment(loader=FileSystemLoader(self.prompt_dir), autoescape=False)

    def load(self, relative_path: str, **kwargs) -> str:
        template = self.env.get_template(relative_path)
        return template.render(**kwargs)
    
prompt_manager = PromptManager()