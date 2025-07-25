from typing import List, Any

class Pipeline:
    """Simple pipeline executor that runs steps sequentially."""

    def __init__(self, steps: List[Any]):
        self.steps = steps

    def execute(self):
        data = None
        for step in self.steps:
            if hasattr(step, 'run'):
                data = step.run(data)
            else:
                raise AttributeError(f'Step {step} has no run() method')
        return data
