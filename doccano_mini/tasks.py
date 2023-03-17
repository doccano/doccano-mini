from enum import Enum


class TaskType(Enum):
    TEXT_CLASSIFICATION = "Text Classification"
    TASK_FREE = "Task Free"


options = [task_type.value for task_type in TaskType]
