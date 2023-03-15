from enum import Enum


class TaskType(Enum):
    TEXT_CLASSIFICATION = "Text Classification"


options = [task_type.value for task_type in TaskType]
