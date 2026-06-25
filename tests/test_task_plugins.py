from flowyforge.task_plugins import (
    CaloGenerationTask,
    ClassificationTask,
    MaskedModelingTask,
    TriggerTask,
)


def test_task_plugins_instantiate() -> None:
    tasks = [
        ClassificationTask(),
        TriggerTask(),
        MaskedModelingTask(),
        CaloGenerationTask(),
    ]

    assert [task.name for task in tasks] == [
        "classification",
        "trigger",
        "masked_modeling",
        "calo_generation",
    ]
    assert all(task.compute_metrics(outputs={}, batch={}) == {} for task in tasks)

