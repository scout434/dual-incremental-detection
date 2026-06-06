"""DuET-aware Ultralytics trainer patch."""

from __future__ import annotations

from duet_repro.core.task_vectors import task_vector


class DuETDetectionTrainer:
    """Patch an Ultralytics trainer so DuET loss can refresh task vectors.

    Ultralytics creates the concrete trainer inside ``YOLO.train()``. This
    wrapper patches the trainer class once, then every optimizer step updates
    the criterion with the current task vector when the criterion supports it.
    """

    _patched: bool = False

    def __init__(
        self,
        base_trainer_cls,
        reference_state: dict,
        shared_key_exclude: tuple[str, ...],
    ):
        from ultralytics.utils.torch_utils import unwrap_model

        self._ref_state = reference_state
        self._shared_exclude = shared_key_exclude

        if DuETDetectionTrainer._patched:
            return

        original_optimizer_step = base_trainer_cls.optimizer_step

        def patched_optimizer_step(trainer) -> None:
            original_optimizer_step(trainer)
            criterion = getattr(unwrap_model(trainer.model), "criterion", None)
            if criterion is None or not hasattr(criterion, "set_curr_task_vector"):
                return

            curr_state = {
                key: value.detach().clone()
                for key, value in unwrap_model(trainer.model).state_dict().items()
            }
            curr_task_vector = task_vector(
                self._ref_state,
                curr_state,
                shared_key_exclude=self._shared_exclude,
            )
            criterion.set_curr_task_vector(curr_task_vector)

        base_trainer_cls.optimizer_step = patched_optimizer_step
        DuETDetectionTrainer._patched = True
