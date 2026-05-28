from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SyncIterInfoHook(Hook):
    """Keep model attributes in sync with runner iteration/epoch counters."""

    priority = 'VERY_HIGH'

    def before_train_iter(self, runner, batch_idx, data_batch=None) -> None:  # type: ignore[override]
        model = runner.model
        module = getattr(model, 'module', model)
        module.num_iter = runner.iter
        module.num_epoch = runner.epoch
        if runner.message_hub is not None:
            runner.message_hub.update_info('iter', runner.iter)
