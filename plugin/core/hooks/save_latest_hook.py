import os
import shutil
from typing import Dict

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SaveLatestCheckpointHook(Hook):
    """Ensure the most recent checkpoint is duplicated to ``latest.pth``.

    MMEngine already tracks the latest checkpoint path via the message hub, but
    some downstream tooling still expects an explicit ``latest.pth`` file in the
    work directory. This hook mirrors the most recently saved checkpoint after
    every checkpointing event.
    """

    priority = 'VERY_LOW'

    def after_save_checkpoint(self, runner, checkpoint: Dict) -> None:  # type: ignore[override]
        """Copy the newest checkpoint to ``latest.pth``."""
        last_ckpt = runner.message_hub.get_info('last_ckpt', None)
        if last_ckpt is None:
            return
        if runner.world_size > 1 and runner.rank != 0:
            return
        if not isinstance(last_ckpt, str) or not os.path.isfile(last_ckpt):
            # Distributed runtimes such as DeepSpeed may produce directory
            # checkpoints; skip mirroring in that scenario.
            return

        latest_path = os.path.join(runner.work_dir, 'latest.pth')
        try:
            shutil.copy2(last_ckpt, latest_path)
        except OSError as exc:  # pragma: no cover - filesystem issues
            runner.logger.warning(
                f'Failed to update latest checkpoint at {latest_path}: {exc}')
        else:
            runner.logger.debug(f'Updated latest checkpoint: {latest_path}')

    def after_train(self, runner) -> None:  # type: ignore[override]
        """Ensure ``latest.pth`` exists when training stops early."""
        last_ckpt = runner.message_hub.get_info('last_ckpt', None)
        if last_ckpt is None:
            last_file = os.path.join(runner.work_dir, 'last_checkpoint')
            if os.path.isfile(last_file):
                try:
                    with open(last_file, 'r') as f:
                        path = f.read().strip()
                except OSError:
                    path = None
                if path:
                    runner.message_hub.update_info('last_ckpt', path)
                    last_ckpt = path

        if isinstance(last_ckpt, str) and os.path.isfile(last_ckpt):
            latest_path = os.path.join(runner.work_dir, 'latest.pth')
            try:
                shutil.copy2(last_ckpt, latest_path)
            except OSError as exc:  # pragma: no cover
                    runner.logger.warning(
                        f'Failed to update latest checkpoint at {latest_path}: {exc}')
