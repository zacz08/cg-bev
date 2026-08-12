"""Step + epoch metric logger callback for CG-BEV training.

Writes two CSVs into ``log_dir``:
    - ``step_log.csv``  -> one row per ``log_step_freq`` global steps,
                            consuming ``pl_module._step_loss_dict`` populated
                            in ``training_step``.
    - ``epoch_log.csv`` -> one row per training epoch, snapshotted from
                            ``trainer.callback_metrics`` (averaged ``*_epoch``
                            train metrics + per-epoch ``val/*`` metrics).

After every epoch a combined ``loss_plot.png`` is regenerated, plotting the
step-level curves (thin translucent) and the epoch-level curves (bold,
solid=train / dashed=val) on a shared step axis.
"""

import csv
import logging
import os

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #

def parse_csv_and_plot(step_csv, epoch_csv, output_path):
    """Combined step + epoch learning-curve plot.

    - Loss components on the top axis (one color per metric base name; train
      solid, val dashed; thin translucent step curves underneath the bold
      epoch curves).
    - ``val/IoU`` (and any other ``IoU`` / ``mIoU`` style metric) on a
      separate bottom axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    def _read_csv(path, required_cols, min_rows):
        if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        try:
            df = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.warning(f"Could not parse {path}: {exc}")
            return None
        for c in required_cols:
            if c not in df.columns:
                return None
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=required_cols).copy()
        if len(df) < min_rows:
            return None
        return df

    step_df = _read_csv(step_csv, required_cols=['step', 'epoch'], min_rows=2)
    epoch_df = _read_csv(epoch_csv, required_cols=['epoch'], min_rows=1)

    if step_df is None and epoch_df is None:
        return

    metric_cols = {'val/IoU', 'val/mIoU'}

    def _base_name(col):
        if col.startswith('train/'):
            return col[len('train/'):]
        if col.startswith('val/'):
            return col[len('val/'):]
        return col

    # Discover loss-component base names. Step CSV columns are unprefixed.
    base_names = set()
    if step_df is not None:
        for c in step_df.columns:
            if c in ('step', 'epoch'):
                continue
            base_names.add(c)
    if epoch_df is not None:
        for c in epoch_df.columns:
            if c == 'epoch' or c in metric_cols:
                continue
            if c.startswith('train/') or c.startswith('val/'):
                base = _base_name(c)
                # Strip Lightning's "_epoch" suffix on epoch-aggregated metrics
                # so they collapse onto the same color as the step-level metric.
                if base.endswith('_epoch'):
                    base = base[:-len('_epoch')]
                base_names.add(base)

    has_metrics = epoch_df is not None and any(c in epoch_df.columns for c in metric_cols)

    n_plots = 1 + int(has_metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=(11, 5 * n_plots))
    if n_plots == 1:
        axes = [axes]
    ax_loss = axes[0]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sorted_bases = sorted(base_names)
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(sorted_bases)}

    # --- Step-level (thin translucent solid) ---
    if step_df is not None:
        x_step = step_df['step'].values
        for base in sorted_bases:
            if base in step_df.columns:
                vals = pd.to_numeric(step_df[base], errors='coerce')
                valid = vals.notna()
                if valid.any():
                    ax_loss.plot(x_step[valid], vals[valid].values,
                                 color=color_map[base], alpha=0.25,
                                 linewidth=0.8, linestyle='-')

    # Map epoch index to a step-equivalent x for a shared axis.
    if epoch_df is not None and step_df is not None and len(step_df):
        max_step = float(step_df['step'].max())
        max_epoch = float(step_df['epoch'].max())
        steps_per_epoch = max_step / max(max_epoch, 1) if max_epoch > 0 else max_step
        x_epoch = (epoch_df['epoch'].values + 1) * steps_per_epoch
        x_label = 'Step'
    elif epoch_df is not None:
        x_epoch = epoch_df['epoch'].values
        x_label = 'Epoch'
    else:
        x_epoch = None
        x_label = 'Step'

    # --- Epoch-level (bold) ---
    if epoch_df is not None and x_epoch is not None:
        for base in sorted_bases:
            color = color_map[base]
            # Match either bare or _epoch-suffixed train columns.
            for tcol in (f'train/{base}_epoch', f'train/{base}'):
                if tcol in epoch_df.columns:
                    vals = pd.to_numeric(epoch_df[tcol], errors='coerce')
                    valid = vals.notna()
                    if valid.any():
                        ax_loss.plot(x_epoch[valid], vals[valid].values,
                                     color=color, linewidth=2.0, linestyle='-',
                                     marker='o', markersize=4, label=f'train/{base}')
                    break
            vcol = f'val/{base}'
            if vcol in epoch_df.columns:
                vals = pd.to_numeric(epoch_df[vcol], errors='coerce')
                valid = vals.notna()
                if valid.any():
                    ax_loss.plot(x_epoch[valid], vals[valid].values,
                                 color=color, linewidth=2.0, linestyle='--',
                                 marker='x', markersize=6, label=f'val/{base}')

    ax_loss.set_xlabel(x_label)
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss Components (solid=train, dashed=val; thin=step, bold=epoch)')
    ax_loss.legend(fontsize=7, loc='upper right', ncol=2)
    ax_loss.grid(True, alpha=0.3)

    if has_metrics:
        ax_metric = axes[1]
        for col in sorted(metric_cols):
            if col in epoch_df.columns:
                vals = pd.to_numeric(epoch_df[col], errors='coerce')
                valid = vals.notna()
                if valid.any():
                    ax_metric.plot(epoch_df['epoch'].values[valid], vals[valid].values,
                                   marker='o', markersize=5, linewidth=2.0, label=col)
        ax_metric.set_xlabel('Epoch')
        ax_metric.set_ylabel('IoU')
        ax_metric.set_title('Validation Metrics')
        ax_metric.legend(fontsize=8)
        ax_metric.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Callback                                                                    #
# --------------------------------------------------------------------------- #

class MetricLogger(Callback):
    """Per-step + per-epoch CSV logger with combined plot.

    The Lightning module's ``training_step`` is expected to assign a dict to
    ``pl_module._step_loss_dict`` containing the loss components (already
    converted to python floats), plus the special keys ``_global_step`` and
    ``_epoch``. This callback drains that attribute in
    ``on_train_batch_end`` and writes a row every ``log_step_freq`` steps.
    Per-epoch metrics are pulled from ``trainer.callback_metrics``.
    """

    def __init__(self, log_dir, log_step_freq=50):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_step_freq = max(int(log_step_freq), 1)

        self._step_csv = os.path.join(log_dir, "step_log.csv")
        self._epoch_csv = os.path.join(log_dir, "epoch_log.csv")
        self._plot_path = os.path.join(log_dir, "loss_plot.png")
        self._step_fields = self._read_csv_header(self._step_csv)
        self._epoch_fields = self._read_csv_header(self._epoch_csv)
        # Cached this-epoch IoU computed from seg_metric BEFORE the module
        # resets it in its own on_validation_epoch_end. See note in
        # on_validation_epoch_end below for the PL 2.x timing rationale.
        self._cached_val_iou = None

    # --------------- CSV plumbing ---------------
    @staticmethod
    def _read_csv_header(csv_path):
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return None
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for fields in reader:
                if any(field.strip() for field in fields):
                    return fields
        return None

    @staticmethod
    def _csv_needs_header(csv_path):
        return not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    @staticmethod
    def _format(value, precision):
        if value is None:
            return ""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.detach().cpu().item()
            else:
                return str(value.detach().cpu().tolist())
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if not np.isfinite(value):
                return ""
            return f"{value:.{precision}f}"
        return value

    def _rewrite_with_fields(self, csv_path, fieldnames):
        if self._csv_needs_header(csv_path):
            return
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    def _ensure_fields(self, csv_path, current, row):
        if current is None:
            current = self._read_csv_header(csv_path)
        if current is None:
            current = list(row.keys())
        new_keys = [k for k in row if k not in current]
        if new_keys:
            current = current + new_keys
            self._rewrite_with_fields(csv_path, current)
        return current

    def _append(self, csv_path, fieldnames, row, precision):
        write_header = self._csv_needs_header(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow({k: self._format(row.get(k), precision) for k in fieldnames})

    # --------------- Lightning hooks ---------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step_data = getattr(pl_module, '_step_loss_dict', None)
        if step_data is None:
            return
        pl_module._step_loss_dict = None

        gstep = int(step_data.pop('_global_step', pl_module.global_step))
        ep = int(step_data.pop('_epoch', pl_module.current_epoch))

        # Throttle by log_step_freq (always log step 0 so the curve has a head).
        if gstep != 0 and (gstep % self.log_step_freq) != 0:
            return

        row = {'step': gstep, 'epoch': ep}
        row.update(step_data)
        self._step_fields = self._ensure_fields(self._step_csv, self._step_fields, row)
        self._append(self._step_csv, self._step_fields, row, precision=6)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Snapshot val/IoU directly from the seg_metric accumulator.

        PL 2.x calls callback hooks *before* LightningModule hooks for
        ``on_validation_epoch_end``. The module then computes ``val/IoU``
        from ``first_stage_model.seg_metric`` AND resets it. If we wait
        until ``on_train_epoch_end`` to read ``trainer.callback_metrics``,
        the train/*_epoch values are present but ``val/IoU`` from THIS
        epoch is too — provided we cache it here, before the module's hook
        resets the metric. (Mirrors GSFormer's ``MetricLogger`` workaround
        of reading ``miou_metric`` directly.)
        """
        if trainer.sanity_checking:
            self._cached_val_iou = None
            return

        seg_metric = getattr(getattr(pl_module, 'first_stage_model', None),
                             'seg_metric', None)
        iou_value = None
        if seg_metric is not None:
            try:
                score = seg_metric.compute()
                if hasattr(score, 'numel') and score.numel() > 0:
                    iou_value = float(score.mean().item())
            except Exception:
                iou_value = None
        self._cached_val_iou = iou_value

    def on_train_epoch_end(self, trainer, pl_module):
        """Write one row to epoch_log.csv per training epoch.

        Runs AFTER the validation loop (PL 2.x fit-loop ordering), so both
        ``train/*_epoch`` and the val metrics aggregated in
        ``trainer.callback_metrics`` are already populated. We further
        normalize:
          - drop the ``_step`` and bare unsuffixed train metrics (which are
            just the last batch's instantaneous values, not epoch averages);
          - rename ``train/foo_epoch`` -> ``train/foo`` so train and val
            share the same base name and the plot can color-pair them.
        """
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        epoch = int(pl_module.current_epoch)
        row = {"epoch": epoch}

        # Identify train base names that have an ``_epoch`` aggregate so we
        # can suppress the corresponding bare / _step duplicates.
        epoch_bases = {
            k[len('train/'):-len('_epoch')]
            for k in metrics
            if k.startswith('train/') and k.endswith('_epoch')
        }

        for key, val in metrics.items():
            if not key.startswith(('train/', 'val/')):
                continue
            out_key = key
            if key.startswith('train/'):
                base = key[len('train/'):]
                if base.endswith('_epoch'):
                    out_key = f'train/{base[:-len("_epoch")]}'
                elif base.endswith('_step'):
                    continue  # drop step-level dup
                elif base in epoch_bases:
                    continue  # drop bare-key dup of <base>_epoch
            if hasattr(val, 'item'):
                try:
                    val = val.item()
                except Exception:
                    continue
            row[out_key] = val

        # Inject the val/IoU cached BEFORE the module reset seg_metric.
        if self._cached_val_iou is not None:
            row['val/IoU'] = self._cached_val_iou
        self._cached_val_iou = None

        self._epoch_fields = self._ensure_fields(self._epoch_csv, self._epoch_fields, row)
        self._append(self._epoch_csv, self._epoch_fields, row, precision=4)

        try:
            parse_csv_and_plot(self._step_csv, self._epoch_csv, self._plot_path)
        except Exception as exc:
            logger.warning(f"loss_plot.png regen failed: {exc}")
