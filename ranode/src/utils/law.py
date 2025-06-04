import os
import subprocess
import numpy as np
import luigi
import law
import pandas as pd

from src.utils.utils import str_encode_value


class BaseTask(law.Task):
    """
    Base task which provides some convenience methods
    """

    version = law.Parameter()

    use_full_stats = luigi.BoolParameter(default=False)

    def store_parts(self):
        task_name = self.__class__.__name__
        return (
            os.getenv("OUTPUT_DIR"),
            f"version_{self.version}",
            task_name,
            f"use_full_stats_{self.use_full_stats}",
        )

    def local_path(self, *path):
        sp = self.store_parts()
        sp += path
        return os.path.join(*(str(p) for p in sp))

    def local_target(self, *path, **kwargs):
        return law.LocalFileTarget(self.local_path(*path), **kwargs)

    def local_directory_target(self, *path, **kwargs):
        return law.LocalDirectoryTarget(self.local_path(*path), **kwargs)


class ProcessMixin:

    mx = luigi.IntParameter(default=100)
    my = luigi.IntParameter(default=500)

    ensemble = luigi.IntParameter(default=1)

    def store_parts(self):
        return super().store_parts() + (
            f"mx_{self.mx}",
            f"my_{self.my}",
            f"ensemble_{self.ensemble}",
        )


class SignalStrengthMixin:

    # S/(S+B) ratio
    s_ratio_index = luigi.IntParameter(default=8)

    @property
    def s_ratio(self):
        conversion = {
            0: 0.0,
            1: 0.0001025915,
            2: 0.0001801174,
            3: 0.00031622776601683794,
            4: 0.0005551935914386209,
            5: 0.0009747402255566064,
            6: 0.001711328304161781,
            7: 0.0030045385302046933,
            8: 0.00527499706370262,
            9: 0.009261187281287938,
            10: 0.01625964693881482,
            11: 0.02854667663497933,
            12: 0.05011872336272722,
        }

        return conversion[self.s_ratio_index]

    def store_parts(self):
        round_s_ratio = np.round(self.s_ratio, 6)
        return super().store_parts() + (
            f"s_index_{self.s_ratio_index}_ratio_{str_encode_value(round_s_ratio)}",
        )


class TemplateRandomMixin:

    train_random_seed = luigi.IntParameter(default=233)

    def store_parts(self):
        return super().store_parts() + (f"train_seed_{self.train_random_seed}",)


class FoldSplitRandomMixin:

    fold_split_seed = luigi.IntParameter(default=0)

    def store_parts(self):
        return super().store_parts() + (f"fold_split_seed_{self.fold_split_seed}",)


class FoldSplitUncertaintyMixin:

    # controls how many times we split the data for uncertainty estimation
    fold_split_num = luigi.IntParameter(default=5)

    def store_parts(self):
        return super().store_parts() + (f"fold_split_num_{self.fold_split_num}",)


class BkgTemplateUncertaintyMixin:

    num_bkg_templates = luigi.IntParameter(default=20)

    def store_parts(self):
        return super().store_parts() + (f"num_templates_{self.num_bkg_templates}",)


class BkgModelMixin:

    use_perfect_bkg_model = luigi.BoolParameter(default=False)

    use_bkg_model_gen_data = luigi.BoolParameter(default=False)

    def store_parts(self):

        # use perfect bkg model and use bkg model to generate data cannot both be true
        assert not (
            self.use_perfect_bkg_model and self.use_bkg_model_gen_data
        ), "use_perfect_bkg_model and use_bkg_model_gen_data cannot both be true"

        return super().store_parts() + (
            f"use_perfect_bkg_model_{self.use_perfect_bkg_model}",
            f"use_bkg_model_gen_data_{self.use_bkg_model_gen_data}",
        )


class SigTemplateTrainingUncertaintyMixin:

    # controls the random seed for the training
    train_num_sig_templates = luigi.IntParameter(default=20)

    def store_parts(self):
        return super().store_parts() + (
            f"train_num_templates_{self.train_num_sig_templates}",
        )


class WScanMixin:

    w_min = luigi.FloatParameter(default=1e-5)
    w_max = luigi.FloatParameter(default=0.1)
    scan_number = luigi.IntParameter(default=20)

    def store_parts(self):
        return super().store_parts() + (
            f"w_min_{str_encode_value(self.w_min)}_w_max_{str_encode_value(self.w_max)}_scan_{self.scan_number}",
        )

    @property
    def w_range(self):
        w_range = np.logspace(
            np.log10(self.w_min), np.log10(self.w_max), self.scan_number
        )

        # round to 6 decimal places
        w_range = np.round(w_range, 6)
        return w_range
