import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt


class nnUNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False, 
                 aux_metrics:list=None):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            # 'train_losses': list(),
            # 'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        if aux_metrics is not None:
            for i in aux_metrics:
                if i not in self.my_fantastic_logging:
                    self.my_fantastic_logging[i] = list()
        self.aux_metrics = aux_metrics
        self.verbose = verbose
        self.unique_auc_metrics = [i for i in self.aux_metrics if 'val' not in i]
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)

    def plot_progress_png(self, output_folder, image_name=None):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        plot_num = 3 + len(self.unique_auc_metrics)
        fig, ax_all = plt.subplots(plot_num, 1,  figsize=(30, int(plot_num * 18)))

        x_values = list(range(epoch + 1))
        current_plot_num = 0
        # plot aux metrics first'
        for i in range(len(self.unique_auc_metrics)):
            ax = ax_all[current_plot_num]
            aux_name = self.unique_auc_metrics[i]
            ax.plot(x_values, self.my_fantastic_logging[aux_name][:epoch + 1], color='b', ls='-', label=aux_name, linewidth=4)
            if f"{aux_name}_val" in self.my_fantastic_logging:
                ax.plot(x_values, self.my_fantastic_logging[f"{aux_name}_val"][:epoch + 1], color='r', ls='-', label=f"{aux_name}_val", linewidth=4)
            ax.set_xlabel("epoch")
            ax.set_ylabel("value")
            ax.legend(loc=(0, 1))
            current_plot_num += 1

        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[current_plot_num]
        ax.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice",
                 linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
                 linewidth=4)
        ax.set_xlabel("epoch")
        # ax.set_ylabel("loss")
        ax.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        # ax2.legend(loc=(0.2, 1))
        current_plot_num += 1

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[current_plot_num]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))
        current_plot_num += 1

        # learning rate
        ax = ax_all[current_plot_num]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))
        current_plot_num += 1

        # # auxiliary metric plot;
        # if self.aux_metrics is not None:
        #     ax = ax_all[3]
        #     colors = ['r', 'g', 'b', 'y', 'm', 'c']
        #     linestyles = ['-', '--', '-.', '-*']
        #     assert len(self.aux_metrics) < len(colors) and len(self.aux_metrics) < len(linestyles), \
        #     "The auxiliary metric has more item than  color or type, may lead to duplicate line type."
        #     for i,metric_name in enumerate(self.aux_metrics):
        #         ax.plot(x_values, self.my_fantastic_logging[metric_name][:epoch + 1], 
        #                 color=colors[i], ls=linestyles[i], label=metric_name, linewidth=4)
        #     ax.legend(loc=(0, 1))


        plt.tight_layout()

        fig.savefig(join(output_folder, image_name or "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint





class nnUNetLogger_V1(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False, 
                 aux_metrics:list=None,
                 aux_metrics_1:list=None):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        if aux_metrics is not None:
            for i in aux_metrics:
                if i not in self.my_fantastic_logging:
                    self.my_fantastic_logging[i] = list()
        self.aux_metrics = aux_metrics
        if aux_metrics_1 is not None:
            for i in aux_metrics_1:
                if i not in self.my_fantastic_logging:
                    self.my_fantastic_logging[i] = list()
        self.aux_metrics_1 = aux_metrics_1
        self.verbose = verbose
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)

    def plot_progress_png(self, output_folder, image_name=None):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        plot_num = 3 if self.aux_metrics is None else 4
        if self.aux_metrics_1 is not None:
            plot_num += 1
        fig, ax_all = plt.subplots(plot_num, 1,  figsize=(30, int(plot_num * 18)))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
                 linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        # auxiliary metric plot;
        if self.aux_metrics is not None:
            ax = ax_all[3]
            colors = ['r', 'g', 'b']
            linestyles = ['-', '--', '-.', '-*']
            assert len(self.aux_metrics) < len(colors) and len(self.aux_metrics) < len(linestyles), \
            "The auxiliary metric has more item than  color or type, may lead to duplicate line type."
            for i,metric_name in enumerate(self.aux_metrics):
                ax.plot(x_values, self.my_fantastic_logging[metric_name][:epoch + 1], 
                        color=colors[i], ls=linestyles[i], label=metric_name, linewidth=4)
            ax.legend(loc=(0, 1))

        # auxiliary metric plot;
        if self.aux_metrics_1 is not None:
            ax = ax_all[4]
            colors = ['r', 'g', 'b']
            linestyles = ['-', '--', '-.', '-*']
            assert len(self.aux_metrics_1) < len(colors) and len(self.aux_metrics_1) < len(linestyles), \
            "The auxiliary metric has more item than  color or type, may lead to duplicate line type."
            for i,metric_name in enumerate(self.aux_metrics_1):
                ax.plot(x_values, self.my_fantastic_logging[metric_name][:epoch + 1], 
                        color=colors[i], ls=linestyles[i], label=metric_name, linewidth=4)
            ax.legend(loc=(0, 1))


        plt.tight_layout()

        fig.savefig(join(output_folder, image_name or "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint