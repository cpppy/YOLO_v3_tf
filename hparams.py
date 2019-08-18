import json
import config


class HParams:

    def __init__(self):
        self.learning_rate = 1e-3
        self.decay_rate = 0.95
        self.decay_steps = 1000000
        self.train_steps = 3000000
        self.eval_steps = 100000000
        self.batch_size = 1
        self.gpu_nums = 1
        self.log_freq = 1
        self.summary_freq = 1  # self.log_freq
        self.save_checkpoints_steps = 10
        self.keep_checkpoint_max = 3
        self.dropout_rate = 0.5


    def reload_params(self, config_fpath):
        if config.dev_mode:
            print('-------dev_mode, use default parameters')
            return
        try:
            with open(config_fpath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                print('config_dict: ', config_dict)
                self.learning_rate = float(config_dict['learning_rate'])
                self.decay_rate = float(config_dict['decay_rate'])
                self.decay_steps = int(config_dict['decay_steps'])
                self.train_steps = int(config_dict['train_steps'])
                self.eval_steps = int(config_dict['eval_steps'])
                self.batch_size = int(config_dict['batch_size'])
                self.gpu_nums = int(config_dict['gpu_nums'])
                self.log_freq = int(config_dict['log_freq'])
                self.save_checkpoints_steps = int(config_dict['save_checkpoints_steps'])
                self.keep_checkpoint_max = int(config_dict['keep_checkpoint_max'])
                print('load configuration success')
        except Exception as e:
            print('fail to reload config file', e)
            exit(1)


def main():
    _hparams = HParams()
    config_fpath = './train_config.json'
    _hparams.reload_params(config_fpath)


if __name__ == '__main__':
    config.dev_mode = False
    main()
