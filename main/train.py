import os
import logging
import traceback
import subprocess
import copy
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from factory.trainer_factory import TrainerFactory
from utils.logger.basic_logger import BasicLogger
from utils.logger.dummy_logger import DummyLogger





if __name__ == '__main__':
    git_version = subprocess.check_output(["git", 'rev-parse', 'HEAD']).strip().decode()
    logging.info(f'Your program version is {git_version}')
    try:
        # manage config
        logging_logger = logging.getLogger()
        logging_logger.setLevel(logging.NOTSET)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        ch.setFormatter(formatter)

        config = Configuration()
        args = config.get_shell_args_train()
        config.load_config(args.cfg_dir)
        config.overwrite_config_by_shell_args(args)

        # instantiating all modules by non-singleton factory
        dataset = DatasetFactory.get_dataset(config.dataset_config)
        
        
        if config.training_config.enable_val:
            val_dataset_config = copy.deepcopy(config.dataset_config)
            
            val_dataset_config['config_file']['expanded'].paras.for_train = False
            val_dataset_config['config_file']['expanded'].paras.shuffle = False
            val_dataset_config['config_file']['expanded'].paras.mode = 'val'
            val_dataset = DatasetFactory.get_dataset(val_dataset_config)
        else:
            val_dataset = None

        
        trainer = TrainerFactory.get_trainer(config.training_config)

        if config.extra_config['distributed']:
            logging.info("using distributed training......")
            trainer.config_distributed_computing(launcher=config.extra_config['launcher'],
                                                 tcp_port=config.extra_config['tcp_port'],
                                                 local_rank=config.extra_config['local_rank'])
        model = ModelFactory.get_model(config.model_config)
            
        logger = None
        if args.log_dir is not None :
            config._logging_config['log_dir'] = args.log_dir
        
        if (not config.extra_config['distributed']) or (os.environ['RANK'] == str(0)):
            logger = BasicLogger.get_logger(config)
            logger.log_config(config)
        else:
            logger = DummyLogger.get_logger(config)
        trainer.set_model(model)
        trainer.set_dataset(dataset)
        if config.training_config.enable_val:
            trainer.set_val_dataset(val_dataset)
        trainer.set_logger(logger)
        if args.check_point_file is not None:
            trainer.load_state(args.check_point_file)
        elif "ckpt" in config.training_config:
            trainer.load_state(config.training_config.ckpt)
        logging.info("Preparation done! Trainer run!")
        trainer.run()
        if args.screen_log is not None:
            logger.log_model_params(model, force=True)
            logger.copy_screen_log(args.screen_log)
    except Exception as e:
        if logger is not None:
            logger.log_model_params(model, force=True)
        logging.exception(traceback.format_exc())
        exit(-1)
