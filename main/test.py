import logging
import traceback
import subprocess
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from factory.trainer_factory import TrainerFactory





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
        test_dataset_config = config.dataset_config
        test_dataset_config['config_file']['expanded'].paras.shuffle = False
        dataset = DatasetFactory.get_dataset(test_dataset_config)

        model_config = config.model_config
        model_config.config_file.expanded.update({'mode': "testing"})
        model = ModelFactory.get_model(model_config)
        trainer = TrainerFactory.get_trainer(config.testing_config)
        if config.extra_config['distributed']:
            logging.info("using distributed training......")
            trainer.config_distributed_computing(launcher=config.extra_config['launcher'],
                                                 tcp_port=config.extra_config['tcp_port'],
                                                 local_rank=config.extra_config['local_rank'])

        trainer.set_model(model)
        trainer.set_test_dataset(dataset)

        # trainer.set_logger(logger)
        
        # load checkpoint
        if args.check_point_file is not None:
            trainer.load_state(args.check_point_file)
        elif "ckpt" in config.testing_config:
            trainer.load_state(config.testing_config.ckpt)
        else:
            logging.warning("No Checkpoint provided for this test!")
            
        logging.info("Preparation done! Trainer run!")
        trainer.run_test()

    except Exception as e:
        logging.exception(traceback.format_exc())
        exit(-1)
