"""Goals in this file:
 - We want to be able to run any configuration of intrinsic reward model we've built with only 1 function call
 - We want the results of that run to be saved to Wandb
 - We want a video of what went down, also
"""
from IntrinsicRewards.MohammadDQN import gridworld_utils as empowerment
from typing import Optional
import copy
import os
from acme.utils import loggers
import json
import IntrinsicRewards.Experiments.helpers as helpers
import wandb
from empax import evaluation
import IntrinsicRewards.Experiments.observers as observers
from acme import specs
from matplotlib import pyplot as plt
from ray import tune
import ray
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.integration.wandb import WandbLogger


"""We will use the experimenter to run any experiment function
It should simply take in "args" (a RayTune dictionary) and
an experiment function, and run all the experiments while doing saving as necessary"""
class RayExperimenter:
    def __init__(self,
                 name, #a string saying what name this experiment has
                 experiment_class,
                 args: dict, #arguments required for the experiment function, including "other_arguments"
                 wandb_project: str = "rl_skills",
                 wandb_username: str = "nrocketmann",
                 local_mode: bool = False
                 ):
        self.experiment_class = experiment_class
        self.name = name
        self.args = args
        self.args['name'] = name
        self.wandb_project = wandb_project
        self.wandb_username = wandb_username
        self.local_mode = local_mode

        self.args['wandb_username']= wandb_username
        self.args['wandb_project'] = wandb_project

        #set environment variables
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'

    #The public function that is called to run experiments
    def run_experiments(self):
        print("Running experiments!")
        if self.local_mode:
            ray.init(local_mode=True)
        analysis = tune.run(
            self.experiment_class,
            config=self.args,
        resources_per_trial={'gpu': 1})
        return


class Trainable(tune.Trainable):
    def logging_setup(self, config):


        os.chdir('/home/nameer/empowerment/IntrinsicRewards') #TODO: make this less dumb
        #self.args is what we are actually going to pass to the model
        self.name = config['name']
        self.args = copy.deepcopy(config)

        # logging
        logger = loggers.CSVLogger('logging/logdir/' + self.name + self.trial_id)
        self.logger = logger
        eval_logger = loggers.CSVLogger('logging/eval_logdir/' + self.name + self.trial_id)
        self.eval_logger = eval_logger

        # checkpointing
        checkpoint_path = 'logging/checkpoints/' + self.name + self.trial_id
        os.mkdir(checkpoint_path)
        self.args['checkpoint'] = True
        self.args['checkpoint_subpath'] = checkpoint_path

        # metadata
        writeable_args = {"modelname": self.name, "trial_id": self.trial_id}
        for k, v in self.args.items():
            if helpers.is_jsonable(k) and helpers.is_jsonable(v):
                writeable_args[k] = v

        json.dump(writeable_args, open('logging/metadata/' + self.name + self.trial_id + '.json', 'w'))

        # wandb logging
        self.tf_logdir = 'logging/tflogs/' + self.name + self.trial_id
        wandb.init(project=self.args['wandb_project'],entity=self.args['wandb_username'],config = writeable_args)
        wandb.run.name = self.name + self.trial_id

        #artifact logging
        self.artifacts_path = 'logging/artifacts/' + self.name + self.trial_id
        os.mkdir(self.artifacts_path)

        del self.args['name']
        del self.args['wandb_project']
        del self.args['wandb_username']


"""Required arguments in config:
        envname: str,
        model_type: str,
        sequence_length: int,
        num_steps: int,
        eval_every: int,
        num_eval_episodes: int,
"""
class EmpowermentTrainable(Trainable):
    def setup(self, config):
        self.logging_setup(config)

        model_type = config['model_type']
        envname = config['envname']
        sequence_length = config['sequence_length']
        beta = config['beta']
        self.num_steps = config['num_steps']
        self.eval_every = config['eval_every']
        self.num_eval_episodes = config['num_eval_episodes']

        model_function_dict = {
            'simple': empowerment.make_networks_simple,
        }

        #get model factory function
        if model_type in model_function_dict:
            network_function = model_function_dict[model_type]
        else:
            raise TypeError("No network function associated with " + model_type + ". Valid network functions: " +
                            str(list(model_function_dict.keys())))

        #make environment
        environment = empowerment.make_environment(envname)
        spec = specs.make_environment_spec(environment)

        #make networks
        Qnet, qnet, featnet, rnet, feat_dims = network_function(spec.actions)

        # make observer and logger for eval environment
        eval_observer_metric = observers.EmpowermentGraph(rnet, featnet, beta,
                                                          spec.actions.num_values,
                                                          self.artifacts_path, sequence_length)
        self.eval_observer = evaluation.observers.EvaluationObserver(
            episode_metrics=[
                eval_observer_metric,
            ],
            logger=self.eval_logger,
            artifacts_path=self.artifacts_path,
            do_render=True
        )

        del self.args['model_type']
        del self.args['envname']
        del self.args['sequence_length']
        del self.args['beta']
        del self.args['num_steps']
        del self.args['eval_every']
        del self.args['num_eval_episodes']


        self.environment_loop, self.eval_loop = empowerment.make_environment_loop(Qnet, qnet, featnet, rnet, feat_dims,
                                                                        environment, self.eval_observer, sequence_length,
                                                                        **self.args)
        self.num_iter = self.num_steps// self.eval_every
        self.iteration_counter = 0

    def step(self):
        # Train.
        self.environment_loop.run(num_steps=self.eval_every)

        # eval
        self.eval_observer.reset()
        global_step = int(self.environment_loop._counter.get_counts()['steps'])
        print(f"Evaluation: steps = {global_step}")

        eval_metrics = evaluation.evaluate.run_eval_step(
            self.eval_loop, global_step=global_step, num_episodes=self.num_eval_episodes)
        self.eval_observer.write_results(eval_metrics, steps=global_step)
        print("-" * 100)
        self.iteration_counter+=1

        if self.iteration_counter==self.num_iter:
            eval_metrics["done"] = True
        else:
            eval_metrics["done"] = False
        return eval_metrics

#Environments:
#1: room with one object you can interact with
#2: multiple objects, can place object on/in other one
#3: lots of composition with many objects that can interact

#on this: FARM/vanilla + DADS, DQN + Mohammad