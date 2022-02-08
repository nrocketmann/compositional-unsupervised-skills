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


"""Ok, time to make a plan:
Experimenter class governs a large set of experiments

"""

#helper function to set up logging for any ray tune boi
#input config is a raytune config object
def setup_logging_config(config):
    print(config)
    return {}

    # trial_id = config['trial_id'] #TODO: actually get the trial id lol
    #
    # # logging
    # logger = loggers.CSVLogger('logdir/' + config['name'] + trial_id)
    #
    # # checkpointing
    # args['checkpoint'] = True
    # checkpoint_path = 'checkpoints/' + self.name + str(modelnum)
    # os.mkdir(checkpoint_path)
    # args['checkpoint_subpath'] = checkpoint_path
    #
    # # metadata
    # writeable_args = {"modelname": self.name, "modelnum": modelnum}
    # for k, v in args.items():
    #     if helpers.is_jsonable(k) and helpers.is_jsonable(v):
    #         writeable_args[k] = v
    #
    # json.dump(writeable_args, open('metadata/' + self.name + str(modelnum) + '.json', 'w'))
    #
    # # wandb logging
    # tf_logdir = 'tflogs/' + self.name + str(modelnum)
    # run = wandb.init(project=self.wandb_project, entity=self.wandb_username, reinit=True, config=writeable_args)
    # wandb.run.name = self.name + str(modelnum)
    # args['tflogs'] = tf_logdir
    # args['modelname'] = self.name
    # args['modelnum'] = modelnum

"""We will use the experimenter to run any experiment function
It should simply take in "args" (a RayTune dictionary) and
an experiment function, and run all the experiments while doing saving as necessary"""
class RayExperimenter:
    def __init__(self,
                 name, #a string saying what name this experiment has
                 experiment_function,
                 args: dict, #arguments required for the experiment function, including "other_arguments"
                 wandb_project: str = "rl_skills",
                 wandb_username: str = "nrocketmann"
                 ):
        self.experiment_function = experiment_function
        self.name = name
        self.args = args
        self.args['name'] = name
        self.wandb_project = wandb_project
        self.wandb_username = wandb_username

    #The public function that is called to run experiments
    def run_experiments(self):
        print("Running experiments!")
        #ray.init(local=True)
        analysis = tune.run(
            self.experiment_function,
            config=self.args)


class Trainable(tune.Trainable):
    def setup(self, config):
        args = {}
        # logging
        logger = loggers.CSVLogger('logdir/' + config['name'] + self.trial_id)
        self.logger = logger
        # checkpointing
        args['checkpoint'] = True
        checkpoint_path = 'checkpoints/' + self.name + self.trial_id
        os.mkdir(checkpoint_path)
        args['checkpoint_subpath'] = checkpoint_path

        # metadata
        writeable_args = {"modelname": self.name, "modelnum": modelnum}
        for k, v in args.items():
            if helpers.is_jsonable(k) and helpers.is_jsonable(v):
                writeable_args[k] = v

        json.dump(writeable_args, open('metadata/' + self.name + str(modelnum) + '.json', 'w'))

        # wandb logging
        tf_logdir = 'tflogs/' + self.name + str(modelnum)
        run = wandb.init(project=self.wandb_project, entity=self.wandb_username, reinit=True, config=writeable_args)
        wandb.run.name = self.name + str(modelnum)
        args['tflogs'] = tf_logdir
        args['modelname'] = self.name
        args['modelnum'] = modelnum

    def step(self):  # This is called iteratively.
        score = objective(self.x, self.a, self.b)
        self.x += 1
        return {"score": score}


"""Required arguments in config:
        envname: str,
        model_type: str,
        sequence_length: int,
        num_steps: int,
        eval_every: int,
        num_eval_episodes: int,
"""
def ray_gridworld_empowerment_experiment(
        config: dict
):
    config = setup_logging_config(config)
    modelname = config['name']
    trial_id = config['trial_id']
    envname = config['envname']
    model_type = config['model_type']
    sequence_length = config['sequence_length']
    num_steps = config['num_steps']
    eval_every = config['eval_every']
    num_eval_episodes = config['num_eval_episodes']
    return

    # First get the model function
    model_function_dict = {
        'simple': empowerment.make_networks_simple,
    }
    if model_type in model_function_dict:
        network_function = model_function_dict[model_type]
    else:
        raise TypeError("No network function associated with " + model_type + ". Valid network functions: " +
                        str(list(model_function_dict.keys())))

    artifacts_path = 'logging/artifacts/' + modelname + trial_id
    os.mkdir(artifacts_path)
    #make environment
    environment = empowerment.make_environment(envname)
    spec = specs.make_environment_spec(environment)

    #save plot of environment
    plt.figure()
    plt.imshow(environment.render())
    plt.savefig(artifacts_path + '/environment.png')
    plt.show()

    #make networks
    Qnet, qnet, featnet, rnet, feat_dims = network_function(spec.actions)

    #make observer and logger for eval environment
    eval_logger = loggers.CSVLogger('eval_logdir/' + modelname + trial_id)
    eval_observer_metric = observers.EmpowermentGraph(rnet, featnet,other_arguments['beta'],spec.actions.num_values,
                                                      artifacts_path, sequence_length)
    eval_observer = evaluation.observers.EvaluationObserver(
        episode_metrics=[
            eval_observer_metric,
        ],
        logger=eval_logger,
        artifacts_path=artifacts_path,
    )

    #make environment loops
    environment_loop, eval_loop = empowerment.make_environment_loop(Qnet, qnet, featnet, rnet, feat_dims,environment, eval_observer, sequence_length, **other_arguments)

    for i in range(num_steps//eval_every):
        # Train.
        environment_loop.run(num_steps=eval_every)

        # eval
        eval_observer.reset()
        global_step = int(environment_loop._counter.get_counts()['steps'])
        print(f"Evaluation: steps = {global_step}")

        eval_metrics = evaluation.evaluate.run_eval_step(
            eval_loop, global_step=global_step, num_episodes=num_eval_episodes)
        eval_observer.write_results(eval_metrics, steps=global_step)
        print("-" * 100)


    return
