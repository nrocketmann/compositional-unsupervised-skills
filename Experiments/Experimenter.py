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


#TODO: Add in observers


"""We will use the experimenter to run any experiment function
It should simply take in "args", an experiment function, and run all the experiments while doing saving as necessary"""
class Experimenter:
    def __init__(self,
                 name, #a string saying what name this experiment has
                 experiment_function,
                 args: dict, #arguments required for the experiment function, including "other_arguments"
                 wandb_project: str = "rl_skills",
                 wandb_username: str = "nrocketmann"
                 ):
        self.experiment_function = experiment_function
        self.name = name

        #things we always want to have:
        args['checkpoint'] = True

        self.args = args
        self.wandb_project = wandb_project
        self.wandb_username = wandb_username

    #This function actually runs all the experiments in "args"
    #It does so recursively over all Arglist's
    #ONLY TO BE USED INTERNALLY
    def _run_experiments(self, args: dict):
        leaf_node = True #Bool to tell us if we can safely run this arg set
        for k,v in args.items():
            if isinstance(v, Arglist): #recurse through the arglist for parameter search
                leaf_node = False
                for arg in v:
                    argscopy = copy.copy(args)
                    argscopy[k] = arg
                    self._run_experiments(argscopy)
        # now after doing all that recursing... we can actually run stuff
        if leaf_node:
            print("Running experiment with args: " + str(args))

            #saving logging, checkpointing, and metadata
            modelnum = self.get_modelnum()

            #logging
            args['logger'] = loggers.CSVLogger('logdir/' + self.name + str(modelnum))

            #checkpointing
            args['checkpoint'] = True
            checkpoint_path = 'checkpoints/' + self.name + str(modelnum)
            os.mkdir(checkpoint_path)
            args['checkpoint_subpath'] = checkpoint_path

            #metadata
            writeable_args = {"name":self.name, "modelnum":modelnum}
            for k,v in args.items():
                if helpers.is_jsonable(k) and helpers.is_jsonable(v):
                    writeable_args[k] = v

            json.dump(writeable_args, open('metadata/' + self.name + str(modelnum) + '.json','w'))

            #wandb logging
            tf_logdir = 'tflogs/' + self.name + str(modelnum)
            run = wandb.init(project=self.wandb_project,entity=self.wandb_username, reinit=True)
            wandb.run.name = self.name + str(modelnum)
            wandb.config = writeable_args
            args['tflogs'] = tf_logdir
            self.experiment_function(**args)
            run.finish()


    #The public function that is called to run experiments
    def run_experiments(self):
        print("Running experiments!")
        self._run_experiments(self.args)

    #Each model with the same name is assigned a number one greater than the last one
    #Logs are saved as [MODELNAME][NUMBER].csv in logdir
    def get_modelnum(self):
        existing_logs = list(filter(lambda x: x.startswith(self.name), os.listdir("logdir/")))
        return len(existing_logs)


"""Wrapper around a list for recursive parameter searches"""
class Arglist(list):
    pass


def gridworld_empowerment_experiment(
        envname: str,
        model_type: str,
        num_episodes: Optional[int] = None,
        num_steps: Optional[int] = None,
        **other_arguments
):

    # First get the model function
    model_function_dict = {
        'simple': empowerment.make_networks_simple,
    }
    if model_type in model_function_dict:
        network_function = model_function_dict[model_type]
    else:
        raise TypeError("No network function associated with " + model_type + ". Valid network functions: " +
                        str(list(model_function_dict.keys())))

    #make environment
    environment = empowerment.make_environment(envname)

    #make environment loop
    environment_loop = empowerment.make_environment_loop(network_function,environment, **other_arguments)

    if num_episodes is not None:
        environment_loop.run(num_episodes=num_episodes)
    elif num_steps is not None:
        environment_loop.run(num_steps=num_steps)
    else:
        raise TypeError("Either num_episodes or num_steps must not be None")

    return
