"""Goals in this file:
 - We want to be able to run any configuration of intrinsic reward model we've built with only 1 function call
 - We want the results of that run to be saved to Wandb
 - We want a video of what went down, also
"""
from MohammadDQN import gridworld_utils as empowerment
from typing import Optional
import copy
import os

#TODO: Add in logging to logdir
#TODO: Add in checkpointing to checkpoints
#TODO: Add in saving metadata to metadata
#TODO: Add in observers
#TODO: Integrate Wandb


"""We will use the experimenter to run any experiment function
It should simply take in "args", an experiment function, and run all the experiments while doing saving as necessary"""
class Experimenter:
    def __init__(self,
                 experiment_function,
                 args: dict, #arguments required for the experiment function, including "other_arguments"
                 ):
        self.experiment_function = experiment_function

        #things we always want to have:
        args['checkpoint'] = True

        self.args = args

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
            modelnum = self.get_modelnum()
            self.experiment_function(**args)

    #The public function that is called to run experiments
    def run_experiments(self):
        print("Running experiments!")
        self._run_experiments(self.args)

    def get_modelnum(self):
        existing_logs = os.listdir("logdir")
        return 0


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


ARGS = {
    #required args
    "envname": 'MiniGrid-Empty-8x8-v0',
    "model_type": 'simple',
    'num_episodes': 25,

    #extra args
    'beta': 0.5,
    'min_replay_size':10,
    'batch_size':32

}
exp = Experimenter(gridworld_empowerment_experiment, ARGS)
exp.run_experiments()