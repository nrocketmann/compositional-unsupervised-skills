

"""Raytune version:"""
# from ray import tune
# from IntrinsicRewards.Experiments import EmpowermentTrainable, RayExperimenter
# ARGS = {
#     #required args
#     #"envname": 'MiniGrid-Empty-8x8-v0',
#     #"envname": 'MiniGrid-FourRooms-v0',
#     #"envname": 'MiniGrid-MultiRoom-N2-S4-v0',
#     #"envname": 'MiniGrid-DoorKey-5x5-v0',
#     #"envname": 'MiniGrid-KeyCorridorS4R3-v0',
#     'envname':'Level1',
#     "model_type": 'simple',
#     'num_steps': int(5e5),
#     'eval_every': int(5e3),
#     'sequence_length':10,
#     'num_eval_episodes':1,
#     'learning_rate':.001,
#
#     #extra args
#     'beta': 10.0,
#     'min_replay_size':int(1e5),
#     'batch_size':64
# }
# exp = RayExperimenter("extrinsic-debug2",EmpowermentTrainable, ARGS,local_mode=False)
# exp.run_experiments()



"""Pre-raytune version:"""
from IntrinsicRewards.Experiments import Experimenter, gridworld_empowerment_experiment
ARGS = {
    #required args
    #"envname": 'MiniGrid-Empty-8x8-v0',
    #"envname": 'MiniGrid-FourRooms-v0',
    #"envname": 'MiniGrid-MultiRoom-N2-S4-v0',
    #"envname": 'MiniGrid-DoorKey-5x5-v0',
    #"envname": 'MiniGrid-KeyCorridorS4R3-v0',
    'envname':'Level1',
    "model_type": 'simple',
    'num_steps': int(5e5),
    'eval_every': 5000,
    'sequence_length':2,
    'num_eval_episodes':1,
    'learning_rate':.001,

    #extra args
    'beta': 10.0,
    'min_replay_size':10000,
    'batch_size':64
}
exp = Experimenter("debugging",gridworld_empowerment_experiment, ARGS)
exp.run_experiments()
