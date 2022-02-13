

"""Raytune version:"""
from ray import tune
from IntrinsicRewards.Experiments import EmpowermentTrainable, RayExperimenter
ARGS = {
    #required args
    #"envname": 'MiniGrid-Empty-8x8-v0',
    #"envname": 'MiniGrid-FourRooms-v0',
    #"envname": 'MiniGrid-MultiRoom-N2-S4-v0',
    #"envname": 'MiniGrid-DoorKey-5x5-v0',
    #"envname": 'MiniGrid-KeyCorridorS4R3-v0',
    'envname':'Level1',
    "model_type": 'simple',
    'num_steps': 5e6,
    'eval_every': 2e5,
    'sequence_length':10,
    'num_eval_episodes':1,

    #extra args
    'beta': tune.grid_search([0.1,1.0,10.0]),
    'min_replay_size':1e4,
    'batch_size':64
}
exp = RayExperimenter("test",EmpowermentTrainable, ARGS)
exp.run_experiments()



"""Pre-raytune version:"""
#from IntrinsicRewards.Experiments import Experimenter, gridworld_empowerment_experiment
# ARGS = {
#     #required args
#     #"envname": 'MiniGrid-Empty-8x8-v0',
#     #"envname": 'MiniGrid-FourRooms-v0',
#     #"envname": 'MiniGrid-MultiRoom-N2-S4-v0',
#     #"envname": 'MiniGrid-DoorKey-5x5-v0',
#     'envname':'BabyAI-PutNext-v0',
#     "model_type": 'simple',
#     'num_steps': 1000,
#     'eval_every': 100,
#     'sequence_length':10,
#     'num_eval_episodes': 1,
#
#     #extra args
#     'beta': 0.5,
#     'min_replay_size':10,
#     'batch_size':32
# }
# exp = Experimenter("test",gridworld_empowerment_experiment, ARGS)
# exp.run_experiments()
