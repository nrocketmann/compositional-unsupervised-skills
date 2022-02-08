from IntrinsicRewards.Experiments import Experimenter, gridworld_empowerment_experiment

"""Pre-raytune version:"""
ARGS = {
    #required args
    #"envname": 'MiniGrid-Empty-8x8-v0',
    #"envname": 'MiniGrid-FourRooms-v0',
    #"envname": 'MiniGrid-MultiRoom-N2-S4-v0',
    #"envname": 'MiniGrid-DoorKey-5x5-v0',
    'envname':'BabyAI-PutNext-v0',
    "model_type": 'simple',
    'num_steps': 1000,
    'eval_every': 100,
    'sequence_length':10,
    'num_eval_episodes': 1,

    #extra args
    'beta': 0.5,
    'min_replay_size':10,
    'batch_size':32
}
exp = Experimenter("test",gridworld_empowerment_experiment, ARGS)
exp.run_experiments()

# """Raytune version:"""

# from IntrinsicRewards.Experiments import ray_gridworld_empowerment_experiment, RayExperimenter
# ARGS = {
#     #required args
#     #"envname": 'MiniGrid-Empty-8x8-v0',
#     #"envname": 'MiniGrid-FourRooms-v0',
#     #"envname": 'MiniGrid-MultiRoom-N2-S4-v0',
#     "envname": 'MiniGrid-DoorKey-5x5-v0',
#     #"envname": 'MiniGrid-KeyCorridorS4R3-v0',
#     "model_type": 'simple',
#     'num_steps': 1000,
#     'eval_every': 100,
#     'sequence_length':10,
#
#     #extra args
#     'beta': 0.5,
#     'min_replay_size':10,
#     'batch_size':32
# }
# exp = RayExperimenter("test",ray_gridworld_empowerment_experiment, ARGS)
# exp.run_experiments()


#TODO: make raytune execution work
#TODO: add videos
#TODO: plot graphs of env every eval step