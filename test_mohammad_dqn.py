from IntrinsicRewards.Experiments import Experimenter, gridworld_empowerment_experiment


ARGS = {
    #required args
    #"envname": 'MiniGrid-Empty-8x8-v0',
    #"envname": 'MiniGrid-FourRooms-v0',
    #"envname": 'MiniGrid-MultiRoom-N2-S4-v0',
    #"envname": 'MiniGrid-DoorKey-5x5-v0',
    "envname": 'MiniGrid-KeyCorridorS4R3-v0',
    "model_type": 'simple',
    'num_steps': 1000,
    'eval_every': 100,
    'sequence_length':10,

    #extra args
    'beta': 0.5,
    'min_replay_size':10,
    'batch_size':32
}
exp = Experimenter("test",gridworld_empowerment_experiment, ARGS)
exp.run_experiments()

#TODO: save the charts nicely and upload to W&B
#TODO: plot trajectories?
#TODO: implement raytune