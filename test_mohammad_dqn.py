from IntrinsicRewards.Experiments import Experimenter, gridworld_empowerment_experiment


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
exp = Experimenter("test",gridworld_empowerment_experiment, ARGS)
exp.run_experiments()