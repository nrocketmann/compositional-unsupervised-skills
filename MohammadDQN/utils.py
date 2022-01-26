import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

"""Really simple MLP model that outputs a probability of a thing given another thing"""
class ProbabilityPredictor(snt.Module):
    def __init__(self,
                 shared_hiddens: tuple = (256,256),
                 output_dims: int = 1,
                 activation = tf.nn.relu,
                 categorical = True):
        super(ProbabilityPredictor,self).__init__()
        self.shared_layers = [snt.Linear(i) for i in shared_hiddens]
        self.mean_layer = snt.Linear(output_dims)
        if not categorical:
            self.std_layer = snt.Linear(output_dims) #outputs log std
        self.activation = activation
        self.categorical = categorical

    def __call__(self, variable, given): #outputs logprobs
        shared_rep = given
        for layer in self.shared_layers:
            shared_rep = layer(shared_rep)
            shared_rep = self.activation(shared_rep)
        mean = self.activation(self.mean_layer(shared_rep))
        if not self.categorical:
            log_std = self.activation(self.std_layer(shared_rep))
            distro = tfp.distributions.Normal(loc=mean,std = tf.exp(log_std))
        else:
            distro = tfp.distributions.OneHotCategorical(logits=mean)
            variable = tf.cast(variable, tf.int32)
        return distro.log_prob(variable)


"""See appendix D of https://arxiv.org/pdf/1509.08731.pdf
This network calculates the probability of a sequence as a product of conditional probabilities.
It just contains a ProbabilityPredictor and repeats it a bunch on the inputs"""
class ConditionalProductNetwork(snt.Module):
    def __init__(self,
                 shared_hiddens: tuple = (256,256),
                 output_dims: int = 1,
                 activation = tf.nn.relu,
                 categorical = True):
        super(ConditionalProductNetwork,self).__init__()
        self.probnet = ProbabilityPredictor(shared_hiddens, output_dims, activation, categorical)
        self.output_dims = output_dims

    def __call__(self, action_sequence, givens):
        #action sequence has shape batch x timesteps x dims
        #givens has shape batch x dims
        action_sequence = tf.cast(action_sequence[:,:-1],tf.float32)
        givens = tf.cast(givens,tf.float32)
        input_shape = tf.shape(action_sequence)
        batch_size = input_shape[0]
        timesteps = input_shape[1]
        given_actions = tf.concat([tf.zeros([batch_size, 1, self.output_dims],dtype=action_sequence.dtype),action_sequence[:,:-1]],axis=1)

        #now concatenate all givens
        givens = tf.concat([tf.tile(tf.expand_dims(givens,1),[1,timesteps, 1]),given_actions],axis=-1)

        log_probs = self.probnet(action_sequence, givens)
        return tf.reduce_sum(log_probs, axis=1) #sum over all timesteps to get product of probabilities


class RNetwork(snt.Module):
    def __init__(self,
                 shared_hiddens: tuple = (256, 256),
                 output_dims: int = 1,
                 activation=tf.nn.relu,
                 categorical=True,
                 phi_network_layers: tuple = (256,256)):
        super(RNetwork, self).__init__()
        self.h_network = ConditionalProductNetwork(shared_hiddens,output_dims,activation,categorical)
        self.phi_network = snt.nets.MLP(list(phi_network_layers) + [1])

    def __call__(self, action_sequence, givens):
        action_sequence = tf.cast(action_sequence,tf.float32)
        probability = self.h_network(action_sequence, givens)
        score = self.phi_network(givens)
        return probability + score, score
