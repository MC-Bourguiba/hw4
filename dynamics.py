import tensorflow as tf
import numpy as np
import pdb

# Predefined function to build a feedforward neural network


def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

def compute_normalized_input_output(data,normalization):

    mean_obs = normalization[0]
    std_obs = normalization[1]
    mean_deltas = normalization[2]

    std_deltas = normalization[3]
    mean_action = normalization[4]
    std_action = normalization[5]
    epsilon = 0.0000001
    std_obs    = std_obs+epsilon
    std_deltas = std_deltas+epsilon
    std_action = std_action+epsilon



    states=np.array(data['observations'])
    actions=np.array(data['actions'])
    deltas=np.array(data['deltas'])
    normalized_states=(states-mean_obs)/std_obs
    normalized_actions=(actions-mean_action)/std_action
    normalized_deltas=(deltas-mean_deltas)/std_deltas

    normalized_input=np.concatenate((normalized_states,normalized_actions),axis=1)



    return normalized_input,normalized_deltas

def sample(input,output,batch_size):
    input_batch=[]
    output_batch=[]
    full_data_set_size=len(input)
    for i in range(batch_size):
        import random
        index=random.choice(list(range(full_data_set_size)))
        input_batch.append(np.array(input[index]))
        output_batch.append(np.array(output[index]))

    return np.array(input_batch),np.array(output_batch)

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """




        self.env=env
        self.n_layers=n_layers
        self.size=size
        self.activation=activation
        self.output_activation=output_activation
        self.batch_size=batch_size
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.initialized=False
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.input_ph = tf.placeholder(shape=[None, self.ob_dim + self.ac_dim],
                                       name="input", dtype=tf.float32)
        self.nn=build_mlp(self.input_ph,self.ob_dim,'dynamicModel',self.n_layers,
                              self.size,self.activation,self.output_activation)


        self.sess=sess
        self.normalization=normalization
        self.sy_output_t = tf.placeholder(shape=[None, self.ob_dim], name="output", dtype=tf.float32)

        self.loss = tf.losses.mean_squared_error(labels=self.sy_output_t,
                                                 predictions=self.nn)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)



    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """

        #normalize data

        normalized_input, normalized_output = compute_normalized_input_output(data, self.normalization)
        for i in range(self.iterations):
            input_batch,output_batch=sample(normalized_input,normalized_output,self.batch_size)

            a,b=self.sess.run([self.update_op,self.loss],feed_dict={self.input_ph: input_batch,self.sy_output_t:output_batch})
            print(b)


    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """

        states=np.array(states)
        # preprocessing input
        mean_obs = self.normalization[0]
        std_obs = self.normalization[1]
        mean_deltas = self.normalization[2]

        std_deltas = self.normalization[3]
        mean_action = self.normalization[4]
        std_action = self.normalization[5]
        epsilon = 0.0000001
        std_obs = std_obs + epsilon
        std_deltas = std_deltas + epsilon
        std_action = std_action + epsilon


        normalized_states = (states - mean_obs) / std_obs
        normalized_actions = (actions - mean_action) / std_action
        #pdb.set_trace()
        normalized_input = np.concatenate((normalized_states, normalized_actions), axis=1)
        normalized_output=self.sess.run(self.nn,feed_dict={self.input_ph:normalized_input})
        #pdb.set_trace()
        unormalized_output=normalized_output*std_deltas+mean_deltas
        output=states+unormalized_output

        return output