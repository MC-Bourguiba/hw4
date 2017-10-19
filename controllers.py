import numpy as np
from cost_functions import trajectory_cost_fn
import time
import pdb

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        ac_dim = self.env.action_space.shape[0]


        rand_ac=np.random.uniform(self.env.action_space.low,self.env.action_space.high,ac_dim)
        return rand_ac


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE """

        import time
        start_time=time.time()

        ac_dim = self.env.action_space.shape[0]
        states=np.array([state[0]]*self.num_simulated_paths)
        states=np.expand_dims(states, axis=0)

        sample_actions=np.random.uniform(self.env.action_space.low,self.env.action_space.high,
                                         (self.horizon,self.num_simulated_paths,ac_dim))

        for i in range(self.horizon):
            import pdb

            next_states=self.dyn_model.predict(states[-1],sample_actions[i])
            next_states = np.expand_dims(next_states, axis=0)
            states=np.vstack((states,next_states))

        
        costs=trajectory_cost_fn(self.cost_fn, states[:-1], sample_actions, states[1:])

        return sample_actions[0][np.argmin(costs),:]