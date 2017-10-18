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
        initial_states=[state[0]]*self.num_simulated_paths
        #pdb.set_trace()
        initial_actions=[np.random.uniform(self.env.action_space.low,self.env.action_space.high,ac_dim) for _
                        in initial_states]
        states_list=[[]]*(self.horizon+1)
        action_list=[[]]*(self.horizon+1)
        states_list[0]=initial_states
        action_list[0]=initial_actions




        for i in range(1,self.horizon+1):

            act=[np.random.uniform(self.env.action_space.low,self.env.action_space.high,ac_dim) for _
                        in initial_states]
            st=states_list[i-1]
            #pdb.set_trace()
            next_states = self.dyn_model.predict(st, act)
            states_list[i]=next_states
            action_list[i]=act

        states_list=np.array(states_list)
        states_list=np.transpose(states_list,(1,0,2))
        action_list=np.array(action_list)
        action_list=np.transpose(action_list,(1,0,2))




        obs=[np.array(x) for x in states_list[0][:-1]]
        next_obs=[np.array(x) for x in states_list[0][1:]]
        actions=[np.array(x) for x in action_list[0][:-1]]

        best_costs = trajectory_cost_fn(self.cost_fn,
                                   obs, actions,
                                   next_obs)

        argmin=0
        for i in range(self.num_simulated_paths):
            obs = [np.array(x) for x in states_list[i][:-1]]
            next_obs = [np.array(x) for x in states_list[i][1:]]
            actions = [np.array(x) for x in action_list[i][:-1]]
            cost= trajectory_cost_fn(self.cost_fn,
                                            obs, actions,
                                            next_obs)
            if cost<=best_costs:

                argmin=i
        end_time = time.time()


        return action_list[argmin][0]
