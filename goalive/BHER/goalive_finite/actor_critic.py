import tensorflow as tf
from BHER.goalive_finite.util import store_args, nn


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.goalive_finite.Normalizer): normalizer for observations
            g_stats (baselines.goalive_finite.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        g_list = [self.g_stats.normalize(item) for item in inputs_tf['fake_goal']]
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor
        input_pi_list = [tf.concat(axis=1, values=[o, item]) for item in g_list]

        # Networks.
        with tf.compat.v1.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
            self.pi_tf_list = [self.max_u * tf.tanh(nn(
                item, [self.hidden] * self.layers + [self.dimu],reuse=True)) for item in input_pi_list]
        with tf.compat.v1.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            input_Q_list = [tf.concat(axis=1, values=[o, item, self.pi_tf / self.max_u])  for item in g_list]
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            self.Q_pi_tf_list = [nn(item, [self.hidden] * self.layers + [1],reuse=True) for item in input_Q_list]
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            input_Q_list = [tf.concat(axis=1, values=[o, item, self.u_tf / self.max_u])  for item in g_list]
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
            self.Q_tf_list = [nn(item, [self.hidden] * self.layers + [1],reuse=True) for item in input_Q_list]