import tensorflow as tf
import tensorflow.keras as keras

W_INIT = keras.initializers.glorot_normal(seed=2)
B_INIT = tf.constant_initializer(0.1)


def create_actor_network(input_state_shape, action_dim, action_bound, scope=""):
    """
    create a policy net
    :param self:
    :param input_state_shape:
    :param name:
    :return:
    """
    input = keras.layers.Input(shape=input_state_shape)
    layer = keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer=W_INIT)(input)
    layer = keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer=W_INIT)(layer)

    mu = keras.layers.Dense(units=action_dim, activation=tf.nn.tanh, kernel_initializer=W_INIT, name='mu')(layer)
    sigma = keras.layers.Dense(units=action_dim, activation=tf.nn.softplus, kernel_initializer=W_INIT, name='sigma')(layer)
    model = keras.models.Model(inputs=input, outputs=[mu, sigma], name=scope + '_actor')
    print(model.summary())

    return model



def create_critic_network(input_state_shape, input_action_shape, scope=''):
    """

    :param self:
    :param input_state_shape:
    :param input_action_shape:
    :param name:
    :return:
    """
    status_input = keras.layers.Input(shape=input_state_shape, name='critic_state_input')
    layer = keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer=W_INIT)(status_input)
    layer = keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer=W_INIT)(layer)
    out = keras.layers.Dense(units=1, kernel_initializer=W_INIT, name='output_Q')(layer)
    model = keras.models.Model(inputs=status_input, outputs=out, name=scope + '_critic')

    print(model.summary())

    return model
