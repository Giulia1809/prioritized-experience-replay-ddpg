""" 
-----------------------------------
For main.py                      #
-----------------------------------
"""
WARM_POPULATE = 10000
TRAINER_MAX_EPOCHS = 300000
VAL_CHECK_INTERVAL = 100
TRAIN = False

""" 
-----------------------------------
For agent.py                      #
-----------------------------------
"""
OU_NOISE_STD = 0.8
RENDER = True

""" 
-----------------------------------
For buffers.py                    #
-----------------------------------
"""
ALPHA = 0.6
BETA = 0.1

""" 
-----------------------------------
For ddpg.py                       #
-----------------------------------
"""
ENV = "MountainCarContinuous-v0"  # "Pendulum-v1"  # "LunarLanderContinuous-v2"  #
USE_PRIORITIZED_BUFFER = True
BATCH_SIZE = 64
ACTOR_LR = 1e-4
CRITIC_LR = 5e-4
GAMMA = 0.99
EPISODE_LENGTH = 200
TRAIN_EPISODES = 300000
SYNC_RATE = 1
TAU = 5e-3
TEST_EPISODES = 10
PRIORITIZED_REPLAY_ALPHA = 0.6
PRIORITIZED_REPLAY_BETA0 = 0.4
PRIORITIZED_REPLAY_BETA_ITERS = None
PRIORITIZED_REPLAY_EPS = 1e-6
