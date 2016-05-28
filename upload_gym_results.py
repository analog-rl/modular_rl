import gym
key = open('open_ai_key.txt', 'r').read()
gym.upload('cem10-  5-mountaincar.dir', api_key=key)
