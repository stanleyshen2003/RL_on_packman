from ddqn_agent_atari import AtariDDQNAgent

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
		"gpu": True,
		"training_steps": 1e7,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_min": 0.1,
		"warmup_steps": 200,
		"eps_decay": 1000000,
		"eval_epsilon": 0.000,
		"replay_buffer_capacity": 100000,
		"logdir": 'log/DDQN/',
		"update_freq": 4,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
        "eval_interval": 100,
        "eval_episode": 5,
		"env_id": 'ALE/MsPacman-v5',
	}
    agent = AtariDDQNAgent(config)
    #agent.train()
    agent.load_and_evaluate("model_save\DDQN\model_9984918_2568.pth")