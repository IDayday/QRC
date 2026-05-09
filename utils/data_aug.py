import torch
import numpy as np


def random_translate(imgs, pad=8):
	n, c, h, w = imgs.size()
	imgs = torch.nn.functional.pad(imgs, (pad, pad, pad, pad))
	w1 = torch.randint(0, 2*pad + 1, (n,))
	h1 = torch.randint(0, 2*pad + 1, (n,))
	cropped = torch.empty((n, c, h, w), dtype=imgs.dtype, device=imgs.device)
	for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
		cropped[i][:] = img[:, h11:h11 + h, w11:w11 + w]
	return cropped

def sample_and_preprocess_batch(replay_buffer, batch_size=1024, distance_threshold=0.5, device=torch.device("cuda")):
    # Extract 
    batch = replay_buffer.random_batch(batch_size)
    state_batch         = batch["observations"]
    action_batch        = batch["actions"]
    next_state_batch    = batch["next_observations"]
    goal_batch          = batch["resampled_goals"]
    reward_batch        = batch["rewards"]
    done_batch          = batch["terminals"] 
    
    # Compute sparse rewards: -1 for all actions until the goal is reached
    reward_batch = - np.sqrt(np.power(np.array(next_state_batch - goal_batch)[:, :2], 2).sum(-1, keepdims=True))
    done_batch   = 1.0 * (reward_batch > -distance_threshold) 
    reward_batch = - np.ones_like(done_batch)

    # Convert to Pytorch
    state_batch         = torch.FloatTensor(state_batch).to(device)
    action_batch        = torch.FloatTensor(action_batch).to(device)
    reward_batch        = torch.FloatTensor(reward_batch).to(device)
    next_state_batch    = torch.FloatTensor(next_state_batch).to(device)
    done_batch          = torch.FloatTensor(done_batch).to(device)
    goal_batch          = torch.FloatTensor(goal_batch).to(device)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch