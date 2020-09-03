import torch
import numpy as np

def compute_returns(rewards, dones):
    """ Vanilla method of computing discounted returns. Does not use GAE """
    
    GAMMA = .95
    R = []
    r2 = rewards[-1] # this is ~v(s)
    R.append(r2)
    for i in reversed(range(0, len(rewards)-1)):
        reward = rewards[i]
        r1_is_terminal = (dones[i]-1)*-1 # so Dones are zeros
        r1 = reward + ((r2 * GAMMA)*r1_is_terminal) # If r1 is terminal, don't add in the discounted returns of next state
        R.append(r1)
        r2=r1
        
    R.reverse()
    R = np.array(R)
    return R

def compute_returns_gae(rewards, dones, values):
    """ Compute discounted returns w Generalized Advantage Estimation (GAE) """
    
    GAMMA = .999
    LAM = .95
    lastgaelam = 0
    A = [np.zeros(rewards.shape[-1])]
    for t in reversed(range(0, len(rewards)-1)):
        nextnonterminal = 1.0 - dones[t] # openai baselines records dones in different way. Customized this. Was t+1
        nextvalues = values[t+1]
        delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + GAMMA * LAM * nextnonterminal * lastgaelam
        A.append(lastgaelam)
    A.reverse()
    A = np.array(A)
    R = A + values
    return R

def get_batch(batch_ix, device, frames, returns, actions, old_action_probs, old_state_estimates):
    return (frames[batch_ix].to(device),
            returns[batch_ix].to(device), 
            actions[batch_ix].to(device),
            old_action_probs[batch_ix].to(device),
            old_state_estimates[batch_ix].to(device))

def calculate_action_gain_ppo(action_probs, old_action_probs, actions_taken, advantages, device, verbose=False):
    """ Actions resulting in positive advantage made more probable, and vice versa. Do not allow to differ from old
    probs by too much. Log probs are from current agent. Actions_taken are from OLD version of agent. """
    
    a = actions_taken.unsqueeze(-1).to(device)
    if verbose: print("\n\nactions unsqueezed\n", a)
        
    chosen_action_probs = action_probs.gather(1, a)
    chosen_action_probs = chosen_action_probs.squeeze(-1)
    
    old_chosen_action_probs = old_action_probs.gather(1, a)
    old_chosen_action_probs = old_chosen_action_probs.squeeze(-1)
    if verbose: print("\n\nchosen action probs, new and old\n", chosen_action_probs,'\n', old_chosen_action_probs, '\n\nadvantages values', advantages)
    
    ratio = torch.exp(chosen_action_probs - old_chosen_action_probs)
    if verbose: print('\n\nratio', ratio)
    
    unclipped_action_gain = ratio * advantages
    clipped_action_gain = torch.clamp(ratio, .8, 1.2) * advantages
    if verbose: print('\n\nunclipped and clipped action gains\n', unclipped_action_gain, '\n', clipped_action_gain)
    
    action_gain = torch.min(unclipped_action_gain, clipped_action_gain)
    if verbose: print('\n\n conservative lower bound action gain\n', action_gain)
        
    action_gain = action_gain.mean()
    if verbose: print('\n\nmean', action_gain)
        
    return action_gain # single scalar


def get_critic_loss(old_state_estimates_b, state_estimates, returns_b, verbose=False):
    """ How good is critic at estimating state? Don't allow to differ too much from old state estimates """
    
    state_estimates = state_estimates.squeeze(-1)
    clipped_state_estimate = old_state_estimates_b + torch.clamp(state_estimates - old_state_estimates_b, -.2, .2)
    if verbose: print("\nstate estimates, new and old\n\n", state_estimates, '\n', old_state_estimates_b, '\n\nreturns', returns_b)
    critic_loss_1 = ((returns_b - clipped_state_estimate)**2)
    critic_loss_2 = ((returns_b - state_estimates)**2)
    critic_loss = torch.max(critic_loss_1, critic_loss_2)
    if verbose: print('\nCritic Losses: clipped, unclipped, conservative:\n', critic_loss_1, '\n', critic_loss_2, '\n', critic_loss)
    critic_loss = critic_loss.mean() * .5
    return critic_loss # single scalar

def get_entropy_bonus(action_probs):
    """ Encourage humility """
    e = -(action_probs.exp() * (action_probs+1e-8))
    e = e.sum(dim=1)
    e = e.mean()
    return e

def run_batch(agent, batch_ix, device, frames, returns, actions, old_action_probs, old_state_estimates, verbose=False):
    """ 
    Run a single batch of data. Takes in indices, uses them to pull from global database of rollouts,
    calculates and returns losses. No gradient steps here. 
    """
    
    frames_b, returns_b, actions_taken_b, old_action_probs_b, old_state_estimates_b = get_batch(batch_ix,
                                                                                                device,
                                                                                                frames,
                                                                                                returns,
                                                                                                actions, 
                                                                                                old_action_probs, 
                                                                                                old_state_estimates)

    action_probs, state_estimates = agent(frames_b)
    
    entropy_bonus = get_entropy_bonus(action_probs)
    
    critic_loss = get_critic_loss(old_state_estimates_b, state_estimates, returns_b, verbose=verbose)
    
    with torch.no_grad():
        advantages = (returns_b - state_estimates.squeeze(-1).detach()) # don't want to propogate gradients through this
        advantages -= advantages.mean()
        advantages /= (advantages.std() + 1e-8)
    
    action_gain = calculate_action_gain_ppo(action_probs, 
                                            old_action_probs_b,
                                            actions_taken_b, 
                                            advantages, 
                                            device, 
                                            verbose=verbose)
    
    return entropy_bonus, action_gain, critic_loss