def sample_action(policy_parameters):
    """
    Stochastically sampling from the policy distribution

    arguments:
        policy_parameters: logits of a categorical distribution over actions
                sy_logits_na: (batch_size, self.ac_dim)

    returns:
        sy_sampled_ac: (batch_size,)
    """

    sy_logits_na = policy_parameters
    #========================================================================================#
    #                           ----------PROBLEM 1----------
    #========================================================================================#
    # Stochastically sampling an action from the policy distribution $\pi_\theta(a|s)$.
    # ------------------------------------------------------------------
    # START OF YOUR CODE
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # END OF YOUR CODE
    # ------------------------------------------------------------------

    return sy_sampled_ac


def sample_trajectory(env):
    ob = env.reset()
    obs, acs, rewards = [], [], []
    steps = 0
    while True:

        obs.append(ob)
        #====================================================================================#
        #                           ----------PROBLEM 1----------
        #====================================================================================#
        # obtain the action 'ac' for current observation 'ob'
        # ------------------------------------------------------------------
        # START OF YOUR CODE
        # ------------------------------------------------------------------




        # ------------------------------------------------------------------
        # END OF YOUR CODE
        # ------------------------------------------------------------------
        ac = ac.numpy()[0]
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        rewards.append(rew)
        steps += 1
        if done or steps > max_path_length:
            break
    path = {"observation" : np.array(obs, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32)}
    return path


def sum_of_rewards(re_n):
    """ Monte Carlo estimation of the Q function.

    let sum_of_path_lengths be the sum of the lengths of the paths sampled from
        the function sample_trajectories
    let num_paths be the number of paths sampled from sample_trajectories

    arguments:
        re_n: length: num_paths. Each element in re_n is a numpy array
            containing the rewards for the particular path

    returns:
        q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
            whose length is the sum of the lengths of the paths
    ----------------------------------------------------------------------------------

    Your code should construct numpy arrays for Q-values which will be used to compute
    advantages.


    You will write code for trajectory-based PG: 

          We use the total discounted reward summed over
          entire trajectory (regardless of which time step the Q-value should be for).

          For this case, the policy gradient estimator is

              E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]

          where

              tau=(s_0, a_0, ...) is a trajectory,
              Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.

          Thus, you should compute

              Q_t = Ret(tau)

    Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
    like the 'ob_no' and 'ac_na' above.
    """
    #====================================================================================#
    #                           ----------PROBLEM 1----------
    #====================================================================================#
    # q_n: A single vector for the estimated q values whose length is the sum of the lengths of the paths.
    # Q-values: Q_t = Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}. 
    # Store the Q-values for all timesteps and all trajectories in a variable 'q_n'.
    # ------------------------------------------------------------------
    # START OF YOUR CODE
    # ------------------------------------------------------------------



    # # ------------------------------------------------------------------
    # END OF YOUR CODE
    # ------------------------------------------------------------------
    return q_n


def estimate_return(ob_no, re_n):
    """ Estimates the returns over a set of trajectories.

    let sum_of_path_lengths be the sum of the lengths of the paths sampled from
        sample_trajectories
    let num_paths be the number of paths sampled from sample_trajectories

    arguments:
        ob_no: shape: (sum_of_path_lengths, ob_dim)
        re_n: length: num_paths. Each element in re_n is a numpy array
            containing the rewards for the particular path

    returns:
        q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
            whose length is the sum of the lengths of the paths
        adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
            advantages whose length is the sum of the lengths of the paths
    """
    q_n = sum_of_rewards(re_n)
    adv_n = compute_advantage(ob_no, q_n)
    #====================================================================================#
    #                           ----------PROBLEM 2----------
    # Advantage Normalization
    #====================================================================================#
    if normalize_advantages:
        # On the next line, implement a trick which is known empirically to reduce variance
        # in policy gradient methods: normalize adv_n to have mean zero and std=1.
        # ------------------------------------------------------------------
        # START OF YOUR CODE
        # ------------------------------------------------------------------



        # ------------------------------------------------------------------
        # END OF YOUR CODE
        # ------------------------------------------------------------------
    return q_n, adv_n


def get_log_prob(policy_parameters, sy_ac_na):
    """
    Computing the log probability of a set of actions that were actually taken according to the policy

    arguments:
        policy_parameters: logits of a categorical distribution over actions
                sy_logits_na: (batch_size, self.ac_dim)

        sy_ac_na: (batch_size,)

    returns:
        sy_logprob_n: (batch_size)

    Hint:
        For the discrete case, use the log probability under a categorical distribution.
    """

    sy_logits_na = policy_parameters
    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    # sy_logprob_n = \sum_{t=1}^T \log \pi_\theta(a_{it}|s_{it})
    # ------------------------------------------------------------------
    # START OF YOUR CODE
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # END OF YOUR CODE
    # ------------------------------------------------------------------
    return sy_logprob_n


def update_parameters(ob_no, ac_na, q_n, adv_n):
    """
    Update the parameters of the policy and (possibly) the neural network baseline,
    which is trained to approximate the value function.

    arguments:
        ob_no: shape: (sum_of_path_lengths, ob_dim)
        ac_na: shape: (sum_of_path_lengths).
        q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
            whose length is the sum of the lengths of the paths
        adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
            advantages whose length is the sum of the lengths of the paths

    returns:
        nothing
    """
    #====================================================================================#
    #                           ----------PROBLEM 2----------
    #====================================================================================#
    # Performing the Policy Update based on the current batch of rollouts.
    # 
    # ------------------------------------------------------------------
    # START OF YOUR CODE
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # END OF YOUR CODE
    # ------------------------------------------------------------------



