# cart_pole_v0

During my study in reinforcement learning I was curious about Monte Carlo algorithm. In order to better understand the algorithm, I've tried to apply it to the cart-pole [problem](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py).

In order to run the training is sufficient to run train.py. If a log is required it is possible to use:
```
/bin/python3 -u /home/mm/github/cart_pole_v0/train.py | tee log_000.log
```
Description of the problem:
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

A reward of +1 is given for each step of the episode. The episode terminates if 200 consecutive steps are reached or if the pole falls.

The problem is considered solved when the average return is greater than or equal to 195.0 over 100 consecutive trials.

The problem has been solved with an average score of 196.6 over 100 episodes after 2200 training episodes.

