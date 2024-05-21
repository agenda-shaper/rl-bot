from pearl.pearl_agent import PearlAgent
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.policy_learners.sequential_decision_making.ppo import (
    ProximalPolicyOptimization,
)
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment

env = GymEnvironment("CartPole-v1")

num_actions = env.action_space.n
agent = PearlAgent(
    policy_learner=DeepQLearning(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        hidden_dims=[64, 64],
        training_rounds=20,
        action_representation_module=OneHotActionTensorRepresentationModule(
            max_number_actions=num_actions
        ),
    ),
    replay_buffer=FIFOOffPolicyReplayBuffer(10_000),
)

observation, action_space = env.reset()
agent.reset(observation, action_space)
done = False
while not done:
    action = agent.act(exploit=False)
    action_result = env.step(action)
    agent.observe(action_result)
    agent.learn()
    done = action_result.done


# #!/usr/bin/env python3

# import RocketSim as rs

# # Make an arena instance (this is where our simulation takes place, has its own btDynamicsWorld instance)
# arena = rs.Arena(rs.GameMode.SOCCAR)

# # Make a new car
# car = arena.add_car(rs.Team.BLUE)

# arena.reset_to_random_kickoff()

# # Set up an initial state for our car
# car.set_state(rs.CarState(pos=rs.Vec(z=17), vel=rs.Vec(x=50)))


# # Setup a ball state
# arena.ball.set_state(rs.BallState(pos=rs.Vec(y=400, z=100)))

# # Make our car drive forward and turn
# car.set_controls(rs.CarControls(throttle=1, steer=1))

# # Simulate for 100 ticks
# arena.step(100)

# # Lets see where our car went!
# print(f"After {arena.tick_count} ticks, our car is at: {car.get_state().pos:.2f}")
