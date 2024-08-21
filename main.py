import numpy as np
import rlgym_sim
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger

from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    VelocityPlayerToBallReward,
    VelocityBallToGoalReward,
    EventReward,
    LiuDistancePlayerToBallReward,
    LiuDistanceBallToGoalReward,
    FaceBallReward,
)
from rlgym_ppo import Learner
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import (
    TimeoutCondition,
    GoalScoredCondition,
)
from rlgym_sim.utils import common_values
from rlgym_sim.utils.action_parsers import ContinuousAction
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from numpy import ndarray


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [
            game_state.players[0].car_data.linear_velocity,
            game_state.players[0].car_data.rotation_mtx(),
            game_state.orange_score,
        ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {
            "x_vel": avg_linvel[0],
            "y_vel": avg_linvel[1],
            "z_vel": avg_linvel[2],
            "Cumulative Timesteps": cumulative_timesteps,
        }
        wandb_run.log(report)


class KickoffReward(RewardFunction):
    """
    a simple reward that encourages driving towards the ball fast while it's in the neutral kickoff position
    """

    def __init__(self):
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.vel_dir_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: ndarray
    ) -> float:
        reward = 0
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            reward += self.vel_dir_reward.get_reward(player, state, previous_action)
        return reward


def build_rocketsim_env():

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(
        round(timeout_seconds * game_tick_rate / tick_skip)
    )  # 1200 ticks reset game

    terminal_conditions = [
        TimeoutCondition(timeout_ticks),
        GoalScoredCondition(),
    ]
    action_parser = ContinuousAction()

    reward_fn = CombinedReward.from_zipped(
        # Format is (func, weight)
        (FaceBallReward(), 0.05),
        (KickoffReward(), 0.8),
        (LiuDistancePlayerToBallReward(), 0.1),
        (VelocityPlayerToBallReward(), 0.02),
        (VelocityBallToGoalReward(), 0.1),
        (LiuDistanceBallToGoalReward(), 0.1),
        (
            EventReward(
                team_goal=1,
                concede=-1,
                demo=0.6,
                boost_pickup=0.04,
                touch=0.02,
            ),
            10.0,
        ),
    )

    obs_builder = DefaultObs(
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
    )

    state_setter = DefaultState()
    env = rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=team_size,
        copy_gamestate_every_step=True,
        spawn_opponents=spawn_opponents,
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_setter=state_setter,
    )
    import rocketsimvis_rlgym_sim_client as rsv

    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env


if __name__ == "__main__":

    metrics_logger = ExampleLogger()

    # 1 processes
    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rocketsim_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=metrics_logger,
        ppo_batch_size=50000,
        ts_per_iteration=50000,
        exp_buffer_size=150000,
        ppo_minibatch_size=50000,
        ppo_ent_coef=0.005,  # more = more entropy in actions
        ppo_epochs=1,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=500_000,
        timestep_limit=1_000_000_000,
        log_to_wandb=True,
        # render=True,
        # render_delay=0,
    )
    learner.learn()
