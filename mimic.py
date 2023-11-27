import brax
from brax import jumpy as jp
from brax.envs import env
from utils import *


class Mimic(env.Env):
    """Trains a humanoid to mimic reference motion."""

    def __init__(self, system_config, reference_traj, obs_type):
        super().__init__(config=system_config)
        self.reference_qp = deserialize_qp(reference_traj)
        self.reference_len = reference_traj.shape[0]
        self.obs_type = obs_type

    def reset(self, rng: jp.ndarray) -> env.State:
        """Resets the environment to an initial state."""
        reward, done, zero = jp.zeros(3)
        qp = self._get_ref_state(zero)
        metrics = {'step_index': zero, 'pose_error': zero, 'fall': zero}
        obs = self._get_obs(qp, step_index=zero)
        state = env.State(qp, obs, reward, done, metrics)
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        step_index = state.metrics['step_index'] + 1
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, step_index)
        state = state.replace(qp=qp, obs=obs)
        return state

    def _get_obs(self, qp: brax.QP, step_index: jp.ndarray) -> jp.ndarray:
        """Observe humanoid body position, velocities, and angles."""
        pos, rot, vel, ang = qp.pos[:-1], qp.rot[:-1], qp.vel[:-1], qp.ang[:-1]  # Remove floor
        rot_6d = quaternion_to_rotation_6d(rot)
        rel_pos = (pos - pos[0])[1:]

        obs = jp.concatenate([rel_pos.reshape(-1), rot_6d.reshape(-1), vel.reshape(-1),
                            ang.reshape(-1)], axis=-1)
        return obs

    def _get_ref_state(self, step_idx) -> brax.QP:
        mask = jp.where(step_idx == jp.arange(0, self.reference_len), jp.float32(1), jp.float32(0))
        ref_state = jp.tree_map(lambda x: (mask @ x.transpose(1, 0, 2)), self.reference_qp)
        return ref_state
