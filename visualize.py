import numpy as np
import jax.numpy as jnp
import brax
from brax import QP, envs
from brax.io import html
import streamlit.components.v1 as components
import streamlit
from mimic import Mimic
from humanoid_system_config import HumanoidSystemConfig
from utils import serialize_qp, deserialize_qp


envs.register_environment("mimic", Mimic)

streamlit.title("CS6323 - Project")
streamlit.subheader("Shubham Shekhar Jha (sxj220028)")
streamlit.subheader("Vedant Sapra (vks220000)")


def main():
    t = np.load("default_50it.npy")
    t = t[:, 0]

    rollout_qp = [deserialize_qp(t[i]) for i in range(t.shape[0])]
    t = serialize_qp(deserialize_qp(t))

    env = envs.get_environment(
        env_name="mimic",
        system_config=HumanoidSystemConfig,
        reference_traj=t,
        obs_type="timestamp",
        cyc_len=None,
        reward_scaling=1.0,
        rot_weight=1.0,
        vel_weight=0.0,
        ang_weight=0.0,
    )
    components.html(html.render(env.sys, rollout_qp), height=500)


if __name__ == "__main__":
    main()
