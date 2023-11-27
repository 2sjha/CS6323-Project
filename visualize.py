import numpy as np
from brax.io import html
from brax import envs
import streamlit as st
from mimic import Mimic
from humanoid_system_config import HumanoidSystemConfig
from utils import serialize_qp, deserialize_qp


envs.register_environment("mimic", Mimic)

st.title("CS6323 - Project")
st.subheader("Shubham Shekhar Jha (sxj220028)")
st.subheader("Vedant Sapra (vks220000)")


def main():
    iter_cntnr = st.container()
    it_c1, it_c2 = iter_cntnr.columns(2)

    with it_c1:
        st.components.v1.html(render_npy("default_50it.npy"), height=400, width=500)
        st.components.v1.html(render_npy("default_200it.npy"), height=400, width=500)

    with it_c2:
        st.components.v1.html(render_npy("default_200it.npy"), height=400, width=500)
        st.components.v1.html(render_npy("default_200it.npy"), height=400, width=500)


def render_npy(npy_name):
    t = np.load(npy_name)
    t = t[:, 0]

    rollout_qp = [deserialize_qp(t[i]) for i in range(t.shape[0])]
    t = serialize_qp(deserialize_qp(t))

    env = envs.get_environment(
        env_name="mimic",
        system_config=HumanoidSystemConfig,
        reference_traj=t,
        obs_type="timestamp"
    )
    return html.render(env.sys, rollout_qp)

if __name__ == "__main__":
    main()
