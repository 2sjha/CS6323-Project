"""Main driver script"""
import numpy as np
from brax.io import html
from brax import envs
import streamlit as st
from mimic import Mimic
from humanoid_system_config import HumanoidSystemConfig
from utils import serialize_qp, deserialize_qp


envs.register_environment("mimic", Mimic)

st.set_page_config(layout="wide")
st.title("CS6323 - Project: Diffmimic")
st.subheader("Shubham Shekhar Jha (sxj220028)")
st.subheader("Vedant Sapra (vks220000)")


def main():
    """
    Main function to render all elements
    """
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("Number of Interations")
    st.write("In 200 iterations, it does not stick the landing perfectly. In 500 and 1000 iterations, it sticks the landing perfectly but 500 iterations has its legs spaced out during the backflip and at landing.")
    iter_cntnr = st.container()
    it_c1, it_c2 = iter_cntnr.columns(2)

    with it_c1:
        st.write("50 iterations")
        st.components.v1.html(render_npy("./data/default_50it.npy"), height=350, width=700)
        st.write("500 iterations")
        st.components.v1.html(render_npy("./data/default_500it.npy"), height=350, width=700)

    with it_c2:
        st.write("200 iterations")
        st.components.v1.html(render_npy("./data/default_200it.npy"), height=350, width=700)
        st.write("1000 iterations")
        st.components.v1.html(render_npy("./data/default_1000it.npy"), height=350, width=700)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("Demo Replay")
    dm_cntnr = st.container()
    dm_c1, dm_c2 = dm_cntnr.columns(2)
    with dm_c1:
        st.write("Demo Replay Mode: None")
        st.components.v1.html(render_npy("./data/no_demoreplay_300it.npy"), height=350, width=700)
    with dm_c2:
        st.write("Demo Replay Mode: threshold, threshold value = 0.5")
        st.components.v1.html(render_npy("./data/threshold_0.5_300it.npy"), height=350, width=700)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("Number of Training and Evaluation environments")
    envs_cntnr = st.container()
    with envs_cntnr:
        st.write("Training envs = 300, Evaluation envs = 32")
        st.components.v1.html(render_npy("./data/envs_330,32_300it.npy"), height=350, width=700)

    # st.markdown("<hr>", unsafe_allow_html=True)
    # st.header("Learning Rate")
    # envs_cntnr = st.container()
    # with envs_cntnr:
    #     st.write("Learning Rate = 0.001")
    #     st.components.v1.html(render_npy("./data/lr_0,001_300it.npy"), height=350, width=700)

    # st.markdown("<hr>", unsafe_allow_html=True)
    # st.header("Reward Scaling")
    # envs_cntnr = st.container()
    # with envs_cntnr:
    #     st.write("Reward Scaling = 0.01")
    #     st.components.v1.html(render_npy("./data/reward_scaling_0,1_300it.npy"), 
    #                           height=350, width=700)


def render_npy(npy_name):
    """
    Returns HTML to render the brax environment with npy loaded
    """
    t = np.load(npy_name)
    t = t[:, 0]

    rollout_qp = [deserialize_qp(t[i]) for i in range(t.shape[0])]
    t = serialize_qp(deserialize_qp(t))

    env = envs.get_environment(
        env_name="mimic",
        system_config=HumanoidSystemConfig,
        reference_traj=t,
        obs_type="timestamp",
        cyc_len=54
    )
    return html.render(env.sys, rollout_qp)


if __name__ == "__main__":
    main()
