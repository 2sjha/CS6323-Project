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
st.title("CS6323 - Project: DiffMimic")
st.subheader("Shubham Shekhar Jha (sxj220028)")
st.subheader("Vedant Sapra (vks220000)")


def main():
    """
    Main function to render all elements
    """
    tabs = [
        "Iterations",
        "Demo Replay",
        "Environments",
        "Learning Rate",
        "Reward Scaling",
    ]
    it_tab, dr_tab, env_tab, lr_tab, rs_tab = st.tabs(tabs)

    with it_tab:
        st.header("Number of Interations")
        st.markdown(
            """
            In 50 iterations the model barely learns anything and the policy learned is nowhere 
            close to the intended backflip. In 200 iterations, the landing is not optimal and 
            the humanoid is not stable after its landing.
            In 500 and 1000 iterations, it sticks the landing perfectly but 500 iterations has 
            its legs spaced out during the backflip and at landing.
            """
        )
        iter_cntnr = st.container()
        it_c1, it_c2 = iter_cntnr.columns(2)

        with it_c1:
            st.write("50 iterations")
            st.components.v1.html(
                render_npy("./data/default_50it.npy"), height=350, width=700
            )
            st.write("500 iterations")
            st.components.v1.html(
                render_npy("./data/default_500it.npy"), height=350, width=700
            )

        with it_c2:
            st.write("200 iterations")
            st.components.v1.html(
                render_npy("./data/default_200it.npy"), height=350, width=700
            )
            st.write("1000 iterations")
            st.components.v1.html(
                render_npy("./data/default_1000it.npy"), height=350, width=700
            )

    with dr_tab:
        st.header("Demo Replay")
        st.markdown(
            """
            With demo replay enabled (with threshold value 0.5) - The model performs 
            reasonably well but the landing is unstable as seen below.
            Without demo replay, the humanoid has the hand motions seemingly right 
            in the beginning but one of the legs stays grounded during the entire flip. 
            """
        )
        dm_cntnr = st.container()
        dm_c1, dm_c2, dm_c3 = dm_cntnr.columns(3)
        with dm_c1:
            st.write("Demo Replay Mode: threshold, threshold value = 0.4")
            st.components.v1.html(
                render_npy("./data/default_500it.npy"), height=350, width=500
            )
        with dm_c2:
            st.write("Demo Replay Mode: None")
            st.components.v1.html(
                render_npy("./data/no_demoreplay_300it.npy"), height=350, width=500
            )
        with dm_c3:
            st.write("Demo Replay Mode: threshold, threshold value = 0.5")
            st.components.v1.html(
                render_npy("./data/threshold_0.5_300it.npy"), height=350, width=500
            )

    with env_tab:
        st.header("Number of Training and Evaluation environments")
        st.markdown(
            """
            With 300, 32 environments for training and evaluation respectively, the learned policy 
            is reasonably similar to the configuration above and takes about 30 minutes to complete, 
            about 10 minutes less than the default configuration
            150, 16.
            """
        )
        envs_cntnr = st.container()
        with envs_cntnr:
            st.write("Training envs = 300, Evaluation envs = 32")
            st.components.v1.html(
                render_npy("./data/envs_300,32_300it.npy"), height=350, width=700
            )

    with lr_tab:
        st.header("Learning Rate")
        st.markdown(
            """
            With learning rate = 0.001, The model tries to converge faster but ends up in an 
            unstable position while landing even though the backflip was executed well.
            """
        )
        lr_cntnr = st.container()
        lr_c1, lr_c2 = lr_cntnr.columns(2)
        with lr_c1:
            st.write("Default Learning Rate = 0.0003")
            st.components.v1.html(
                render_npy("./data/default_500it.npy"), height=350, width=700
            )
        with lr_c2:
            st.write("Learning Rate = 0.001")
            st.components.v1.html(
                render_npy("./data/lr_0,001_300it.npy"), height=350, width=700
            )

    with rs_tab:
        st.header("Reward Scaling")
        st.markdown(
            """
            With reward scaling = 0.01, we observe that the humanoid does not stick the landing 
            very well even after performing a better backflip than the default configuration. 
            """
        )
        rs_cntnr = st.container()
        rs_c1, rs_c2 = rs_cntnr.columns(2)
        with rs_c1:
            st.write("Default Reward Scaling = 0.002")
            st.components.v1.html(
                render_npy("./data/default_500it.npy"), height=350, width=700
            )
        with rs_c2:
            st.write("Reward Scaling = 0.01")
            st.components.v1.html(
                render_npy("./data/reward_scaling_0,1_300it.npy"), height=350, width=700
            )

    st.markdown(
        """
                ## Citation
                If you find our work useful for your research, please consider citing the paper:
                ```
                @inproceedings{ren2023diffmimic,
                author    = {Ren, Jiawei and Yu, Cunjun and Chen, Siwei and Ma, Xiao and Pan, Liang and Liu, Ziwei},
                title     = {DiffMimic: Efficient Motion Mimicking with Differentiable Physics},
                journal   = {ICLR},
                year      = {2023},
                }
                ```
                """
    )
    st.markdown(
        """
                ### Our adaptation of Diffmimic: https://github.com/2sjha/CS6323-diffmimic/
                """
    )
    st.markdown(
        """
                ### Source code for this visualization: https://github.com/2sjha/CS6323-Project/
                """
    )


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
        cyc_len=54,
    )
    return html.render(env.sys, rollout_qp)


if __name__ == "__main__":
    main()
