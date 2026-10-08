"""Capture a high-angle overview of the Safe-DNav Isaac Sim training scene."""

import os
import time

import hydra
from omegaconf import DictConfig
from omni.isaac.kit import SimulationApp


CFG_DIR = "/root/NavRL/isaac-training/training_net2_SAMnet_reward_cbf/cfg"


@hydra.main(config_path=CFG_DIR, config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    output_path = os.path.abspath(cfg.capture.output)
    app = SimulationApp(
        {
            "headless": True,
            "anti_aliasing": 2,
            "width": int(cfg.viewer.resolution[0]),
            "height": int(cfg.viewer.resolution[1]),
        }
    )

    from env import NavigationEnv
    from omni.isaac.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    env = NavigationEnv(cfg)
    env.reset()
    set_camera_view(eye=cfg.viewer.eye, target=cfg.viewer.lookat)

    # Allow simulation, materials, lighting, and the viewport to settle.
    for _ in range(40):
        env.sim.step(render=True)
    viewport = get_active_viewport()
    capture_viewport_to_file(viewport, output_path)
    for _ in range(30):
        app.update()
        time.sleep(0.05)

    print(f"CAPTURE_SAVED={output_path}")
    app.close()


if __name__ == "__main__":
    main()
