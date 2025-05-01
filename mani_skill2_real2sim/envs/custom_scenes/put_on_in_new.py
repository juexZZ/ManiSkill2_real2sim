from collections import OrderedDict
from typing import List

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim import ASSET_DIR

from .base_env import CustomBridgeObjectsInSceneEnv, CustomBridgeObjectsInSceneEnvV1
from .put_on_in_scene import PutOnInSceneEnv, PutOnBridgeInSceneEnv, PutOnBridgeInSceneEnvV1

# new tasks for testing the generalization boundary

# * seen objects, unseen combination
# * clear background, straightforward language instruction
@register_env("PutGreenCubeOnPlateInScene-v2", max_episode_steps=60)
class PutGreenCubeOnPlateInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "baked_green_cube_3cm"
        target_obj_name = "bridge_plate_objaverse_larger"
        
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [
            np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi/4), [1, 0, 0, 0]]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put green cube on plate"
  
    
@register_env("PutSmallPlateOnGreenCubeInScene-v2", max_episode_steps=60)
class PutSmallPlateOnGreenCubeInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_plate_objaverse_smaller"
        target_obj_name = "baked_green_cube_3cm"
        
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [
            np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([[1, 0, 0, 0], euler2quat(0, 0, np.pi/4)]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put the small plate on the green cube"


# * coke on plate, unseen source object, seen target object
# * clear background, straightforward language instruction
@register_env("PutCokeCanOnPlateInScene-v2", max_episode_steps=60)
class PutCokeCanOnPlateInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "coke_can"
        target_obj_name = "bridge_plate_objaverse_larger"
        
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [
            np.array(
                [
                    euler2quat(np.pi/2, 0, 0),  # upright can
                    [1, 0, 0, 0]
                ]
            ),
            np.array(
                [
                    euler2quat(0, 0, np.pi/2),  # lay vertically can
                    [1, 0, 0, 0]
                ]
            ),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put coke can on plate"


@register_env("PutCarrotOnSpongeInScene-v2", max_episode_steps=60)
class PutCarrotOnSpongeInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "sponge"
        
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [
            np.array(
                [
                    euler2quat(0, 0, np.pi), 
                    [1, 0, 0, 0]
                ]
            ),
            np.array(
                [
                    euler2quat(0, 0, -np.pi / 2), 
                    [1, 0, 0, 0]
                ]
            ),
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put carrot on sponge"
        
        
@register_env("PutCarrotOnKeyboardInScene-v2", max_episode_steps=60)
class PutCarrotOnKeyboardInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "keyboard"
        
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [
            np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put carrot on keyboard"
    
@register_env("PutCokeCanOnKeyboardInScene-v2", max_episode_steps=60)
class PutCokeCanOnKeyboardInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "coke_can"
        target_obj_name = "keyboard"
        
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [
            np.array(
                [
                    euler2quat(np.pi/2, 0, 0),  # upright can
                    [1, 0, 0, 0]
                ]
            ),
            np.array(
                [
                    euler2quat(0, 0, np.pi/2),  # lay vertically can
                    [1, 0, 0, 0]
                ]
            ),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put coke can on keyboard"