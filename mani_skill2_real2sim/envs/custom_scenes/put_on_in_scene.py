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
from .move_near_in_scene import MoveNearInSceneEnv

class PutOnInSceneEnv(MoveNearInSceneEnv):
    
    def reset(self, *args, **kwargs):
        self.consecutive_grasp = 0
        return super().reset(*args, **kwargs)

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            moved_correct_obj=False,
            moved_wrong_obj=False,
            is_src_obj_grasped=False,
            consecutive_grasp=False,
            src_on_target=False,
            source_intention=False,
        )

    def _set_model(self, model_ids, model_scales):
        """Set the model id and scale. If not provided, choose one randomly."""

        if model_ids is None:
            src_model_id = random_choice(self.model_ids, self._episode_rng)
            tgt_model_id = (self.model_ids.index(src_model_id) + 1) % len(
                self.model_ids
            )
            model_ids = [src_model_id, tgt_model_id]

        return super()._set_model(model_ids, model_scales)

    def evaluate(self, success_require_src_completely_on_target=True, z_flag_required_offset=0.02, **kwargs):
        source_obj_pose = self.source_obj_pose
        target_obj_pose = self.target_obj_pose

        # whether robot has intention to move the source object ever
        # determined by contact between the source object and the robot gripper
        made_contact = any(self.agent.check_contact_fingers(self.episode_source_obj))
        ee_pose = self.agent.ee_pose.p
        #ee_source_xy_dist = np.linalg.norm(ee_pose[:2] - source_obj_pose.p[:2])
        #ee_source_z_dist = abs(ee_pose[2] - source_obj_pose.p[2])
        ee_source_dist = np.linalg.norm(ee_pose - source_obj_pose.p)
        close_enough = ee_source_dist <0.05 # ee_source_xy_dist < 0.03 and ee_source_z_dist < 0.03
        source_intention_now = made_contact or close_enough

        # whether moved the correct object
        source_obj_xy_move_dist = np.linalg.norm(
            self.episode_source_obj_xyz_after_settle[:2] - self.source_obj_pose.p[:2]
        )
        other_obj_xy_move_dist = []
        for obj, obj_xyz_after_settle in zip(
            self.episode_objs, self.episode_obj_xyzs_after_settle
        ):
            if obj.name == self.episode_source_obj.name:
                continue
            other_obj_xy_move_dist.append(
                np.linalg.norm(obj_xyz_after_settle[:2] - obj.pose.p[:2])
            )
        moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (
            all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        )
        moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
            [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        )

        # whether the source object is grasped
        is_src_obj_grasped = self.agent.check_grasp(self.episode_source_obj)
        if is_src_obj_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
            self.episode_target_obj_bbox_world / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.episode_source_obj_bbox_world / 2

        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        offset = pos_src - pos_tgt
        xy_flag = (
            np.linalg.norm(offset[:2])
            <= np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003
        )
        z_flag = (offset[2] > 0) and (
            offset[2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
            <= z_flag_required_offset
        )
        src_on_target = xy_flag and z_flag

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            contacts = self._scene.get_contacts()
            flag = True
            robot_link_names = [x.name for x in self.agent.robot.get_links()]
            tgt_obj_name = self.episode_target_obj.name
            ignore_actor_names = [tgt_obj_name] + robot_link_names
            for contact in contacts:
                actor_0, actor_1 = contact.actor0, contact.actor1
                other_obj_contact_actor_name = None
                if actor_0.name == self.episode_source_obj.name:
                    other_obj_contact_actor_name = actor_1.name
                elif actor_1.name == self.episode_source_obj.name:
                    other_obj_contact_actor_name = actor_0.name
                if other_obj_contact_actor_name is not None:
                    # the object is in contact with an actor
                    contact_impulse = np.sum(
                        [point.impulse for point in contact.points], axis=0
                    )
                    if (other_obj_contact_actor_name not in ignore_actor_names) and (
                        np.linalg.norm(contact_impulse) > 1e-6
                    ):
                        # the object has contact with an actor other than the robot link or the target object, so the object is not yet put on the target object
                        flag = False
                        break
            src_on_target = src_on_target and flag

        success = src_on_target

        self.episode_stats["moved_correct_obj"] = moved_correct_obj
        self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] or is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] or consecutive_grasp
        )
        # add source intention to the episode stats, true if Ever made contact or close enough
        self.episode_stats["source_intention"] = (
            self.episode_stats["source_intention"] or source_intention_now
        )

        return dict(
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            is_src_obj_grasped=is_src_obj_grasped,
            consecutive_grasp=consecutive_grasp,
            src_on_target=src_on_target,
            episode_stats=self.episode_stats,
            success=success,
        )

    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        tgt_name = self._get_instruction_obj_name(self.episode_target_obj.name)
        return f"put {src_name} on {tgt_name}"


class PutOnBridgeInSceneEnv(PutOnInSceneEnv, CustomBridgeObjectsInSceneEnvV1):
    """Original PutOnBridgeInSceneEnv, but now with more objects in the model_db"""
    def __init__(
        self,
        source_obj_name: str = None,
        target_obj_name: str = None,
        xy_configs: List[np.ndarray] = None,
        quat_configs: List[np.ndarray] = None,
        **kwargs,
    ):
        self._source_obj_name = source_obj_name
        self._target_obj_name = target_obj_name
        self._xy_configs = xy_configs
        self._quat_configs = quat_configs
        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "widowx"
        ret["control_freq"] = 5
        ret["sim_freq"] = 500
        ret["control_mode"] = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        ret["scene_name"] = "bridge_table_1_v1"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/bridge_real_eval_1.png"
        )
        ret["rgb_overlay_cameras"] = ["3rd_view_camera"]

        return ret

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        episode_id = obj_init_options.get(
            "episode_id",
            self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
        )
        xy_config = self._xy_configs[
            (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
            // len(self._quat_configs)
        ]
        quat_config = self._quat_configs[episode_id % len(self._quat_configs)]

        options["model_ids"] = [self._source_obj_name, self._target_obj_name]
        obj_init_options["source_obj_id"] = 0
        obj_init_options["target_obj_id"] = 1
        obj_init_options["init_xys"] = xy_config
        obj_init_options["init_rot_quats"] = quat_config
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.147, 0.028],
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False

    def _load_model(self):
        self.episode_objs = []
        for (model_id, model_scale) in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            density = self.model_db[model_id].get("density", 1000)

            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            self.episode_objs.append(obj)


@register_env("PutSpoonOnTableClothInScene-v0", max_episode_steps=60)
class PutSpoonOnTableClothInScene(PutOnBridgeInSceneEnv):
    def __init__(
        self,
        source_obj_name="bridge_spoon_generated_modified",
        target_obj_name="table_cloth_generated_shorter",
        **kwargs,
    ):
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
            np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def evaluate(self, success_require_src_completely_on_target=False, **kwargs):
        # this environment allows spoons to be partially on the table cloth to be considered successful
        return super().evaluate(success_require_src_completely_on_target, **kwargs)

    def get_language_instruction(self, **kwargs):
        return "put the spoon on the towel"


@register_env("PutCarrotOnPlateInScene-v0", max_episode_steps=60)
class PutCarrotOnPlateInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
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
        return "put carrot on plate"


@register_env("StackGreenCubeOnYellowCubeInScene-v0", max_episode_steps=60)
class StackGreenCubeOnYellowCubeInScene(PutOnBridgeInSceneEnv):
    def __init__(
        self,
        source_obj_name="green_cube_3cm",
        target_obj_name="yellow_cube_3cm",
        **kwargs,
    ):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_xs = [0.05, 0.1]
        half_edge_length_ys = [0.05, 0.1]
        xy_configs = []

        for (half_edge_length_x, half_edge_length_y) in zip(
            half_edge_length_xs, half_edge_length_ys
        ):
            grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
            grid_pos = (
                grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
                + xy_center[None]
            )

            for i, grid_pos_1 in enumerate(grid_pos):
                for j, grid_pos_2 in enumerate(grid_pos):
                    if i != j:
                        xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [np.array([[1, 0, 0, 0], [1, 0, 0, 0]])]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "stack the green block on the yellow block"


@register_env("StackGreenCubeOnYellowCubeBakedTexInScene-v0", max_episode_steps=60)
class StackGreenCubeOnYellowCubeBakedTexInScene(StackGreenCubeOnYellowCubeInScene):
    DEFAULT_MODEL_JSON = "info_bridge_custom_baked_tex_v0.json"

    def __init__(self, **kwargs):
        source_obj_name = "baked_green_cube_3cm"
        target_obj_name = "baked_yellow_cube_3cm"
        super().__init__(
            source_obj_name=source_obj_name, target_obj_name=target_obj_name, **kwargs
        )
        
@register_env("StackGreenCubeOnYellowCubeBakedTexInScene-LangV1", max_episode_steps=60)
class StackGreenCubeOnYellowCubeBakedTexInSceneLangV1(StackGreenCubeOnYellowCubeBakedTexInScene):
    def get_language_instruction(self, **kwargs):
        return "pick up the green block and drop it on top of the yellow block"


@register_env("PutEggplantInBasketScene-v0", max_episode_steps=120)
class PutEggplantInBasketScene(PutOnBridgeInSceneEnv):
    def __init__(
        self,
        **kwargs,
    ):
        source_obj_name = "eggplant"
        target_obj_name = "dummy_sink_target_plane"  # invisible

        target_xy = np.array([-0.125, 0.025])
        xy_center = [-0.105, 0.206]

        half_span_x = 0.01
        half_span_y = 0.015
        num_x = 2
        num_y = 4

        grid_pos = []
        for x in np.linspace(-half_span_x, half_span_x, num_x):
            for y in np.linspace(-half_span_y, half_span_y, num_y):
                grid_pos.append(np.array([x + xy_center[0], y + xy_center[1]]))

        xy_configs = [np.stack([pos, target_xy], axis=0) for pos in grid_pos]

        quat_configs = [
            np.array([
                euler2quat(0, 0, 0, 'sxyz'),
                [1, 0, 0, 0]
            ]),
            np.array([
                euler2quat(0, 0, 1 * np.pi / 4, 'sxyz'),
                [1, 0, 0, 0]
            ]),
            np.array([
                euler2quat(0, 0, -1 * np.pi / 4, 'sxyz'),
                [1, 0, 0, 0]
            ]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            rgb_always_overlay_objects=['sink', 'dummy_sink_target_plane'],
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put eggplant into yellow basket"

    def _load_model(self):
        super()._load_model()
        self.sink_id = 'sink'
        self.sink = self._build_actor_helper(
            self.sink_id,
            self._scene,
            density=self.model_db[self.sink_id].get("density", 1000),
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction, dynamic_friction=self.obj_dynamic_friction, restitution=0.0
            ),
            root_dir=self.asset_root,
        )
        self.sink.name = self.sink_id

    def _initialize_actors(self):
        # Move the robot far away to avoid collision
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        self.sink.set_pose(sapien.Pose(
            [-0.16, 0.13, 0.88],
            [1, 0, 0, 0]
        ))
        self.sink.lock_motion()

        super()._initialize_actors()

    def evaluate(self, *args, **kwargs):
        return super().evaluate(success_require_src_completely_on_target=False, 
                                z_flag_required_offset=0.06,
                                *args, **kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = super()._setup_prepackaged_env_init_config()
        ret["robot"] = "widowx_sink_camera_setup"
        ret["scene_name"] = "bridge_table_1_v2"
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/bridge_sink.png"
        )
        return ret

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.127, 0.06],
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False # in env reset options, no need to reconfigure the environment

    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow

        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_directional_light(
            [0, 0, -1],
            [0.3, 0.3, 0.3],
            position=[0, 0, 1],
            shadow=shadow,
            scale=5,
            shadow_map_size=2048,
        )


class PutOnBridgeInSceneEnvV1(PutOnInSceneEnv, CustomBridgeObjectsInSceneEnvV1):
    """adding object distrctions, now support 4 objects in the scene"""
    def __init__(self,
        source_obj_name: str = None,
        target_obj_name: str = None,
        other_obj_names: List[str] = None,
        xy_configs: List[np.ndarray] = None,
        quat_configs: List[np.ndarray] = None,
        **kwargs,
    ):
        if other_obj_names is None:
            self._other_obj_names = []
        else:
            self._other_obj_names = other_obj_names

        self._source_obj_name = source_obj_name
        self._target_obj_name = target_obj_name
        self._xy_configs = xy_configs
        self._quat_configs = quat_configs
        super().__init__(**kwargs)


    # override the _setup_prepackaged_env_init_config method
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "widowx"
        ret["control_freq"] = 5
        ret["sim_freq"] = 500
        ret["control_mode"] = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        ret["scene_name"] = "bridge_table_1_v1"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/bridge_real_eval_1.png"
        )
        ret["rgb_overlay_cameras"] = ["3rd_view_camera"]
        
        return ret
    
    # modify to include additional objects
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        episode_id = obj_init_options.get(
            "episode_id",
            self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
        )
        xy_config = self._xy_configs[
            (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
            // len(self._quat_configs)
        ]
        quat_config = self._quat_configs[episode_id % len(self._quat_configs)]
        # make sure the source is always at 0 and the target is always at 1
        options["model_ids"] = [self._source_obj_name, self._target_obj_name] + self._other_obj_names
        obj_init_options["source_obj_id"] = 0
        obj_init_options["target_obj_id"] = 1
        obj_init_options["init_xys"] = xy_config # size: 4 x 2
        obj_init_options["init_rot_quats"] = quat_config # size: 4 x 4
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info
    
    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.147, 0.028],
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False

    def _load_model(self):
        self.episode_objs = []
        for (model_id, model_scale) in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            density = self.model_db[model_id].get("density", 1000)

            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            self.episode_objs.append(obj)
            
            
class PutonBridgeInSceneEnvV1Bgd(PutOnBridgeInSceneEnvV1):
    """with distractions in the background, and support 4 objects in the scene"""
    # override the _setup_prepackaged_env_init_config method
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "widowx"
        ret["control_freq"] = 5
        ret["sim_freq"] = 500
        ret["control_mode"] = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        ret["scene_name"] = "bridge_table_1_v1"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/bridge_real_eval_1_rabbit.JPG" # ! background image changed
        )
        ret["rgb_overlay_cameras"] = ["3rd_view_camera"]
        
        return ret
    
# * object distraction.
@register_env("PutSpoonOnTableClothInScene-distract", max_episode_steps=60)
class PutSpoonOnTableClothInSceneDistract(PutOnBridgeInSceneEnvV1):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_spoon_generated_modified"
        target_obj_name = "table_cloth_generated_shorter"
        # add addional object to the scene
        additional_obj_name = ["bridge_carrot_generated_modified", "bridge_plate_objaverse_larger"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name
        
        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                [1, 0, 0, 0],  # spoon, following the original config in their separate two-object env
                [1, 0, 0, 0],  # table cloth
                euler2quat(0, 0, np.pi),  # carrot
                [1, 0, 0, 0],  # plate
            ]), # size: 4 x 4
            np.array([
                euler2quat(0, 0, np.pi/2),  # spoon
                [1, 0, 0, 0],  # table cloth
                euler2quat(0, 0, -np.pi/2),  # carrot
                [1, 0, 0, 0],  # plate
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )   
    
    def get_language_instruction(self, **kwargs):
        return "put the spoon on the towel"
    

@register_env("PutCarrotOnPlateInScene-distract", max_episode_steps=60)
class PutCarrotOnPlateInSceneDistract(PutOnBridgeInSceneEnvV1):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "bridge_plate_objaverse_larger"
        # add addional object to the scene
        additional_obj_name = ["bridge_spoon_generated_modified", "table_cloth_generated_shorter"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name
        
        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                euler2quat(0, 0, np.pi),  # carrot
                [1, 0, 0, 0],  # plate
                [1, 0, 0, 0],  # spoon, following the original config in their separate two-object env
                [1, 0, 0, 0]  # table cloth
            ]), # size: 4 x 4
            np.array([
                euler2quat(0, 0, -np.pi/2),  # carrot
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, np.pi/2),  # spoon
                [1, 0, 0, 0]  # table cloth
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )   
    
    def get_language_instruction(self, **kwargs):
        return "put carrot on plate"
        

@register_env("PutEggplantOnPlateInScene-v1", max_episode_steps=60)
class PutEggplantOnPlateInSceneV1(PutonBridgeInSceneEnvV1Bgd):
    def __init__(self, **kwargs):
        source_obj_name = "eggplant"
        target_obj_name = "bridge_plate_objaverse_larger"
        # add addional object to the scene
        additional_obj_name = ["bridge_carrot_generated_modified", "bridge_spoon_generated_modified"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name
        
        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                euler2quat(0, 0, 0, 'sxyz'),  # eggplant
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, np.pi),  # carrot
                [1, 0, 0, 0],  # spoon, following the original config in their separate two-object env
                
            ]), # size: 4 x 4
            np.array([
                euler2quat(0, 0, 1 * np.pi / 4, 'sxyz'),  # eggplant
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, -np.pi/2),  # carrot
                euler2quat(0, 0, np.pi/2),  # spoon
                
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )   
    
    def get_language_instruction(self, **kwargs):
        return "put the eggplant on the plate"
    

@register_env("PutEggplantOnCarrotInScene-v1", max_episode_steps=60)
class PutEggplantOnCarrotInSceneV1(PutonBridgeInSceneEnvV1Bgd):
    def __init__(self, **kwargs):
        source_obj_name = "eggplant"
        target_obj_name = "bridge_carrot_generated_modified"
        # add addional object to the scene
        additional_obj_name = ["bridge_plate_objaverse_larger", "bridge_spoon_generated_modified"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name
        
        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                euler2quat(0, 0, 0, 'sxyz'),  # eggplant
                euler2quat(0, 0, np.pi),  # carrot
                [1, 0, 0, 0],  # plate
                [1, 0, 0, 0],  # spoon, following the original config in their separate two-object env
                
            ]), # size: 4 x 4
            np.array([
                euler2quat(0, 0, 1 * np.pi / 4, 'sxyz'),  # eggplant
                euler2quat(0, 0, -np.pi/2),  # carrot
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, np.pi/2),  # spoon
                
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )   
    
    def get_language_instruction(self, **kwargs):
        return "Lay the eggplant on top of the carrot"
    
    
@register_env("PutCokeCanOnPlateInScene-v1", max_episode_steps=60)
class PutCokeCanOnPlateInSceneV1(PutonBridgeInSceneEnvV1Bgd):
    def __init__(self, **kwargs):
        source_obj_name = "coke_can"
        target_obj_name = "bridge_plate_objaverse_larger"
        additional_obj_name = ["bridge_carrot_generated_modified", "bridge_spoon_generated_modified"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name
        
        
        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                euler2quat(np.pi/2, 0, 0),  # upright can
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, np.pi),  # carrot
                [1, 0, 0, 0],  # spoon, following the original config in their separate two-object env
                
            ]), # size: 4 x 4
            np.array([
                euler2quat(0, 0, np.pi/2),  # lay vertically can
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, -np.pi/2),  # carrot
                euler2quat(0, 0, np.pi/2),  # spoon
                
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )
        
    def get_language_instruction(self, **kwargs):
        return "put the coke can on the plate"

# * unseen target, seen source
@register_env("PutCarrotOnCokeCanInScene-v1", max_episode_steps=60)
class PutCarrotOnCokeCanInSceneV1(PutonBridgeInSceneEnvV1Bgd):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "coke_can"
        additional_obj_name = ["bridge_plate_objaverse_larger", "bridge_spoon_generated_modified"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name
        
        
        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                euler2quat(0, 0, np.pi),  # carrot
                euler2quat(np.pi/2, 0, 0),  # upright can
                [1, 0, 0, 0],  # plate
                [1, 0, 0, 0],  # spoon, following the original config in their separate two-object env
                
            ]), # size: 4 x 4
            np.array([
                euler2quat(0, 0, -np.pi/2),  # carrot
                euler2quat(np.pi/2, 0, 0),  # upright can
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, np.pi/2),  # spoon
                
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )
        
    def get_language_instruction(self, **kwargs):
        return "Lay the carrot on the coke can"

# * unseen target (unseen combination) seen objects
@register_env("PutCarrotOnGreenCubeInScene-v1", max_episode_steps=60)
class PutCarrotOnGreenCubeInSceneV1(PutonBridgeInSceneEnvV1Bgd):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "baked_green_cube_3cm"
        additional_obj_name = ["bridge_plate_objaverse_larger", "bridge_spoon_generated_modified"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name
        
        
        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                euler2quat(0, 0, np.pi),  # carrot
                [1, 0, 0, 0],  # cube
                [1, 0, 0, 0],  # plate
                [1, 0, 0, 0],  # spoon, following the original config in their separate two-object env
                
            ]), # size: 4 x 4
            np.array([
                euler2quat(0, 0, -np.pi/2),  # carrot
                [1, 0, 0, 0],  # cube
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, np.pi/2),  # spoon
                
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )
        
    def get_language_instruction(self, **kwargs):
        return "Place the carrot on the green cube"


# * ood action, seen objects.
@register_env("PutPlateOnGreenCubeInScene-v1", max_episode_steps=60)
class PutPlateOnGreenCubeInSceneV1(PutonBridgeInSceneEnvV1Bgd):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_plate_objaverse_larger"
        target_obj_name = "baked_green_cube_3cm"
        additional_obj_name = ["bridge_carrot_generated_modified", "bridge_spoon_generated_modified"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name
        
        
        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                [1, 0, 0, 0],  # plate
                [1, 0, 0, 0],  # cube
                euler2quat(0, 0, np.pi),  # carrot
                [1, 0, 0, 0],  # spoon, following the original config in their separate two-object env
                
            ]), # size: 4 x 4
            np.array([
                [1, 0, 0, 0],  # plate
                [1, 0, 0, 0],  # cube
                euler2quat(0, 0, -np.pi/2),  # carrot
                euler2quat(0, 0, np.pi/2),  # spoon
                
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )
        
    def get_language_instruction(self, **kwargs):
        return "Put the plate on the green cube"

        
@register_env("PutCokeCanOnPepsiCanInScene-v1", max_episode_steps=60)
class PutCokeCanOnPepsiCanInSceneV1(PutonBridgeInSceneEnvV1Bgd):
    def __init__(self, **kwargs):
        source_obj_name = "coke_can"
        target_obj_name = "pepsi_can"
        additional_obj_name = ["bridge_plate_objaverse_larger", "bridge_spoon_generated_modified"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name
        
        
        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                euler2quat(np.pi/2, 0, 0),  # upright can
                euler2quat(np.pi/2, 0, 0),  # upright can
                [1, 0, 0, 0],  # plate
                [1, 0, 0, 0],  # spoon, following the original config in their separate two-object env
                
            ]), # size: 4 x 4
            np.array([
                euler2quat(0, 0, np.pi/2),  # lay vertically can
                euler2quat(np.pi/2, 0, 0),  # upright can
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, np.pi/2),  # spoon
                
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )
        
    def get_language_instruction(self, **kwargs):
        return "put the coke can on top of the pepsi can"



# * language variation, inherit existing task and ONLY change language instruction.
@register_env("PutCarrotOnPlateInScene-LangV1", max_episode_steps=60)
class PutCarrotOnPlateInSceneLangV1(PutCarrotOnPlateInScene):
    def get_language_instruction(self, **kwargs):
        return "put rabbit's favorite vegetable on the plate"
    
@register_env("PutCarrotOnPlateInScene-LangV2", max_episode_steps=60)
class PutCarrotOnPlateInSceneLangV2(PutCarrotOnPlateInScene):
    def get_language_instruction(self, **kwargs):
        return "pick up the carrot and drop it off on the plate"

@register_env("PutCarrotOnPlateInScene-LangV3", max_episode_steps=60)
class PutCarrotOnPlateInSceneLangV3(PutCarrotOnPlateInSceneDistract):
    def get_language_instruction(self, **kwargs):
        return "put the carrot on the plate, not the towel"
    
@register_env("PutSpoonOnTableClothInScene-LangV1", max_episode_steps=60)
class PutSpoonOnTableClothInSceneLangV1(PutSpoonOnTableClothInScene):
    def get_language_instruction(self, **kwargs):
        return "put the kitchenware for eating soup on the towel"
    
@register_env("PutSpoonOnTableClothInScene-LangV3", max_episode_steps=60)
class PutSpoonOnTableClothInSceneLangV3(PutSpoonOnTableClothInScene):
    def get_language_instruction(self, **kwargs):
        return "pick up the spoon and drop it off on the towel"
    
@register_env("PutSpoonOnTableClothInScene-LangV2", max_episode_steps=60)
class PutSpoonOnTableClothInSceneLangV2(PutOnBridgeInSceneEnvV1):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_spoon_generated_modified"
        target_obj_name = "table_cloth_generated_shorter"
        # add addional object to the scene
        additional_obj_name = ["bridge_carrot_generated_modified", "sponge"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name
        
        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                [1, 0, 0, 0],  # spoon, following the original config in their separate two-object env
                [1, 0, 0, 0],  # table cloth
                euler2quat(0, 0, np.pi),  # carrot
                [1, 0, 0, 0],  # sponge
            ]), # size: 4 x 4
            np.array([
                euler2quat(0, 0, np.pi/2),  # spoon
                [1, 0, 0, 0],  # table cloth
                euler2quat(0, 0, -np.pi/2),  # carrot
                [1, 0, 0, 0],  # sponge
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )   
    
    def get_language_instruction(self, **kwargs):
        return "put the kitchenware for eating soup on the towel"
    
@register_env("PutEggplantInBasketScene-LangV1", max_episode_steps=120)
class PutEggplantInBasketSceneLangV1(PutEggplantInBasketScene):
    def get_language_instruction(self, **kwargs):
        return "put the purple object into yellow basket"
    
@register_env("PutEggplantInBasketScene-LangV2", max_episode_steps=120)
class PutEggplantInBasketSceneLangV2(PutEggplantInBasketScene):
    def get_language_instruction(self, **kwargs):
        return "put eggplant into where the dishes usually get dried"


@register_env("PutEggplantInBasketScene-LangV3", max_episode_steps=120)
class PutEggplantInBasketSceneLangV3(PutEggplantInBasketScene):
    def get_language_instruction(self, **kwargs):
        return "pick up the eggplant and drop it off into the yellow basket"    


@register_env("PutCarrotOnPlateInScene-LangV4", max_episode_steps=60)
class PutCarrotOnPlateInSceneLangV4(PutCarrotOnPlateInScene):
    def get_language_instruction(self, **kwargs):
        return "pick up the carrot and drop it elsewhere on the table, not on the plate."

    # * modify the evaluate function to reflect this abnormal language instruction, basically a nagation of the success condition.
    def evaluate(self, success_require_src_completely_on_target=True, z_flag_required_offset=0.02, **kwargs):
        source_obj_pose = self.source_obj_pose
        target_obj_pose = self.target_obj_pose

        # whether robot has intention to move the source object ever
        # determined by contact between the source object and the robot gripper
        made_contact = any(self.agent.check_contact_fingers(self.episode_source_obj))
        ee_pose = self.agent.ee_pose.p
        # ee_source_xy_dist = np.linalg.norm(ee_pose[:2] - source_obj_pose.p[:2])
        # ee_source_z_dist = abs(ee_pose[2] - source_obj_pose.p[2])
        # close_enough = ee_source_xy_dist < 0.03 and ee_source_z_dist < 0.03
        ee_source_dist = np.linalg.norm(ee_pose - source_obj_pose.p)
        close_enough = ee_source_dist <0.05 # ee_source_xy_dist < 0.03 and ee_source_z_dist < 0.03
        source_intention_now = made_contact or close_enough

        # whether moved the correct object
        source_obj_xy_move_dist = np.linalg.norm(
            self.episode_source_obj_xyz_after_settle[:2] - self.source_obj_pose.p[:2]
        )
        other_obj_xy_move_dist = []
        for obj, obj_xyz_after_settle in zip(
            self.episode_objs, self.episode_obj_xyzs_after_settle
        ):
            if obj.name == self.episode_source_obj.name:
                continue
            other_obj_xy_move_dist.append(
                np.linalg.norm(obj_xyz_after_settle[:2] - obj.pose.p[:2])
            )
        moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (
            all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        )
        moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
            [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        )

        # whether the source object is grasped
        is_src_obj_grasped = self.agent.check_grasp(self.episode_source_obj)
        if is_src_obj_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
            self.episode_target_obj_bbox_world / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.episode_source_obj_bbox_world / 2

        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        offset = pos_src - pos_tgt
        xy_flag = (
            np.linalg.norm(offset[:2])
            <= np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.01 # ! was 0.003, loosen the threshold because we are negating the success condition.
        )
        z_flag = (offset[2] > 0) and (
            offset[2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
            <= z_flag_required_offset
        )
        src_on_target = xy_flag and z_flag

        # if success_require_src_completely_on_target:
        #     # whether the source object is on the target object based on contact information
        #     contacts = self._scene.get_contacts()
        #     flag = True
        #     robot_link_names = [x.name for x in self.agent.robot.get_links()]
        #     tgt_obj_name = self.episode_target_obj.name
        #     ignore_actor_names = [tgt_obj_name] + robot_link_names
        #     for contact in contacts:
        #         actor_0, actor_1 = contact.actor0, contact.actor1
        #         other_obj_contact_actor_name = None
        #         if actor_0.name == self.episode_source_obj.name:
        #             other_obj_contact_actor_name = actor_1.name
        #         elif actor_1.name == self.episode_source_obj.name:
        #             other_obj_contact_actor_name = actor_0.name
        #         if other_obj_contact_actor_name is not None:
        #             # the object is in contact with an actor
        #             contact_impulse = np.sum(
        #                 [point.impulse for point in contact.points], axis=0
        #             )
        #             if (other_obj_contact_actor_name not in ignore_actor_names) and (
        #                 np.linalg.norm(contact_impulse) > 1e-6
        #             ):
        #                 # the object has contact with an actor other than the robot link or the target object, so the object is not yet put on the target object
        #                 flag = False
        #                 break
        #     src_on_target = src_on_target and flag
        # ! negate the success condition: moved the correct object, and it is not on the target object.
        success = (not src_on_target) and moved_correct_obj and consecutive_grasp

        self.episode_stats["moved_correct_obj"] = moved_correct_obj
        self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["src_on_target"] = not src_on_target
        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] or is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] or consecutive_grasp
        )
        # add source intention to the episode stats, true if Ever made contact or close enough
        self.episode_stats["source_intention"] = (
            self.episode_stats["source_intention"] or source_intention_now
        )

        return dict(
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            is_src_obj_grasped=is_src_obj_grasped,
            consecutive_grasp=consecutive_grasp,
            src_on_target=src_on_target,
            episode_stats=self.episode_stats,
            success=success,
        )


@register_env("PutCarrotOnPlateInScene-LangV5", max_episode_steps=60)
class PutCarrotOnPlateInSceneLangV5(PutOnBridgeInSceneEnvV1):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "bridge_plate_objaverse_larger"
        # add addional object to the scene
        additional_obj_name = ["eggplant", "rabbit"]
        model_ids = [source_obj_name, target_obj_name] + additional_obj_name

        # Define positions for all objects
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]

        # Create configurations for all objects
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    # Add positions for additional objects
                    additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                    xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                    xy_configs.append(xy_config)

        # Define rotations for all objects
        quat_configs = [
            np.array([
                euler2quat(0, 0, np.pi),  # carrot
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, 0, 'sxyz'),  # eggplant
                [1, 0, 0, 0]  # rabbit
            ]), # size: 4 x 4
            np.array([
                euler2quat(0, 0, -np.pi/2),  # carrot
                [1, 0, 0, 0],  # plate
                euler2quat(0, 0, 1 * np.pi / 4, 'sxyz'),  # eggplant
                [1, 0, 0, 0]  # rabbit
            ])
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            other_obj_names=additional_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put rabbit's favorite vegetable on the plate"
    
    
# visual variation
@register_env("PutEggplantInBasketScene-light-v1", max_episode_steps=120)
class PutEggplantInBasketSceneBrighter(PutEggplantInBasketScene):
    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow

        self._scene.set_ambient_light([1.0, 0.2, 0.2])
        self._scene.add_directional_light(
            [0, 0, -1],
            [1.6, 0.6, 0.6],
            position=[0, 0, 1],
            shadow=shadow,
            scale=5,
            shadow_map_size=2048,
        )
        
# visual variation
@register_env("PutEggplantInBasketScene-light-v2", max_episode_steps=120)
class PutEggplantInBasketSceneDarker(PutEggplantInBasketScene):
    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow

        self._scene.set_ambient_light([0.1, 0.1, 0.2])
        self._scene.add_directional_light(
            [0, 0, -1],
            [0.2, 0.2, 1.2],
            position=[0, 0, 1],
            shadow=shadow,
            scale=5,
            shadow_map_size=2048,
        )

@register_env("PutCarrotOnPlateInScene-light-v1", max_episode_steps=60)
class PutCarrotOnPlateInSceneBrighter(PutCarrotOnPlateInScene):
    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow

        self._scene.set_ambient_light([1.6, 0.6, 0.6])
        self._scene.add_directional_light(
            [0, 0, -1],
            [3.2, 1.2, 1.2],
            position=[0, 0, 1],
            shadow=shadow,
            scale=5,
            shadow_map_size=2048,
        )

@register_env("PutCarrotOnPlateInScene-light-v2", max_episode_steps=60)
class PutCarrotOnPlateInSceneDarker(PutCarrotOnPlateInScene):
    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow

        self._scene.set_ambient_light([0.2, 0.2, 0.5])
        self._scene.add_directional_light(
            [0, 0, -1],
            [0.2, 0.2, 2.2],
            position=[0, 0, 1],
            shadow=shadow,
            scale=5,
            shadow_map_size=2048,
        )
