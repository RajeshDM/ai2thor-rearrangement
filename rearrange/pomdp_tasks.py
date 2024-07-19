
import numpy as np
import copy
import traceback
from allenact.utils.system import get_logger
from typing import Any, Tuple, Optional, Dict, Sequence, List, Union, cast, Set
from cospomdp_apps.thor.rearrange_trial import POMDPUnshuffleTaskEnv, POMDPProcUnshuffleTaskEnv
from rearrange.tasks import RearrangeTaskSampler, UnshuffleTask,RearrangeTaskSpec , WalkthroughTask
from rearrange.procthor_rearrange.tasks import RearrangeTaskSampler as RearrangeTaskSamplerProc
from allenact.base_abstractions.sensor import SensorSuite

class RearrangeTaskSamplerPOMDP(RearrangeTaskSampler):
    def __init__(
        self,
        run_walkthrough_phase: bool,
        run_unshuffle_phase: bool,
        stage: str,
        scenes_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
        rearrange_env_kwargs: Optional[Dict[str, Any]],
        sensors: SensorSuite,
        max_steps: Union[Dict[str, int], int],
        discrete_actions: Tuple[str, ...],
        require_done_action: bool,
        force_axis_aligned_start: bool,
        epochs: Union[int, float, str] = "default",
        seed: Optional[int] = None,
        unshuffle_runs_per_walkthrough: Optional[int] = None,
        task_spec_in_metrics: bool = False,
    ) -> None:
        super().__init__(run_walkthrough_phase=run_walkthrough_phase,
                         run_unshuffle_phase=run_unshuffle_phase,
                         stage=stage,
                         scenes_to_task_spec_dicts=scenes_to_task_spec_dicts,
                         rearrange_env_kwargs=rearrange_env_kwargs,
                         sensors=sensors,
                         max_steps=max_steps,
                         discrete_actions=discrete_actions,
                         require_done_action=require_done_action,
                         force_axis_aligned_start=force_axis_aligned_start,
                         epochs=epochs,
                         seed=seed,
                         unshuffle_runs_per_walkthrough=unshuffle_runs_per_walkthrough,
                         task_spec_in_metrics=task_spec_in_metrics)

    def create_POMDPUnshuffleTaskEnv_task(self,task_config=None):
        return POMDPUnshuffleTaskEnv(
            sensors=self.sensors,
            unshuffle_env=self.unshuffle_env,
            walkthrough_env=self.walkthrough_env,
            max_steps=self.max_steps["unshuffle"],
            discrete_actions=self.discrete_actions,
            require_done_action=self.require_done_action,
            task_spec_in_metrics=self.task_spec_in_metrics,
            controller=self.unshuffle_env.controller,
            task_config=task_config
        )

    def create_POMDPUnshuffleTaskEnv_after_walkthrough_task(self, walkthrough_task, task_config=None):
        return POMDPUnshuffleTaskEnv(
            sensors=self.sensors,
            unshuffle_env=self.unshuffle_env,
            walkthrough_env=self.walkthrough_env,
            max_steps=self.max_steps["unshuffle"],
            discrete_actions=self.discrete_actions,
            require_done_action=self.require_done_action,
            locations_visited_in_walkthrough=np.array(
                tuple(walkthrough_task.visited_positions_xzrsh)
            ),
            object_names_seen_in_walkthrough=copy.copy(
                walkthrough_task.seen_pickupable_objects
                | walkthrough_task.seen_openable_objects
            ),
            metrics_from_walkthrough=walkthrough_task.metrics(force_return=True),
            task_spec_in_metrics=self.task_spec_in_metrics,
            controller=self.unshuffle_env.controller,
            task_config=task_config
        )

    def next_task(
        self, forced_task_spec: Optional[RearrangeTaskSpec] = None, **kwargs
    ) -> Optional[UnshuffleTask]:
        """Return a fresh UnshuffleTask setup."""

        walkthrough_finished_and_should_run_unshuffle = (
            forced_task_spec is None
            and self.run_unshuffle_phase
            and self.run_walkthrough_phase
            and (
                self.was_in_exploration_phase
                or self.cur_unshuffle_runs_count < self.unshuffle_runs_per_walkthrough
            )
        )

        if (
            self.last_sampled_task is None
            or not walkthrough_finished_and_should_run_unshuffle
        ):
            self.cur_unshuffle_runs_count = 0

            try:
                if forced_task_spec is None:
                    task_spec: RearrangeTaskSpec = next(self.task_spec_iterator)
                else:
                    task_spec = forced_task_spec
            except StopIteration:
                self._last_sampled_task = None
                return self._last_sampled_task

            runtime_sample = task_spec.runtime_sample

            try:
                if self.run_unshuffle_phase:
                    self.unshuffle_env.reset(
                        task_spec=task_spec,
                        force_axis_aligned_start=self.force_axis_aligned_start,
                    )
                    self.unshuffle_env.shuffle()

                    if runtime_sample:
                        unshuffle_task_spec = self.unshuffle_env.current_task_spec
                        starting_objects = unshuffle_task_spec.runtime_data[
                            "starting_objects"
                        ]
                        openable_data = [
                            {
                                "name": o["name"],
                                "objectName": o["name"],
                                "objectId": o["objectId"],
                                "start_openness": o["openness"],
                                "target_openness": o["openness"],
                            }
                            for o in starting_objects
                            if o["isOpen"] and not o["pickupable"]
                        ]
                        starting_poses = [
                            {
                                "name": o["name"],
                                "objectName": o["name"],
                                "position": o["position"],
                                "rotation": o["rotation"],
                            }
                            for o in starting_objects
                            if o["pickupable"]
                        ]
                        task_spec = RearrangeTaskSpec(
                            scene=unshuffle_task_spec.scene,
                            agent_position=task_spec.agent_position,
                            agent_rotation=task_spec.agent_rotation,
                            openable_data=openable_data,
                            starting_poses=starting_poses,
                            target_poses=starting_poses,
                        )

                self.walkthrough_env.reset(
                    task_spec=task_spec,
                    force_axis_aligned_start=self.force_axis_aligned_start,
                )

                self.walkthrough_env_post_reset()

                if self.run_walkthrough_phase:
                    self.was_in_exploration_phase = True
                    self._last_sampled_task = WalkthroughTask(
                        sensors=self.sensors,
                        walkthrough_env=self.walkthrough_env,
                        max_steps=self.max_steps["walkthrough"],
                        discrete_actions=self.discrete_actions,
                        disable_metrics=self.run_unshuffle_phase,
                    )
                    self._last_sampled_walkthrough_task = self._last_sampled_task
                else:
                    self.cur_unshuffle_runs_count += 1
                    self._last_sampled_task = self.create_POMDPUnshuffleTaskEnv_task(kwargs.get('task_config',None))
            except Exception as e:
                if runtime_sample:
                    get_logger().error(
                        "Encountered exception while sampling a next task."
                        " As this next task was a 'runtime sample' we are"
                        " simply returning the next task."
                    )
                    get_logger().error(traceback.format_exc())
                    return self.next_task()
                else:
                    raise e
        else:
            self.cur_unshuffle_runs_count += 1
            self.was_in_exploration_phase = False

            walkthrough_task = cast(
                WalkthroughTask, self._last_sampled_walkthrough_task
            )

            if self.cur_unshuffle_runs_count != 1:
                self.unshuffle_env.reset(
                    task_spec=self.unshuffle_env.current_task_spec,
                    force_axis_aligned_start=self.force_axis_aligned_start,
                )
                self.unshuffle_env.shuffle()

            self._last_sampled_task = self.create_POMDPUnshuffleTaskEnv_after_walkthrough_task(
                walkthrough_task,kwargs.get('task_config',None)
            )

        return self._last_sampled_task

class RearrangeTaskSamplerPOMDPProc(RearrangeTaskSamplerProc):
    def __init__(
        self,
        run_walkthrough_phase: bool,
        run_unshuffle_phase: bool,
        stage: str,
        scenes_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
        rearrange_env_kwargs: Optional[Dict[str, Any]],
        sensors: SensorSuite,
        max_steps: Union[Dict[str, int], int],
        discrete_actions: Tuple[str, ...],
        require_done_action: bool,
        force_axis_aligned_start: bool,
        epochs: Union[int, float, str] = "default",
        seed: Optional[int] = None,
        unshuffle_runs_per_walkthrough: Optional[int] = None,
        task_spec_in_metrics: bool = False,
    ) -> None:
        super().__init__(run_walkthrough_phase=run_walkthrough_phase,
                         run_unshuffle_phase=run_unshuffle_phase,
                         stage=stage,
                         scenes_to_task_spec_dicts=scenes_to_task_spec_dicts,
                         rearrange_env_kwargs=rearrange_env_kwargs,
                         sensors=sensors,
                         max_steps=max_steps,
                         discrete_actions=discrete_actions,
                         require_done_action=require_done_action,
                         force_axis_aligned_start=force_axis_aligned_start,
                         epochs=epochs,
                         seed=seed,
                         unshuffle_runs_per_walkthrough=unshuffle_runs_per_walkthrough,
                         task_spec_in_metrics=task_spec_in_metrics)

    def create_POMDPUnshuffleTaskEnv_task(self,task_config=None):
        return POMDPProcUnshuffleTaskEnv(
            sensors=self.sensors,
            unshuffle_env=self.unshuffle_env,
            walkthrough_env=self.walkthrough_env,
            max_steps=self.max_steps["unshuffle"],
            discrete_actions=self.discrete_actions,
            require_done_action=self.require_done_action,
            task_spec_in_metrics=self.task_spec_in_metrics,
            controller=self.unshuffle_env.controller,
            task_config=task_config
        )

    def create_POMDPUnshuffleTaskEnv_after_walkthrough_task(self, walkthrough_task, task_config=None):
        return POMDPProcUnshuffleTaskEnv(
            sensors=self.sensors,
            unshuffle_env=self.unshuffle_env,
            walkthrough_env=self.walkthrough_env,
            max_steps=self.max_steps["unshuffle"],
            discrete_actions=self.discrete_actions,
            require_done_action=self.require_done_action,
            locations_visited_in_walkthrough=np.array(
                tuple(walkthrough_task.visited_positions_xzrsh)
            ),
            object_names_seen_in_walkthrough=copy.copy(
                walkthrough_task.seen_pickupable_objects
                | walkthrough_task.seen_openable_objects
            ),
            metrics_from_walkthrough=walkthrough_task.metrics(force_return=True),
            task_spec_in_metrics=self.task_spec_in_metrics,
            controller=self.unshuffle_env.controller,
            task_config=task_config
        )

    def next_task(
        self, forced_task_spec: Optional[RearrangeTaskSpec] = None, **kwargs
    ) -> Optional[UnshuffleTask]:
        """Return a fresh UnshuffleTask setup."""

        walkthrough_finished_and_should_run_unshuffle = (
            forced_task_spec is None
            and self.run_unshuffle_phase
            and self.run_walkthrough_phase
            and (
                self.was_in_exploration_phase
                or self.cur_unshuffle_runs_count < self.unshuffle_runs_per_walkthrough
            )
        )

        if (
            self.last_sampled_task is None
            or not walkthrough_finished_and_should_run_unshuffle
        ):
            self.cur_unshuffle_runs_count = 0

            try:
                if forced_task_spec is None:
                    task_spec: RearrangeTaskSpec = next(self.task_spec_iterator)
                else:
                    task_spec = forced_task_spec
            except StopIteration:
                self._last_sampled_task = None
                return self._last_sampled_task

            runtime_sample = task_spec.runtime_sample

            try:
                if self.run_unshuffle_phase:
                    self.unshuffle_env.reset(
                        task_spec=task_spec,
                        force_axis_aligned_start=self.force_axis_aligned_start,
                    )
                    self.unshuffle_env.shuffle()

                    if runtime_sample:
                        unshuffle_task_spec = self.unshuffle_env.current_task_spec
                        starting_objects = unshuffle_task_spec.runtime_data[
                            "starting_objects"
                        ]
                        openable_data = [
                            {
                                "name": o["name"],
                                "objectName": o["name"],
                                "objectId": o["objectId"],
                                "start_openness": o["openness"],
                                "target_openness": o["openness"],
                            }
                            for o in starting_objects
                            if o["isOpen"] and not o["pickupable"]
                        ]
                        starting_poses = [
                            {
                                "name": o["name"],
                                "objectName": o["name"],
                                "position": o["position"],
                                "rotation": o["rotation"],
                            }
                            for o in starting_objects
                            if o["pickupable"]
                        ]
                        task_spec = RearrangeTaskSpec(
                            scene=unshuffle_task_spec.scene,
                            agent_position=task_spec.agent_position,
                            agent_rotation=task_spec.agent_rotation,
                            openable_data=openable_data,
                            starting_poses=starting_poses,
                            target_poses=starting_poses,
                        )

                self.walkthrough_env.reset(
                    task_spec=task_spec,
                    force_axis_aligned_start=self.force_axis_aligned_start,
                )

                self.walkthrough_env_post_reset()

                if self.run_walkthrough_phase:
                    self.was_in_exploration_phase = True
                    self._last_sampled_task = WalkthroughTask(
                        sensors=self.sensors,
                        walkthrough_env=self.walkthrough_env,
                        max_steps=self.max_steps["walkthrough"],
                        discrete_actions=self.discrete_actions,
                        disable_metrics=self.run_unshuffle_phase,
                    )
                    self._last_sampled_walkthrough_task = self._last_sampled_task
                else:
                    self.cur_unshuffle_runs_count += 1
                    self._last_sampled_task = self.create_POMDPUnshuffleTaskEnv_task(kwargs.get('task_config',None))
            except Exception as e:
                if runtime_sample:
                    get_logger().error(
                        "Encountered exception while sampling a next task."
                        " As this next task was a 'runtime sample' we are"
                        " simply returning the next task."
                    )
                    get_logger().error(traceback.format_exc())
                    return self.next_task()
                else:
                    raise e
        else:
            self.cur_unshuffle_runs_count += 1
            self.was_in_exploration_phase = False

            walkthrough_task = cast(
                WalkthroughTask, self._last_sampled_walkthrough_task
            )

            if self.cur_unshuffle_runs_count != 1:
                self.unshuffle_env.reset(
                    task_spec=self.unshuffle_env.current_task_spec,
                    force_axis_aligned_start=self.force_axis_aligned_start,
                )
                self.unshuffle_env.shuffle()

            self._last_sampled_task = self.create_POMDPUnshuffleTaskEnv_after_walkthrough_task(
                walkthrough_task,kwargs.get('task_config',None)
            )

        return self._last_sampled_task