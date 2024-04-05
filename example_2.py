"""Inference loop for the AI2-THOR object rearrangement task."""
from allenact.utils.misc_utils import NumpyJSONEncoder

from baseline_configs.one_phase.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)
from baseline_configs.two_phase.two_phase_rgb_base import (
    TwoPhaseRGBBaseExperimentConfig,
)
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask

# First let's generate our task sampler that will let us run through all of the
# data points in our training set.

task_sampler_params = TwoPhaseRGBBaseExperimentConfig.stagewise_task_sampler_args(
    stage="train", process_ind=0, total_processes=1,
)
two_phase_rgb_task_sampler: RearrangeTaskSampler = TwoPhaseRGBBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
    only_one_unshuffle_per_walkthrough=True,  # used for efficiency during training, should be False during inference
    epochs=1,
)

how_many_unique_datapoints = two_phase_rgb_task_sampler.total_unique
num_tasks_to_do = 5

print(
    f"Sampling {num_tasks_to_do} tasks from the Two-Phase TRAINING dataset"
    f" ({how_many_unique_datapoints} unique tasks) and taking random actions in them. "
)

for i_task in range(num_tasks_to_do):
    print(f"\nStarting task {i_task}")

    walkthrough_task = two_phase_rgb_task_sampler.next_task()
    print(
        f"Sampled task is from the "
        f" '{two_phase_rgb_task_sampler.current_task_spec.stage}' stage and has"
        f" unique id '{two_phase_rgb_task_sampler.current_task_spec.unique_id}'"
    )

    assert isinstance(walkthrough_task, WalkthroughTask)

    # Take random actions in the walkthrough task until the task is done
    while not walkthrough_task.is_done():
        observations = walkthrough_task.get_observations()

        # Take a random action
        action_ind = walkthrough_task.action_space.sample()
        if walkthrough_task.num_steps_taken() % 10 == 0:
            print(
                f"Walkthrough phase (step {walkthrough_task.num_steps_taken()}):"
                f" taking action {walkthrough_task.action_names()[action_ind]}"
            )
        walkthrough_task.step(action=action_ind)

    # Get the next task from the task sampler, this will be the task
    # of rearranging the environment so that it is back in the same configuration as
    # it was during the walkthrough task.
    unshuffle_task: UnshuffleTask = two_phase_rgb_task_sampler.next_task()

    while not unshuffle_task.is_done():
        observations = unshuffle_task.get_observations()

        # Take a random action
        action_ind = unshuffle_task.action_space.sample()
        if unshuffle_task.num_steps_taken() % 10 == 0:
            print(
                f"Unshuffle phase (step {unshuffle_task.num_steps_taken()}):"
                f" taking action {unshuffle_task.action_names()[action_ind]}"
            )
        unshuffle_task.step(action=action_ind)

    print(f"Both phases complete, metrics: '{unshuffle_task.metrics()}'")

print(f"\nFinished {num_tasks_to_do} Two-Phase tasks.")
two_phase_rgb_task_sampler.close()