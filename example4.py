import prior
from ai2thor.controller import Controller

from baseline_configs.two_phase.procthor.two_phase_rgb_base_procthor import TwoPhaseRGBBaseExperimentConfigProc
from rearrange.procthor_rearrange.tasks import RearrangeTaskSampler, UnshuffleTask 

#dataset = prior.load_dataset("procthor-10k")
#print (dataset)

task_sampler_params = TwoPhaseRGBBaseExperimentConfigProc.stagewise_task_sampler_args(
    stage="train", process_ind=0, total_processes=1,
)

two_phase_rgb_task_sampler: RearrangeTaskSampler = TwoPhaseRGBBaseExperimentConfigProc.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
    only_one_unshuffle_per_walkthrough=True,  # used for efficiency during training, should be False during inference
    epochs=1,
)
num_tasks_to_do = 3

for i_task in range(num_tasks_to_do):
    print(f"\nStarting task {i_task}")
    walkthrough_task = two_phase_rgb_task_sampler.next_task()
    print(
        f"Sampled task is from the "
        f" '{two_phase_rgb_task_sampler.current_task_spec.stage}' stage and has"
        f" unique id '{two_phase_rgb_task_sampler.current_task_spec.unique_id}'"
    )

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