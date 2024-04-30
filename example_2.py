"""Inference loop for the AI2-THOR object rearrangement task."""
from allenact.utils.misc_utils import NumpyJSONEncoder

#from baseline_configs.one_phase.one_phase_rgb_base import (
#    OnePhaseRGBBaseExperimentConfig,
#)
from baseline_configs.two_phase.two_phase_rgb_base import (
    TwoPhaseRGBBaseExperimentConfig,
)
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from rearrange.rearrange_trial import RearrangementThorTrial 

import os
import sys
import cProfile
from line_profiler import LineProfiler

from cospomdp_apps.thor import constants
from cospomdp_apps.thor.common import TaskArgs, make_config

sys.path.append("/home/rajesh/rajesh/pomdp/pomdp_rearrangement/experiments/thor")

import thortils as tt

from experiment_thor import (Methods,
                             OBJECT_CLASSES,
                             read_detector_params)

POUCT_ARGS = dict(max_depth=20,
                  num_sims=150,
                  discount_factor=0.95,
                  exploration_const=100,
                  show_progress=True)

LOCAL_POUCT_ARGS = POUCT_ARGS
MAX_STEPS = 100

TOPO_PLACE_SAMPLES = 20  # specific to hierarchical methods



def make_POMDP_trial(method, run_num, scene_type, scene, target, detector_models,
               corr_objects=None, max_steps=constants.MAX_STEPS,
               visualize=False, viz_res=30):
    """
    Args:
        scene: scene to search in
        target: object class to search for
        corr_objects (list): objects used as correlated objects to help
        correlations: (some kind of data structure that conveys the correlation information),
        detector_models (dict): Maps from object class to a detector models configuration used for POMDP planning;
            e.g. {"Apple": dict(fov=90, min_range=1, max_range=target_range), (target_accuracy, 0.1))}

        method_name: a string, e.g. "HIERARCHICAL_CORR_CRT"
    """
    if corr_objects is None:
        corr_objects = set()

    agent_init_inputs = ['grid_map']
    if "Greedy" in method["agent"]:
        agent_init_inputs.append('agent_pose')
    elif "Random" not in method["agent"]:
        agent_init_inputs.append('camera_pose')

    detector_specs = {
        target: detector_models[target]
    }
    corr_specs = {}
    '''
    if method["use_corr"]:
        for other in corr_objects:
            spcorr = load_correlation(scene, scene_type, target, other, method["corr_type"])
            corr_specs[(target, other)] = (spcorr.func, {'type': spcorr.corr_type}) # corr_func, corr_func_args
            detector_specs[other] = detector_models[other]
    '''

    args = TaskArgs(detectables=set(detector_specs.keys()),
                    scene=scene,
                    target=target,
                    agent_class=method["agent"],
                    task_env="ThorObjectSearch",
                    max_steps=max_steps,
                    agent_init_inputs=agent_init_inputs,
                    save_load_corr=method['use_corr'],
                    use_vision_detector=method['use_vision_detector'],
                    plot_detections=visualize,
                    agent_detector_specs=detector_specs,
                    corr_specs=corr_specs)
    config = make_config(args)

    config['thor']['fastActionEmit'] = True
    config['thor']['commit_id'] = 'a9ccb07faf771377c9ff1615bfe7e0ad01968663'

    if method["agent"] not in {"ThorObjectSearchRandomAgent",
                               "ThorObjectSearchGreedyNbvAgent"}:
        config["agent_config"]["solver"] = "pomdp_py.POUCT"
        config["agent_config"]["solver_args"] = POUCT_ARGS

    if "CompleteCosAgent" in method['agent']:
        config["agent_config"]["num_place_samples"] = TOPO_PLACE_SAMPLES
        config["agent_config"]["local_search_type"] = "3d"
        config["agent_config"]["local_search_params"] = LOCAL_POUCT_ARGS

    config["visualize"] = visualize
    config["viz_config"] = {
        'res': viz_res
    }
    trial_name = f"{scene_type.replace('_', '+')}-{scene}-{target}_{run_num:0>3}_{Methods.get_name(method)}"
    #curr_trial = RearrangementThorTrial(trial_name, config, verbose=True)
    #return curr_trial
    return trial_name, config

def prepare(scene_type):
    detector_models = read_detector_params()
    targets = OBJECT_CLASSES[scene_type]["target"]
    corr_objects = OBJECT_CLASSES[scene_type]["corr"]
    return detector_models, targets, corr_objects


def get_trial(method, scene_type, target_class, scene="FloorPlan21"):
    #valscenes = tt.ithor_scene_names(scene_type, levels=range(21, 31))
    #if scene not in valscenes:
    #    raise ValueError("Only allow validation scenes.")

    detector_models, targets, corr_objects = prepare(scene_type)
    if target_class not in targets:
        raise ValueError("{} is not a valid target class".format(target_class))

    trial_name, config = make_POMDP_trial(method, 0, scene_type, scene, target_class,
                       detector_models, corr_objects=corr_objects,
                       visualize=True)
    #profiler = LineProfiler()
    #trial.run()
    #return curr_trial
    return trial_name, config

def get_thor_controller_kwargs(config):
    thor_controller_kwargs = dict(
        scene                       =config['scene'],
        agentMode                   = config.get('agentMode', constants.AGENT_MODE),
        gridSize                   = config.get("GRID_SIZE"                    ,constants.GRID_SIZE),
        visibilityDistance         = config.get("VISIBILITY_DISTANCE"          ,constants.VISIBILITY_DISTANCE),
        snapToGrid                 = config.get("SNAP_TO_GRID"                 ,constants.SNAP_TO_GRID),
        renderDepthImage           = config.get("RENDER_DEPTH"                 ,constants.RENDER_DEPTH),
        renderInstanceSegmentation = config.get("RENDER_INSTANCE_SEGMENTATION" ,constants.RENDER_INSTANCE_SEGMENTATION),
        width                      = config.get("IMAGE_WIDTH"                  ,constants.IMAGE_WIDTH),
        height                     = config.get("IMAGE_HEIGHT"                 ,constants.IMAGE_HEIGHT),
        fieldOfView                = config.get("FOV"                          ,constants.FOV),
        #rotateStepDegrees          = config.get("H_ROTATION"                   ,constants.H_ROTATION),
        x_display                  = config.get("x_display"                    , None),
        host                       = config.get("host"                         , "127.0.0.1"),
        port                       = config.get("port"                         , 0),
        headless                   = config.get("headless"                     , False),
        fast_action_emit          = config.get("fastActionEmit"               , True),
        commit_id                 = config.get("commit_id"                    , None),
    )
    return thor_controller_kwargs


# First let's generate our task sampler that will let us run through all of the
# data points in our training set.
task_sampler_params = TwoPhaseRGBBaseExperimentConfig.stagewise_task_sampler_args(
    stage="train", process_ind=0, total_processes=1,
)
'''
two_phase_rgb_task_sampler: RearrangeTaskSampler = TwoPhaseRGBBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
    only_one_unshuffle_per_walkthrough=True,  # used for efficiency during training, should be False during inference
    epochs=1,
)
'''
room_type = 'bedroom'
goal_obj_type = 'AlarmClock'
#goal_obj_type = 'Book'
#goal_obj_type = 'CellPhone'
scene = 'FloorPlan304'

trial_name, config = get_trial(Methods.GT_HIERARCHICAL_TARGET, room_type, goal_obj_type, scene=scene)
thor_controller_kwargs = get_thor_controller_kwargs(config['thor'])

if 'thor_controller_kwargs' in task_sampler_params:
    task_sampler_params['thor_controller_kwargs'].update(thor_controller_kwargs)
else :
    task_sampler_params['thor_controller_kwargs'] = thor_controller_kwargs

two_phase_rgb_task_sampler: RearrangeTaskSampler = TwoPhaseRGBBaseExperimentConfig.make_POMDP_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
    only_one_unshuffle_per_walkthrough=True,  # used for efficiency during training, should be False during inference
    epochs=1,
)
'''
two_phase_rgb_task_sampler: RearrangeTaskSampler = TwoPhaseRGBBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
    only_one_unshuffle_per_walkthrough=True,  # used for efficiency during training, should be False during inference
    epochs=1,
)
'''

how_many_unique_datapoints = two_phase_rgb_task_sampler.total_unique
num_tasks_to_do = 1

print(
    f"Sampling {num_tasks_to_do} tasks from the Two-Phase TRAINING dataset"
    f" ({how_many_unique_datapoints} unique tasks) and taking random actions in them. "
)

for i_task in range(num_tasks_to_do):
    print(f"\nStarting task {i_task}")

    #forced_task_spec = None
    two_phase_rgb_task_sampler.task_spec_iterator.set_current_scene(scene)
    forced_task_spec = next(two_phase_rgb_task_sampler.task_spec_iterator)

    walkthrough_task = two_phase_rgb_task_sampler.next_task(forced_task_spec=forced_task_spec,
                                                            task_config=config['task_config'])
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
    unshuffle_task: UnshuffleTask = two_phase_rgb_task_sampler.next_task(task_config=config['task_config'])
    rearrange_trial = RearrangementThorTrial(trial_name, config, controller=unshuffle_task.unshuffle_env.controller,
                                            task_env = unshuffle_task, verbose=True)

    #components = rearrange_trial.setup()
    #agent = components['agent']
    _ = unshuffle_task.action_space.sample()
    rearrange_trial.run()

    while not unshuffle_task.is_done():
        break

        observations = unshuffle_task.get_observations()

        # Take a random action
        action_ind = unshuffle_task.action_space.sample()
        if unshuffle_task.num_steps_taken() % 10 == 0:
            print(
                f"Unshuffle phase (step {unshuffle_task.num_steps_taken()}):"
                f" taking action {unshuffle_task.action_names()[action_ind]}"
            )
        unshuffle_task.step(action=action_ind, agent=agent)

    print(f"Both phases complete, metrics: '{unshuffle_task.metrics()}'")

print(f"\nFinished {num_tasks_to_do} Two-Phase tasks.")
two_phase_rgb_task_sampler.close()