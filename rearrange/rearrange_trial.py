from cospomdp_apps.thor.result_types import PathResult, HistoryResult
from cospomdp_apps.thor.trial import ThorTrial
import thortils
import sys
from cospomdp.utils import cfg
cfg.DEBUG_LEVEL = 0
from sciex import Trial, Event

class RearrangementThorTrial(ThorTrial):
    RESULT_TYPES = [PathResult, HistoryResult]

    def __init__(self, name, config, controller = None, task_env = None , verbose=False):
        super().__init__(name, config, verbose=verbose)
        self.controller = controller
        self.task_env = task_env

    def _start_controller(self):
        if self.controller == None:
            self.controller = thortils.launch_controller(self.config["thor"])
        return self.controller

    def _get_task_env(self):
        return self.task_env

    def setup(self):
        # If shared resource (i.e. detector is provided, use it)
        if self.shared_resource is not None:
            vision_detector = self.shared_resource
            self.config["task_config"]["detector_config"]["vision_detector"] = vision_detector

        controller = self._start_controller()
        task_env = self._get_task_env()
        #task_env = eval(self.config["task_env"])(controller, self.config["task_config"])
        from cospomdp_apps.thor import agent as agentlib
        agent_class = eval("agentlib." + self.config["agent_class"])
        #agent_class = eval(self.config["agent_class"])
        agent_init_inputs = task_env.get_info(self.config["agent_init_inputs"])
        if agent_class.AGENT_USES_CONTROLLER:
            agent = agent_class(controller,
                                self.config['task_config'],
                                **self.config['agent_config'],
                                **agent_init_inputs)
        else:
            if 'keyboard' in self.config["agent_class"].lower():
                agent = agent_class(self.config['task_config'],
                                    agent_init_inputs['grid_map'])
            else:
                agent = agent_class(self.config["task_config"],
                                    **self.config['agent_config'],
                                    **agent_init_inputs)

        # what to return
        result = dict(controller=controller,
                      task_env=task_env,
                      agent=agent)

        if self.config.get("visualize", False):
            viz = task_env.visualizer(**self.config["viz_config"])
            img = viz.visualize(task_env, agent, step=0)
            result['viz'] = viz

            if "save_path" in self.config:
                saver = task_env.saver(self.config["save_path"], agent,
                                       **self.config["save_opts"])
                result['saver'] = saver
                saver.save_step(0, img, None, None)

        return result


    def run(self,
            logging=False,
            step_act_cb=None,
            step_act_args={},
            step_update_cb=None,
            step_update_args={}):
        """
        Functions intended for debugging purposes:
            step_act_cb: Called after the agent has determined its action
            step_update_cb: Called after the agent has executed the action and updated
                given environment observation.
        """
        # # Uncomment below to force visualization for debugging
        # self.config["visualize"] = True
        # self.config["task_config"]["detector_config"]["plot_detections"] = True

        self.print_config()
        components = self.setup()
        agent = components['agent']
        task_env = components['task_env']
        controller = components['controller']
        viz = components.get("viz", None)
        saver = components.get("saver", None)

        _actions = []

        max_steps = self.config["max_steps"]
        for i in range(1, max_steps+1):
            action = agent.act()
            if not logging:
                a_str = action.name if not action.name.startswith("Open")\
                    else "{}({})".format(action.name, action.params)
                sys.stdout.write(f"Step {i} | Action: {a_str}; ")
                sys.stdout.flush()
            if step_act_cb is not None:
                step_act_cb(task_env, agent, viz=viz, step=i, **step_act_args)

            '''
            if cfg.DEBUG_LEVEL > 0:
                _actions.append(action)
                if _rotating_too_much(_actions):
                    import pdb; pdb.set_trace()
            '''

            #observation, reward = task_env.execute(agent, action)
            step_result = task_env.step(action, agent)
            observation = step_result.pomdp_observation
            reward = step_result.pomdp_reward
            agent.update(action, observation)

            if logging:
                _step_info = task_env.get_step_info(step=i)
                self.log_event(Event("Trial %s | %s" % (self.name, _step_info)))
            else:
                sys.stdout.write("Action: {}, Observation: {}; Reward: {}\n"\
                                 .format(action, observation, reward))
                if len(task_env._history) > 1:
                    sys.stdout.write(f"Prev State : {task_env._history[-2]['state']}\n")
                    sys.stdout.write(f"Curr State : {task_env._history[-1]['state']}\n")

                sys.stdout.flush()

            if self.config.get("visualize", False):
                img = viz.visualize(task_env, agent, step=i)

                if saver is not None:
                    saver.save_step(i, img, action, observation)

            if step_update_cb is not None:
                step_update_cb(task_env, agent, viz=viz, **step_update_args)

            if task_env.done(action):
                success, msg = task_env.success(action)
                if logging:
                    self.log_event(Event("Trial %s | %s" % (self.name, msg)))
                else:
                    print(msg)
                break
        results = task_env.compute_results()
        controller.stop()
        if self.config.get("visualize", False):
            viz.on_cleanup()
            if saver is not None:
                saver.finish()

        return results

