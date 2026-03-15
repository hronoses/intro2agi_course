import cv2

from world import World
from agent import Agent
from saliency import SaliencyMap
from model import WorldModel
from action import POLICIES, PatternSeekPolicy
from patterns import load_pattern, PatternMatcher


_KEY_MOVES = {
    ord('w'): 'up',
    ord('s'): 'down',
    ord('a'): 'left',
    ord('d'): 'right',
}

_CONTROLS = """\
Controls:
  w/s/a/d   - Move Up / Down / Left / Right  [keyboard mode]
  SPACE     - Pause / Resume
  + / -     - Speed up / slow down
  e         - Toggle cell at agent position (0 <-> 1)
  z / x     - Kill / Revive cell at agent position
  t         - Toggle keyboard / auto policy
  p         - Cycle policy  (auto mode only)
  q / ESC   - Quit"""


class Simulation:
    def __init__(self, world_size=(50, 50), start_pos=None, perception_radius=9,
                 warmup=400, pattern_file=None, pattern_metric='cosine'):
        self.world = World(*world_size)
        # start in the middle of the world by default, but can be overridden
        if start_pos is None:
            start_pos = (world_size[0] // 2, world_size[1] // 2)
        self.agent = Agent(self.world, start_pos, perception_radius)
        saliency          = SaliencyMap(2 * perception_radius + 1)
        self.model        = WorldModel(*world_size, saliency=saliency)
        self.saliency     = saliency
        self._policies    = list(POLICIES)
        self._policy_idx  = 0
        self.policy       = self._policies[self._policy_idx]
        self.auto_control = False   # True = policy drives the agent

        if pattern_file is not None:
            try:
                pattern, name = load_pattern(pattern_file)
                seek_policy = PatternSeekPolicy(PatternMatcher(pattern, name, metric=pattern_metric))
                self._policies.append(seek_policy)
                # auto-select PatternSeekPolicy and enable AUTO mode
                self._policy_idx  = len(self._policies) - 1
                self.policy       = seek_policy
                self.auto_control = True
                print(f'Pattern loaded: {name}  ({pattern.shape[0]}x{pattern.shape[1]})')
                print(f'Control mode: AUTO ({self.policy.name})')
            except FileNotFoundError:
                print(f'Warning: pattern file not found: {pattern_file}')
        self.paused = False
        self.frames_per_step = 1
        self._frame_count = 0

        print(f"Running {warmup} warm-up steps...")
        for _ in range(warmup):
            self.world.step()
        print(f"Done. Starting at step {self.world.time_step}.")

    def _handle_key(self, key) -> bool:
        if key in (ord('q'), 27):
            return False
        elif key == ord(' '):
            self.paused = not self.paused
        elif key in (ord('+'), ord('=')):
            self.frames_per_step = max(1, self.frames_per_step - 1)
        elif key == ord('-'):
            self.frames_per_step = min(30, self.frames_per_step + 1)
        elif key == ord('t'):
            self.auto_control = not self.auto_control
            mode = f'AUTO ({self.policy.name})' if self.auto_control else 'KEYBOARD'
            print(f'Control mode: {mode}')
        elif key == ord('p') and self.auto_control:
            self._policy_idx = (self._policy_idx + 1) % len(self._policies)
            self.policy = self._policies[self._policy_idx]
            print(f'Policy: {self.policy.name}')
        elif key in _KEY_MOVES and not self.auto_control:
            self.agent.move(_KEY_MOVES[key])
        elif key == ord('e'):
            self.agent.toggle_cell()
        elif key == ord('z'):
            self.agent.set_cell(0)
        elif key == ord('x'):
            self.agent.set_cell(1)
        return True

    def run(self):
        print(_CONTROLS)
        cv2.namedWindow('2D World State', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('2D World State', 600, 600)
        cv2.moveWindow('2D World State', 0, 50)
        cv2.namedWindow('Sensory Input',  cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sensory Input', 400, 400)
        cv2.moveWindow('Sensory Input', 620, 50)
        cv2.namedWindow('World Model', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('World Model', 600, 600)
        cv2.moveWindow('World Model', 0, 670)
        if self.saliency is not None:
            cv2.namedWindow('Saliency Map', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Saliency Map', 400, 220)
            cv2.moveWindow('Saliency Map', 620, 470)

        while True:
            if not self.paused:
                self._frame_count += 1
                if self._frame_count >= self.frames_per_step:
                    self.world.step()
                    self._frame_count = 0

            world_img   = self.world.visualize_opencv(self.agent.position, cell_size=30)
            sensory_img = self.agent.get_sensory_input_opencv(cell_size=30)
            self.model.update(self.agent.perceive(), self.agent.position,
                               self.agent.perception_radius, self.world.time_step)
            model_img = self.model.visualize_opencv(self.agent.position, cell_size=30)

            # Policy step (every world step when not paused)
            if self.auto_control and not self.paused:
                direction = self.policy(self.agent, self.model)
                if direction:
                    self.agent.move(direction)

            if self.saliency is not None:
                saliency_img = self.saliency.visualize(cell_size=20)

            status = "PAUSED" if self.paused else f"step {self.world.time_step}"
            mode_label = f'AUTO:{self.policy.name}' if self.auto_control else 'KEYS'
            cv2.putText(world_img, f"Pos: {self.agent.position}  |  {status}  |  [{mode_label}]",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(world_img, f"Speed: 1 step / {self.frames_per_step} frames",
                        (10, world_img.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1)

            cv2.imshow('2D World State', world_img)
            cv2.imshow('Sensory Input',  sensory_img)
            cv2.imshow('World Model', model_img)
            if self.saliency is not None:
                cv2.imshow('Saliency Map', saliency_img)

            if not self._handle_key(cv2.waitKey(10) & 0xFF):
                break

        cv2.destroyAllWindows()
