import numpy as np
import cv2
import pickle
import os
import time
from copy import deepcopy
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility

class World:
    """1D binary world represented as a binary vector."""
    
    def __init__(self, size=20):
        """Initialize world with random binary values."""
        self.size = size
        self.state = np.random.randint(0, 2, size=size)
    
    def get_value(self, position):
        """Get the value at a specific position (with wrapping)."""
        return self.state[position % self.size]
    
    def set_value(self, position, value):
        """Set the value at a specific position (with wrapping)."""
        self.state[position % self.size] = value
    
    def visualize_opencv(self, agent_pos=None, cell_size=40):
        """Create an OpenCV image of the world state."""
        img = np.zeros((cell_size, self.size * cell_size, 3), dtype=np.uint8)
        
        for i in range(self.size):
            x_start = i * cell_size
            x_end = (i + 1) * cell_size
            
            if i == agent_pos:
                # Agent position - red
                img[:, x_start:x_end] = [0, 0, 255]
            elif self.state[i] == 1:
                # Active cell - white
                img[:, x_start:x_end] = [255, 255, 255]
            else:
                # Inactive cell - black
                img[:, x_start:x_end] = [0, 0, 0]
            
            # Draw grid lines
            cv2.line(img, (x_end-1, 0), (x_end-1, cell_size), (100, 100, 100), 1)
        
        return img


class Agent:
    """Agent that can perceive and move in the world."""
    
    def __init__(self, world, position=0, perception_radius=2):
        """Initialize agent with world, position, and perception radius."""
        self.world = world
        self.position = position
        self.perception_radius = perception_radius
        # Prediction model: (perception_tuple, action) -> next_perception
        self.prediction_model = {}
        self.last_perception = None
        self.last_action = None
        self.predicted_perception = None
    
    def perceive(self):
        """Get local perception of the world around agent's position."""
        perception = []
        for offset in range(-self.perception_radius, self.perception_radius + 1):
            pos = (self.position + offset) % self.world.size
            perception.append(self.world.get_value(pos))
        return np.array(perception)
    
    def move_left(self):
        """Move agent one step to the left."""
        self.position = (self.position - 1) % self.world.size
    
    def move_right(self):
        """Move agent one step to the right."""
        self.position = (self.position + 1) % self.world.size
    
    def stay(self):
        """Agent stays at current position."""
        pass
    
    def get_sensory_input_opencv(self, cell_size=60):
        """Get OpenCV visualization of sensory input."""
        perception = self.perceive()
        perception_size = len(perception)
        img = np.zeros((cell_size, perception_size * cell_size, 3), dtype=np.uint8)
        
        for i, val in enumerate(perception):
            x_start = i * cell_size
            x_end = (i + 1) * cell_size
            
            if i == self.perception_radius:
                # Agent's position in perception - red
                img[:, x_start:x_end] = [0, 0, 255]
            elif val == 1:
                # Active cell - white
                img[:, x_start:x_end] = [255, 255, 255]
            else:
                # Inactive cell - dark gray
                img[:, x_start:x_end] = [50, 50, 50]
            
            # Draw grid lines
            cv2.line(img, (x_end-1, 0), (x_end-1, cell_size), (100, 100, 100), 1)
        
        return img
    
    def predict(self, action):
        """Predict next sensory input based on current perception and action."""
        current_perception = tuple(self.perceive())
        key = (current_perception, action)
        
        if key in self.prediction_model:
            self.predicted_perception = np.array(self.prediction_model[key])
        else:
            # No prediction available - return None
            self.predicted_perception = None
        
        return self.predicted_perception
    
    def update_model(self):
        """Update prediction model with actual outcome after action."""
        if self.last_perception is not None and self.last_action is not None:
            current_perception = tuple(self.perceive())
            key = (self.last_perception, self.last_action)
            self.prediction_model[key] = current_perception
    
    def execute_action(self, action):
        """Execute action and update model."""
        # Store current state
        self.last_perception = tuple(self.perceive())
        self.last_action = action
        
        # Execute action
        if action == 'left':
            self.move_left()
        elif action == 'right':
            self.move_right()
        elif action == 'stay':
            self.stay()
        
        # Update model with actual result
        self.update_model()
    
    def get_prediction_opencv(self, cell_size=60):
        """Get OpenCV visualization of predicted sensory input."""
        if self.predicted_perception is None:
            # No prediction available
            perception_size = 2 * self.perception_radius + 1
            img = np.zeros((cell_size, perception_size * cell_size, 3), dtype=np.uint8)
            img[:, :] = [40, 40, 40]  # Dark gray for no prediction
            return img
        
        perception_size = len(self.predicted_perception)
        img = np.zeros((cell_size, perception_size * cell_size, 3), dtype=np.uint8)
        
        for i, val in enumerate(self.predicted_perception):
            x_start = i * cell_size
            x_end = (i + 1) * cell_size
            
            if i == self.perception_radius:
                # Agent's predicted position - orange/yellow
                img[:, x_start:x_end] = [0, 165, 255]
            elif val == 1:
                # Predicted active cell - light white
                img[:, x_start:x_end] = [200, 200, 255]
            else:
                # Predicted inactive cell - dark blue
                img[:, x_start:x_end] = [100, 50, 50]
            
            # Draw grid lines
            cv2.line(img, (x_end-1, 0), (x_end-1, cell_size), (100, 100, 100), 1)
        
        return img
    
    def save_model(self, filename='agent_model.pkl'):
        """Save prediction model to a file."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.prediction_model, f)
            return True, f"Model saved to {filename} ({len(self.prediction_model)} entries)"
        except Exception as e:
            return False, f"Error saving model: {e}"
    
    def load_model(self, filename='agent_model.pkl'):
        """Load prediction model from a file."""
        try:
            if not os.path.exists(filename):
                return False, f"File {filename} not found"
            
            with open(filename, 'rb') as f:
                self.prediction_model = pickle.load(f)
            return True, f"Model loaded from {filename} ({len(self.prediction_model)} entries)"
        except Exception as e:
            return False, f"Error loading model: {e}"
    
    def get_model_size_bytes(self):
        """Get the size of the prediction model in bytes."""
        return len(pickle.dumps(self.prediction_model))


class SimulationAgent:
    """Agent that uses simulation approach - maintains internal world model."""
    
    def __init__(self, world, position=0, perception_radius=2):
        """Initialize agent with simulation approach."""
        self.world = world
        self.position = position
        self.perception_radius = perception_radius
        
        # Internal world model - starts EMPTY (agent must explore to learn)
        # -1 represents "unknown" cells, agent fills them as it explores
        self.internal_model = np.full(world.size, -1, dtype=int)
        self.model_position = position
        
        # Immediately perceive starting position to learn initial cells
        current_perception = self.perceive()
        for i, offset in enumerate(range(-self.perception_radius, self.perception_radius + 1)):
            pos = (self.position + offset) % self.world.size
            self.internal_model[pos] = current_perception[i]
        
        self.last_perception = None
        self.last_action = None
        self.predicted_perception = None
    
    def perceive(self):
        """Get local perception of the world around agent's position."""
        perception = []
        for offset in range(-self.perception_radius, self.perception_radius + 1):
            pos = (self.position + offset) % self.world.size
            perception.append(self.world.get_value(pos))
        return np.array(perception)
    
    def move_left(self):
        """Move agent one step to the left."""
        self.position = (self.position - 1) % self.world.size
    
    def move_right(self):
        """Move agent one step to the right."""
        self.position = (self.position + 1) % self.world.size
    
    def stay(self):
        """Agent stays at current position."""
        pass
    
    def predict(self, action):
        """Predict next sensory input using internal model and action."""
        # Simulate action in internal model
        temp_position = self.model_position
        
        if action == 'left':
            temp_position = (temp_position - 1) % self.world.size
        elif action == 'right':
            temp_position = (temp_position + 1) % self.world.size
        # if action == 'stay', position doesn't change
        
        # Get perception from the predicted position in internal model
        perception = []
        has_unknown = False
        for offset in range(-self.perception_radius, self.perception_radius + 1):
            pos = (temp_position + offset) % self.world.size
            val = self.internal_model[pos]
            if val == -1:  # Unknown cell
                has_unknown = True
            perception.append(val)
        
        # Only predict if all cells are known
        if has_unknown:
            self.predicted_perception = None
        else:
            self.predicted_perception = np.array(perception)
        
        return self.predicted_perception
    
    def update_model(self):
        """Update internal model and position after action."""
        # Update model position based on last action
        if self.last_action == 'left':
            self.model_position = (self.model_position - 1) % self.world.size
        elif self.last_action == 'right':
            self.model_position = (self.model_position + 1) % self.world.size
        # if 'stay', position doesn't change
        
        # Sync internal model with actual world state
        # This represents updating the internal model with observations
        current_perception = self.perceive()
        for i, offset in enumerate(range(-self.perception_radius, self.perception_radius + 1)):
            pos = (self.position + offset) % self.world.size
            self.internal_model[pos] = current_perception[i]
    
    def execute_action(self, action):
        """Execute action and update internal model."""
        self.last_perception = tuple(self.perceive())
        self.last_action = action
        
        # Execute action in real world
        if action == 'left':
            self.move_left()
        elif action == 'right':
            self.move_right()
        elif action == 'stay':
            self.stay()
        
        # Update internal model
        self.update_model()
    
    def get_sensory_input_opencv(self, cell_size=60):
        """Get OpenCV visualization of sensory input."""
        perception = self.perceive()
        perception_size = len(perception)
        img = np.zeros((cell_size, perception_size * cell_size, 3), dtype=np.uint8)
        
        for i, val in enumerate(perception):
            x_start = i * cell_size
            x_end = (i + 1) * cell_size
            
            if i == self.perception_radius:
                img[:, x_start:x_end] = [0, 0, 255]
            elif val == 1:
                img[:, x_start:x_end] = [255, 255, 255]
            else:
                img[:, x_start:x_end] = [50, 50, 50]
            
            cv2.line(img, (x_end-1, 0), (x_end-1, cell_size), (100, 100, 100), 1)
        
        return img
    
    def get_prediction_opencv(self, cell_size=60):
        """Get OpenCV visualization of predicted sensory input."""
        if self.predicted_perception is None:
            perception_size = 2 * self.perception_radius + 1
            img = np.zeros((cell_size, perception_size * cell_size, 3), dtype=np.uint8)
            img[:, :] = [40, 40, 40]
            return img
        
        perception_size = len(self.predicted_perception)
        img = np.zeros((cell_size, perception_size * cell_size, 3), dtype=np.uint8)
        
        for i, val in enumerate(self.predicted_perception):
            x_start = i * cell_size
            x_end = (i + 1) * cell_size
            
            if i == self.perception_radius:
                img[:, x_start:x_end] = [0, 165, 255]
            elif val == 1:
                img[:, x_start:x_end] = [200, 200, 255]
            else:
                img[:, x_start:x_end] = [100, 50, 50]
            
            cv2.line(img, (x_end-1, 0), (x_end-1, cell_size), (100, 100, 100), 1)
        
        return img
    
    def get_model_size_bytes(self):
        """Get the size of the internal model in bytes."""
        return self.internal_model.nbytes


def conduct_experiment(world_size, perception_radius, num_steps=100):
    """
    Conduct experiment comparing dictionary and simulation approaches.
    Returns metrics for both methods.
    """
    world = World(size=world_size)
    
    # Initialize both agent types
    dict_agent = Agent(world, position=world_size // 2, perception_radius=perception_radius)
    sim_agent = SimulationAgent(world, position=world_size // 2, perception_radius=perception_radius)
    
    # Sync the internal model of simulation agent with real world
    sim_agent.internal_model = deepcopy(world.state)
    
    actions = ['left', 'right', 'stay']
    
    # Metrics
    dict_errors = []
    sim_errors = []
    dict_times = []
    sim_times = []
    
    for step in range(num_steps):
        action = actions[step % len(actions)]
        
        # Dictionary approach
        start_time = time.perf_counter()
        dict_agent.predict(action)
        dict_times.append(time.perf_counter() - start_time)
        
        dict_agent.execute_action(action)
        
        # Calculate error for dictionary approach
        if dict_agent.predicted_perception is not None:
            actual = dict_agent.perceive()
            error = np.mean(np.abs(dict_agent.predicted_perception - actual))
            dict_errors.append(error)
        else:
            dict_errors.append(1.0)  # Maximum error when no prediction
        
        # Simulation approach
        start_time = time.perf_counter()
        sim_agent.predict(action)
        sim_times.append(time.perf_counter() - start_time)
        
        sim_agent.execute_action(action)
        
        # Calculate error for simulation approach
        actual = sim_agent.perceive()
        error = np.mean(np.abs(sim_agent.predicted_perception - actual))
        sim_errors.append(error)
    
    # Calculate final metrics
    results = {
        'world_size': world_size,
        'perception_radius': perception_radius,
        'dict_avg_error': np.mean(dict_errors),
        'dict_max_error': np.max(dict_errors),
        'dict_avg_time': np.mean(dict_times) * 1000,  # Convert to ms
        'dict_model_size': dict_agent.get_model_size_bytes(),
        'sim_avg_error': np.mean(sim_errors),
        'sim_max_error': np.max(sim_errors),
        'sim_avg_time': np.mean(sim_times) * 1000,  # Convert to ms
        'sim_model_size': sim_agent.get_model_size_bytes(),
        'dict_predictions_made': len([e for e in dict_errors if e < 1.0]),
        'sim_predictions_made': len(sim_errors),
    }
    
    return results


def run_comparative_analysis():
    """Run comprehensive comparative analysis of both methods."""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: DICTIONARY vs SIMULATION APPROACH")
    print("="*80)
    
    # Experiment parameters
    world_sizes = [10, 20, 50, 100]
    perception_radii = [1, 2, 3, 4]
    num_steps = 200
    
    # Store results
    all_results = []
    
    print("\nRunning experiments...")
    for world_size in world_sizes:
        for radius in perception_radii:
            print(f"  World size: {world_size}, Perception radius: {radius}", end="... ")
            result = conduct_experiment(world_size, radius, num_steps)
            all_results.append(result)
            print("Done")
    
    # Print results table
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(f"{'World':<8} {'Radius':<8} {'Dict Error':<12} {'Sim Error':<12} {'Dict Time(ms)':<15} {'Sim Time(ms)':<15}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['world_size']:<8} {result['perception_radius']:<8} "
              f"{result['dict_avg_error']:<12.4f} {result['sim_avg_error']:<12.4f} "
              f"{result['dict_avg_time']:<15.6f} {result['sim_avg_time']:<15.6f}")
    
    # Print model size comparison
    print("\n" + "="*80)
    print("MODEL SIZE COMPARISON (in bytes)")
    print("="*80)
    print(f"{'World':<8} {'Radius':<8} {'Dict Size':<15} {'Sim Size':<15} {'Ratio (Dict/Sim)':<15}")
    print("-"*80)
    
    for result in all_results:
        ratio = result['dict_model_size'] / max(result['sim_model_size'], 1)
        print(f"{result['world_size']:<8} {result['perception_radius']:<8} "
              f"{result['dict_model_size']:<15} {result['sim_model_size']:<15} "
              f"{ratio:<15.2f}x")
    
    # Create visualizations
    create_comparison_plots(all_results, world_sizes, perception_radii)
    
    return all_results


def create_comparison_plots(all_results, world_sizes, perception_radii):
    """Create comparison plots for both methods."""
    
    # Organize results by varying world size (fixed radius=2)
    fixed_radius = 2
    results_by_world = [r for r in all_results if r['perception_radius'] == fixed_radius]
    
    # Organize results by varying radius (fixed world_size=50)
    fixed_world = 50
    results_by_radius = [r for r in all_results if r['world_size'] == fixed_world]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison: Dictionary vs Simulation Approach', fontsize=16, fontweight='bold')
    
    # Plot 1: Error vs World Size
    ax = axes[0, 0]
    sizes = sorted(list(set([r['world_size'] for r in results_by_world])))
    dict_errors = [next((r['dict_avg_error'] for r in results_by_world if r['world_size'] == s), None) for s in sizes]
    sim_errors = [next((r['sim_avg_error'] for r in results_by_world if r['world_size'] == s), None) for s in sizes]
    
    ax.plot(sizes, dict_errors, 'o-', label='Dictionary', linewidth=2, markersize=8)
    ax.plot(sizes, sim_errors, 's-', label='Simulation', linewidth=2, markersize=8)
    ax.set_xlabel('World Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Prediction Error', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Error vs World Size (Radius=2)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error vs Perception Radius
    ax = axes[0, 1]
    radii = sorted(list(set([r['perception_radius'] for r in results_by_radius])))
    dict_errors_r = [next((r['dict_avg_error'] for r in results_by_radius if r['perception_radius'] == rad), None) for rad in radii]
    sim_errors_r = [next((r['sim_avg_error'] for r in results_by_radius if r['perception_radius'] == rad), None) for rad in radii]
    
    ax.plot(radii, dict_errors_r, 'o-', label='Dictionary', linewidth=2, markersize=8)
    ax.plot(radii, sim_errors_r, 's-', label='Simulation', linewidth=2, markersize=8)
    ax.set_xlabel('Perception Radius', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Prediction Error', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Error vs Perception Radius (World=50)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Model Size Comparison
    ax = axes[1, 0]
    dict_sizes = [r['dict_model_size'] for r in results_by_world]
    sim_sizes = [r['sim_model_size'] for r in results_by_world]
    
    x = np.arange(len(sizes))
    width = 0.35
    ax.bar(x - width/2, dict_sizes, width, label='Dictionary', alpha=0.8)
    ax.bar(x + width/2, sim_sizes, width, label='Simulation', alpha=0.8)
    ax.set_xlabel('World Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Model Size (bytes)', fontsize=11, fontweight='bold')
    ax.set_title('Model Size vs World Size (Radius=2)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Computation Time Comparison
    ax = axes[1, 1]
    dict_times = [r['dict_avg_time'] for r in results_by_world]
    sim_times = [r['sim_avg_time'] for r in results_by_world]
    
    x = np.arange(len(sizes))
    ax.bar(x - width/2, dict_times, width, label='Dictionary', alpha=0.8)
    ax.bar(x + width/2, sim_times, width, label='Simulation', alpha=0.8)
    ax.set_xlabel('World Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Computation Time vs World Size (Radius=2)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main menu for simulation and analysis."""
    print("\n" + "="*80)
    print("1D BINARY DISCRETE ENVIRONMENT - AGENT LEARNING SIMULATION")
    print("="*80)
    print("\nChoose an option:")
    print("  1 - Run interactive simulation (OpenCV)")
    print("  2 - Run comparative analysis (Dictionary vs Simulation)")
    print("  3 - Run both")
    print("  q - Quit")
    
    choice = input("\nEnter your choice: ").strip().lower()
    
    if choice == '1':
        run_interactive_simulation()
    elif choice == '2':
        run_comparative_analysis()
    elif choice == '3':
        run_comparative_analysis()
        run_interactive_simulation()
    else:
        print("Exiting...")
        return


def run_interactive_simulation():
    """Run interactive simulation with OpenCV comparing both agents."""
    # Initialize world and both agents
    world = World(size=50)
    dict_agent = Agent(world, position=10, perception_radius=3)
    sim_agent = SimulationAgent(world, position=10, perception_radius=3)
    
    cv2.namedWindow('World State', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('DICTIONARY Agent', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('SIMULATION Agent', cv2.WINDOW_AUTOSIZE)
    
    # Position windows
    cv2.moveWindow('World State', 50, 50)
    cv2.moveWindow('DICTIONARY Agent', 50, 550)
    cv2.moveWindow('SIMULATION Agent', 900, 550)
    
    print("=" * 70)
    print("COMPARATIVE Agent-World Simulation (OpenCV)")
    print("Comparing: DICTIONARY vs SIMULATION Approach")
    print("=" * 70)
    print("Controls:")
    print("  'a' - Move Left          'd' - Move Right          's' - Stay")
    print("  'q' or ESC - Quit")
    print("=" * 70)
    
    running = True
    
    while running:
        # Create world visualization
        world_img = world.visualize_opencv(dict_agent.position, cell_size=60)
        world_img_labeled = cv2.copyMakeBorder(world_img, 30, 10, 10, 10, 
                                                cv2.BORDER_CONSTANT, value=[40, 40, 40])
        cv2.putText(world_img_labeled, f'World State (Size: {world.size}) | Agent Position: {dict_agent.position}', 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # DICTIONARY Agent visualization
        dict_sensory_img = dict_agent.get_sensory_input_opencv(cell_size=60)
        dict_pred_img = dict_agent.get_prediction_opencv(cell_size=60)
        dict_combined = cv2.vconcat([dict_sensory_img, dict_pred_img])
        dict_combined = cv2.copyMakeBorder(dict_combined, 30, 30, 10, 10,
                                            cv2.BORDER_CONSTANT, value=[40, 40, 40])
        
        dict_perception = dict_agent.perceive()
        dict_values = '  '.join([str(v) for v in dict_perception])
        cv2.putText(dict_combined, f'DICTIONARY: Sensory={dict_values}', 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if dict_agent.predicted_perception is not None:
            dict_pred_values = '  '.join([str(v) for v in dict_agent.predicted_perception])
            cv2.putText(dict_combined, f'Predicted={dict_pred_values} (Size: {len(dict_agent.prediction_model)})', 
                        (10, dict_combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        # SIMULATION Agent visualization
        sim_sensory_img = sim_agent.get_sensory_input_opencv(cell_size=60)
        sim_pred_img = sim_agent.get_prediction_opencv(cell_size=60)
        sim_combined = cv2.vconcat([sim_sensory_img, sim_pred_img])
        sim_combined = cv2.copyMakeBorder(sim_combined, 30, 30, 10, 10,
                                           cv2.BORDER_CONSTANT, value=[40, 40, 40])
        
        sim_perception = sim_agent.perceive()
        sim_values = '  '.join([str(v) for v in sim_perception])
        cv2.putText(sim_combined, f'SIMULATION: Sensory={sim_values}', 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if sim_agent.predicted_perception is not None:
            sim_pred_values = '  '.join([str(v) for v in sim_agent.predicted_perception])
            cv2.putText(sim_combined, f'Predicted={sim_pred_values} (Internal Map)', 
                        (10, sim_combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        
        # Display
        cv2.imshow('World State', world_img_labeled)
        cv2.imshow('DICTIONARY Agent', dict_combined)
        cv2.imshow('SIMULATION Agent', sim_combined)
        
        # Handle input
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('a'):
            dict_agent.predict('left')
            sim_agent.predict('left')
            print(f"\n→ Action: Move LEFT")
            print(f"   DICTIONARY | Current: {dict_agent.perceive()} | Predicted: {dict_agent.predicted_perception}")
            print(f"   SIMULATION | Current: {sim_agent.perceive()} | Predicted: {sim_agent.predicted_perception}")
            dict_agent.execute_action('left')
            sim_agent.execute_action('left')
            print(f"   After action | DICT: {dict_agent.perceive()} | SIM: {sim_agent.perceive()}")
        elif key == ord('d'):
            dict_agent.predict('right')
            sim_agent.predict('right')
            print(f"\n→ Action: Move RIGHT")
            print(f"   DICTIONARY | Current: {dict_agent.perceive()} | Predicted: {dict_agent.predicted_perception}")
            print(f"   SIMULATION | Current: {sim_agent.perceive()} | Predicted: {sim_agent.predicted_perception}")
            dict_agent.execute_action('right')
            sim_agent.execute_action('right')
            print(f"   After action | DICT: {dict_agent.perceive()} | SIM: {sim_agent.perceive()}")
        elif key == ord('s'):
            dict_agent.predict('stay')
            sim_agent.predict('stay')
            print(f"\n→ Action: Stay")
            print(f"   DICTIONARY | Current: {dict_agent.perceive()} | Predicted: {dict_agent.predicted_perception}")
            print(f"   SIMULATION | Current: {sim_agent.perceive()} | Predicted: {sim_agent.predicted_perception}")
            dict_agent.execute_action('stay')
            sim_agent.execute_action('stay')
            print(f"   After action | DICT: {dict_agent.perceive()} | SIM: {sim_agent.perceive()}")
        elif key == ord('q') or key == 27:
            running = False
            print("\nExiting simulation...")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
    finally:
        print("\nGoodbye!")