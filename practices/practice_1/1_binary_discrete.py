import numpy as np
import cv2
import pickle
import os

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


def main():
    """Main simulation loop."""
    # Initialize world and agent
    world = World(size=50)
    agent = Agent(world, position=10, perception_radius=3)
    
    # Create OpenCV windows
    cv2.namedWindow('World State', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Sensory Input', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Predicted Input', cv2.WINDOW_AUTOSIZE)
    
    # Position windows so they don't overlap
    cv2.moveWindow('World State', 100, 100)
    cv2.moveWindow('Sensory Input', 100, 600)
    cv2.moveWindow('Predicted Input', 100, 900)
    
    print("=" * 50)
    print("Agent-World Simulation (OpenCV)")
    print("=" * 50)
    print("Controls (in OpenCV window):")
    print("  'a' - Move Left")
    print("  'd' - Move Right")
    print("  's' - Stay")
    print("  'v' - Save model to file")
    print("  'l' - Load model from file")
    print("  'q' or ESC - Quit")
    print("=" * 50)
    
    running = True
    
    while running:
        # Create visualizations
        world_img = world.visualize_opencv(agent.position, cell_size=60)
        sensory_input_img = agent.get_sensory_input_opencv(cell_size=60)
        prediction_img = agent.get_prediction_opencv(cell_size=60)
        
        # Add labels to world image with agent position
        world_img_labeled = cv2.copyMakeBorder(world_img, 30, 10, 10, 10, 
                                                cv2.BORDER_CONSTANT, value=[40, 40, 40])
        cv2.putText(world_img_labeled, f'World State (Size: {world.size}) | Agent Position: {agent.position}', 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add labels to sensory input with more space below for values
        sensory_labeled = cv2.copyMakeBorder(sensory_input_img, 30, 40, 10, 10,
                                              cv2.BORDER_CONSTANT, value=[40, 40, 40])
        cv2.putText(sensory_labeled, f'Sensory Input (Radius: {agent.perception_radius})', 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add raw values below sensory input cells with better spacing
        perception = agent.perceive()
        values_text = '  '.join([str(v) for v in perception])
        cv2.putText(sensory_labeled, f'{values_text}', 
                    (10, sensory_input_img.shape[0] + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Add labels to prediction image
        prediction_labeled = cv2.copyMakeBorder(prediction_img, 30, 40, 10, 10,
                                                 cv2.BORDER_CONSTANT, value=[40, 40, 40])
        model_size = len(agent.prediction_model)
        cv2.putText(prediction_labeled, f'Predicted Input (Model size: {model_size})', 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add predicted values if available
        if agent.predicted_perception is not None:
            pred_values_text = '  '.join([str(v) for v in agent.predicted_perception])
            cv2.putText(prediction_labeled, f'{pred_values_text}', 
                        (10, prediction_img.shape[0] + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        else:
            cv2.putText(prediction_labeled, 'No prediction yet', 
                        (10, prediction_img.shape[0] + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Display images
        cv2.imshow('World State', world_img_labeled)
        cv2.imshow('Sensory Input', sensory_labeled)
        cv2.imshow('Predicted Input', prediction_labeled)
        
        # Handle keyboard input
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('a'):
            # Make prediction first
            agent.predict('left')
            print(f"\n→ Action: Move LEFT")
            print(f"   Current perception: {agent.perceive()}")
            if agent.predicted_perception is not None:
                print(f"   Predicted perception: {agent.predicted_perception}")
            else:
                print(f"   Predicted perception: [No prediction available]")
            # Execute action and update model
            agent.execute_action('left')
            print(f"   Actual perception: {agent.perceive()}")
            print(f"   Agent now at position {agent.position}")
        elif key == ord('d'):
            # Make prediction first
            agent.predict('right')
            print(f"\n→ Action: Move RIGHT")
            print(f"   Current perception: {agent.perceive()}")
            if agent.predicted_perception is not None:
                print(f"   Predicted perception: {agent.predicted_perception}")
            else:
                print(f"   Predicted perception: [No prediction available]")
            # Execute action and update model
            agent.execute_action('right')
            print(f"   Actual perception: {agent.perceive()}")
            print(f"   Agent now at position {agent.position}")
        elif key == ord('s'):
            # Make prediction first
            agent.predict('stay')
            print(f"\n→ Action: Stay")
            print(f"   Current perception: {agent.perceive()}")
            if agent.predicted_perception is not None:
                print(f"   Predicted perception: {agent.predicted_perception}")
            else:
                print(f"   Predicted perception: [No prediction available]")
            # Execute action and update model
            agent.execute_action('stay')
            print(f"   Actual perception: {agent.perceive()}")
            print(f"   Agent still at position {agent.position}")
        elif key == ord('v'):
            # Save model
            success, message = agent.save_model()
            print(f"\n💾 {message}")
        elif key == ord('l'):
            # Load model
            success, message = agent.load_model()
            print(f"\n📂 {message}")
        elif key == ord('q') or key == 27:  # 'q' or ESC
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

