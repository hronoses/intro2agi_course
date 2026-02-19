import numpy as np
import cv2

np.random.seed(42)  # For reproducibility

class World:
    """2D binary world represented as a binary matrix."""
    
    def __init__(self, width=20, height=20):
        """Initialize world with random binary values."""
        self.width = width
        self.height = height
        self.state = np.random.randint(0, 2, size=(height, width))
    
    def get_value(self, x, y):
        """Get the value at a specific position (with wrapping)."""
        return self.state[y % self.height, x % self.width]
    
    def set_value(self, x, y, value):
        """Set the value at a specific position (with wrapping)."""
        self.state[y % self.height, x % self.width] = value
    
    def visualize_opencv(self, agent_pos=None, cell_size=30):
        """Create an OpenCV image of the world state."""
        img = np.zeros((self.height * cell_size, self.width * cell_size, 3), dtype=np.uint8)
        
        # We can optimize this by drawing on the whole image at once if possible, 
        # but looping is clearer for this scale.
        for y in range(self.height):
            for x in range(self.width):
                x_start = x * cell_size
                x_end = (x + 1) * cell_size
                y_start = y * cell_size
                y_end = (y + 1) * cell_size
                
                if agent_pos is not None and x == agent_pos[0] and y == agent_pos[1]:
                    # Agent position - red
                    color = [0, 0, 255]
                elif self.state[y, x] == 1:
                    # Active cell - white
                    color = [255, 255, 255]
                else:
                    # Inactive cell - black
                    color = [0, 0, 0]
                
                # Fill cell
                img[y_start:y_end, x_start:x_end] = color
                
                # Draw grid lines (bottom and right edges)
                cv2.line(img, (x_start, y_end-1), (x_end-1, y_end-1), (50, 50, 50), 1)
                cv2.line(img, (x_end-1, y_start), (x_end-1, y_end-1), (50, 50, 50), 1)
        
        return img



class Agent:
    """Agent that can perceive and move in the 2D world."""
    
    def __init__(self, world, start_pos=[0, 0], perception_radius=1):
        """Initialize agent with world, starting position [x, y], and perception radius."""
        self.world = world
        self.position = list(start_pos) # Mutable list [x, y]
        self.perception_radius = perception_radius
    
    def perceive(self):
        """Get local perception of the world around agent's position."""
        # Create a grid for perception
        size = 2 * self.perception_radius + 1
        perception = np.zeros((size, size), dtype=int)
        
        for dy in range(-self.perception_radius, self.perception_radius + 1):
            for dx in range(-self.perception_radius, self.perception_radius + 1):
                # Calculate world coordinates with wrapping
                world_x = (self.position[0] + dx) % self.world.width
                world_y = (self.position[1] + dy) % self.world.height
                
                # Map to local perception grid coordinates
                local_y = dy + self.perception_radius
                local_x = dx + self.perception_radius
                
                perception[local_y, local_x] = self.world.get_value(world_x, world_y)
                
        return perception

    def get_sensory_input_opencv(self, cell_size=30):
        """Get OpenCV visualization of sensory input."""
        perception = self.perceive()
        size = perception.shape[0]
        img = np.zeros((size * cell_size, size * cell_size, 3), dtype=np.uint8)
        
        for y in range(size):
            for x in range(size):
                x_start = x * cell_size
                x_end = (x + 1) * cell_size
                y_start = y * cell_size
                y_end = (y + 1) * cell_size
                
                # Check if this is the agent's center position
                if x == self.perception_radius and y == self.perception_radius:
                    color = [0, 0, 255]  # Agent - Red
                elif perception[y, x] == 1:
                    color = [255, 255, 255]  # Active - White
                else:
                    color = [50, 50, 50]  # Inactive - Dark Gray
                
                img[y_start:y_end, x_start:x_end] = color
                
                # Draw grid lines
                cv2.line(img, (x_start, y_end-1), (x_end-1, y_end-1), (100, 100, 100), 1)
                cv2.line(img, (x_end-1, y_start), (x_end-1, y_end-1), (100, 100, 100), 1)
                
        return img

    def move(self, direction):
        """Move agent in a given direction."""
        dx, dy = 0, 0
        if direction == 'up':
            dy = -1
        elif direction == 'down':
            dy = 1
        elif direction == 'left':
            dx = -1
        elif direction == 'right':
            dx = 1
            
        # Update position with wrapping
        new_x = (self.position[0] + dx) % self.world.width
        new_y = (self.position[1] + dy) % self.world.height
        self.position = [new_x, new_y]

def main():
    """Main simulation loop."""
    # Initialize world and agent
    world = World(width=20, height=15)
    agent = Agent(world, start_pos=[5, 5], perception_radius=2)
    
    # Create OpenCV windows
    cv2.namedWindow('2D World State', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Sensory Input', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Sensory Input', 650, 50)  # Move to the side
    
    print("=" * 50)
    print("2D Agent-World Simulation")
    print("=" * 50)
    print("Controls:")
    print("  'w' - Move Up")
    print("  's' - Move Down")
    print("  'a' - Move Left")
    print("  'd' - Move Right")
    print("  'q' or ESC - Quit")
    print("=" * 50)
    
    running = True
    
    while running:
        # Visualize World
        world_img = world.visualize_opencv(agent.position, cell_size=30)
        
        # Visualize Sensory Input
        sensory_img = agent.get_sensory_input_opencv(cell_size=30)
        
        # Add info text to World View
        cv2.putText(world_img, f"Pos: {agent.position}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('2D World State', world_img)
        cv2.imshow('Sensory Input', sensory_img)
        
        # Handle input
        key = cv2.waitKey(50) & 0xFF
        
        if key == ord('q') or key == 27:
            running = False
        elif key == ord('w'):
            agent.move('up')
        elif key == ord('s'):
            agent.move('down')
        elif key == ord('a'):
            agent.move('left')
        elif key == ord('d'):
            agent.move('right')
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
