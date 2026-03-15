import numpy as np
import cv2
import time


class World:
    """1D continuous world with walls at specific positions."""
    
    def __init__(self, length=50.0, num_walls=20):
        """Initialize world with walls at continuous positions.
        
        Args:
            length: Total length of the world
            num_walls: Number of transitions to generate
        """
        self.length = length
        
        # Generate random wall positions and sort them
        # Walls represent transitions between black and white
        positions = np.sort(np.random.uniform(0, length, num_walls))
        
        # Wall types: 1 = black->white, -1 = white->black
        # Alternate types to create regions
        types = []
        current_type = 1  # Start with black->white
        for _ in range(len(positions)):
            types.append(current_type)
            current_type *= -1  # Alternate
        
        self.walls = list(zip(positions, types))
    
    def get_color_at(self, position):
        """Get the color (0=black, 1=white) at a continuous position."""
        position = position % self.length
        
        # Count how many black->white transitions are to the left
        transitions = 0
        for wall_pos, wall_type in self.walls:
            if wall_pos < position:
                transitions += wall_type
        
        # If odd number of black->white transitions, we're in white region
        return 1 if transitions > 0 else 0
    
    def get_walls_in_range(self, center, radius):
        """Get walls within a range around center position.
        
        Returns:
            List of (relative_position, wall_type) tuples
        """
        center = center % self.length
        walls_in_range = []
        
        for wall_pos, wall_type in self.walls:
            # Handle wrapping
            dist = wall_pos - center
            if dist > self.length / 2:
                dist -= self.length
            elif dist < -self.length / 2:
                dist += self.length
            
            if abs(dist) <= radius:
                walls_in_range.append((dist, wall_type))
        
        return sorted(walls_in_range, key=lambda x: x[0])
    
    def visualize_opencv(self, agent_pos=None, pixels_per_unit=20):
        """Create an OpenCV image of the world state."""
        width = int(self.length * pixels_per_unit)
        height = 60
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sample colors across the world
        for x in range(width):
            world_pos = x / pixels_per_unit
            color_val = self.get_color_at(world_pos)
            color = 255 if color_val == 1 else 0
            img[:, x] = [color, color, color]
        
        # Draw walls as thin lines
        for wall_pos, wall_type in self.walls:
            x = int((wall_pos % self.length) * pixels_per_unit)
            if 0 <= x < width:
                color = (0, 255, 255) if wall_type == 1 else (255, 255, 0)  # Cyan or yellow
                cv2.line(img, (x, 0), (x, height), color, 2)
        
        # Draw agent position
        if agent_pos is not None:
            agent_x = int((agent_pos % self.length) * pixels_per_unit)
            if 0 <= agent_x < width:
                cv2.circle(img, (agent_x, height // 2), 8, (0, 0, 255), -1)
        
        return img


class Agent:
    """Agent with continuous physics-based movement."""
    
    def __init__(self, world, position=0.0, perception_radius=5.0):
        """Initialize agent with physics properties."""
        self.world = world
        self.position = position  # Continuous position
        self.velocity = 0.0  # Current velocity
        self.perception_radius = perception_radius
        
        # Physics parameters
        self.mass = 1.0
        self.friction_coeff = 5.0  # Friction coefficient
        self.kick_force = 50.0  # Force applied per kick
    
    def apply_force(self, force):
        """Apply an impulse force to the agent (delta function)."""
        # F = ma, but impulse changes velocity directly
        self.velocity += force / self.mass
    
    def update(self, dt):
        """Update agent physics with time step dt."""
        # Apply friction (opposing velocity)
        friction_force = -self.friction_coeff * self.velocity
        acceleration = friction_force / self.mass
        
        # Update velocity and position
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Wrap position
        self.position = self.position % self.world.length
        
        # Stop if velocity is very small
        if abs(self.velocity) < 0.01:
            self.velocity = 0.0
    
    def kick_left(self):
        """Apply leftward force."""
        self.apply_force(-self.kick_force)
    
    def kick_right(self):
        """Apply rightward force."""
        self.apply_force(self.kick_force)
    
    def perceive(self):
        """Get walls within perception radius.
        
        Returns:
            List of (relative_position, wall_type) tuples
        """
        return self.world.get_walls_in_range(self.position, self.perception_radius)
    
    def get_sensory_input_opencv(self, height=60):
        """Get OpenCV visualization of sensory input."""
        width = int(2 * self.perception_radius * 40)  # 40 pixels per unit
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw perception window background (sample colors)
        pixels_per_unit = width / (2 * self.perception_radius)
        for x in range(width):
            world_offset = (x / pixels_per_unit) - self.perception_radius
            world_pos = self.position + world_offset
            color_val = self.world.get_color_at(world_pos)
            color = 255 if color_val == 1 else 50
            img[:, x] = [color, color, color]
        
        # Draw walls within perception
        walls = self.perceive()
        for rel_pos, wall_type in walls:
            # Convert relative position to pixel
            x = int((rel_pos + self.perception_radius) * pixels_per_unit)
            if 0 <= x < width:
                color = (0, 255, 255) if wall_type == 1 else (255, 255, 0)
                cv2.line(img, (x, 0), (x, height), color, 3)
        
        # Draw agent at center
        center_x = width // 2
        cv2.circle(img, (center_x, height // 2), 8, (0, 0, 255), -1)
        
        return img


def main():
    """Main simulation loop with continuous physics."""
    # Initialize world and agent
    world = World(length=50.0, num_walls=30)
    agent = Agent(world, position=25.0, perception_radius=5.0)
    
    # Create OpenCV windows
    cv2.namedWindow('World State', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Sensory Input', cv2.WINDOW_AUTOSIZE)
    
    # Position windows so they don't overlap
    cv2.moveWindow('World State', 100, 100)
    cv2.moveWindow('Sensory Input', 100, 300)
    
    print("=" * 50)
    print("Continuous Agent-World Simulation (OpenCV)")
    print("=" * 50)
    print("Controls (in OpenCV window):")
    print("  'a' - Kick Left")
    print("  'd' - Kick Right")
    print("  'q' or ESC - Quit")
    print("=" * 50)
    print("Physics enabled: agent has velocity and friction")
    print("=" * 50)
    
    # Simulation parameters
    dt = 0.01  # Time step
    last_time = time.time()
    
    running = True
    
    while running:
        # Calculate actual elapsed time
        current_time = time.time()
        elapsed = current_time - last_time
        
        # Run physics updates to catch up to real time
        accumulated_time = elapsed
        while accumulated_time >= dt:
            agent.update(dt)
            accumulated_time -= dt
        
        last_time = current_time
        
        # Create visualizations
        world_img = world.visualize_opencv(agent.position, pixels_per_unit=20)
        sensory_input_img = agent.get_sensory_input_opencv(height=60)
        
        # Add labels to world image with agent info
        world_img_labeled = cv2.copyMakeBorder(world_img, 30, 30, 10, 10, 
                                                cv2.BORDER_CONSTANT, value=[40, 40, 40])
        cv2.putText(world_img_labeled, 
                    f'World (Length: {world.length:.1f}) | Agent Pos: {agent.position:.2f} | Vel: {agent.velocity:.2f}', 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Legend
        cv2.putText(world_img_labeled, 
                    'Cyan: Black->White | Yellow: White->Black | Red: Agent', 
                    (10, world_img.shape[0] + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add labels to sensory input with wall info
        walls = agent.perceive()
        sensory_labeled = cv2.copyMakeBorder(sensory_input_img, 30, 50, 10, 10,
                                              cv2.BORDER_CONSTANT, value=[40, 40, 40])
        cv2.putText(sensory_labeled, 
                    f'Sensory Input (Radius: {agent.perception_radius:.1f}) | Walls detected: {len(walls)}', 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display wall information
        if walls:
            wall_info = ', '.join([f'{pos:.2f}{"↑" if wtype==1 else "↓"}' for pos, wtype in walls[:5]])
            if len(walls) > 5:
                wall_info += '...'
            cv2.putText(sensory_labeled, 
                        f'Walls: {wall_info}', 
                        (10, sensory_input_img.shape[0] + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Display images
        cv2.imshow('World State', world_img_labeled)
        cv2.imshow('Sensory Input', sensory_labeled)
        
        # Handle keyboard input (non-blocking)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            agent.kick_left()
            print(f"← Kick LEFT | Pos: {agent.position:.2f} | Vel: {agent.velocity:.2f}")
        elif key == ord('d'):
            agent.kick_right()
            print(f"→ Kick RIGHT | Pos: {agent.position:.2f} | Vel: {agent.velocity:.2f}")
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

