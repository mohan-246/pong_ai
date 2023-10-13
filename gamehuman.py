import pygame
import sys

class PongGame:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Constants for the game
        self.WIDTH, self.HEIGHT = 800, 600
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 10
        self.BALL_RADIUS = 10
        self.PADDLE_SPEED = 50
        self.BALL_SPEED_X = 1
        self.BALL_SPEED_Y = 1
        self.WHITE = (255, 255, 255)
        self.Game_over = False
        
        # Create the game window
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Pong Game")

        # Initialize game objects
        self.player_paddle = pygame.Rect((self.WIDTH - self.PADDLE_WIDTH) // 2, self.HEIGHT - self.PADDLE_HEIGHT,
                                         self.PADDLE_WIDTH , self.PADDLE_HEIGHT)
        self.ball = pygame.Rect(self.WIDTH // 2 - self.BALL_RADIUS, self.HEIGHT // 2 - self.BALL_RADIUS,
                                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        # Initialize ball direction
        self.ball_dx = self.BALL_SPEED_X
        self.ball_dy = self.BALL_SPEED_Y

        # Initialize game variables (score, high score, etc.)
        self.score = 0
        self.high_score = 0

    def handle_input(self):
        # Handle user input (e.g., move the paddle)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    # Move paddle up
                    self.player_paddle.x -= self.PADDLE_SPEED
                elif event.key == pygame.K_RIGHT:
                    # Move paddle down
                    self.player_paddle.x += self.PADDLE_SPEED

    def update_game_state(self):
        # Update game state (e.g., move ball, check collisions, update score)
        self.ball.x += self.ball_dx
        self.ball.y += self.ball_dy
        reward = 0
        
        if self.ball.colliderect(self.player_paddle) and self.ball_dy > 0:
            self.score += 1
            reward = 10
            self.ball_dy *= -1  # Reverse ball direction
            if self.score > self.high_score:
                self.high_score = self.score
        if self.ball.x <= 0 or self.ball.x >= self.WIDTH :
            self.ball_dx *= -1  # Bounce off top and bottom walls
        if self.ball.y <= 0 and self.ball_dy < 0:
            print("D")
            self.ball_dy *= -1
        if self.ball.y > self.HEIGHT:
            self.Game_over = True  # Game over when ball goes out of bounds
        return [reward , self.Game_over , self.score]

    def render(self):
        # Render game objects on the screen
        self.screen.fill((0, 0, 0))  # Clear the screen
        # Draw paddles, ball, score, and any other game elements
        pygame.draw.rect(self.screen, self.WHITE, self.player_paddle)
        pygame.draw.circle(self.screen, self.WHITE, self.ball.center, self.BALL_RADIUS)
        # Display the score and high score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.WHITE)
        high_score_text = font.render(f"High Score: {self.high_score}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(high_score_text, (10, 50))
        # Update the display
        pygame.display.flip()

    def run(self):
        # Main game loop
        while not self.Game_over:
            #print(self.ball.x , self.ball.y , self.ball_dy)
            self.handle_input()  # Handle user input
            self.update_game_state()  # Update game state
            self.render()  # Render the game

if __name__ == '__main__':
    game = PongGame()
    game.run()
