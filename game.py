import pygame
import random

class PongGame:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        # Constants for the game
        self.WIDTH, self.HEIGHT = 800, 600
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 200, 10
        self.BALL_RADIUS = 10
        self.PADDLE_SPEED = 200
        self.BALL_SPEED_X = 5
        self.BALL_SPEED_Y = 5
        self.WHITE = (255, 255, 255)
        self.Game_over = False
        self.iterations = 0
        self.reward = 0

        # Create the game window
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Pong Game")

        # Initialize game objects
        self.player_paddle = pygame.Rect((self.WIDTH) // 2, self.HEIGHT - self.PADDLE_HEIGHT,
                                         self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.ball = pygame.Rect(self.WIDTH // 2, self.BALL_RADIUS,
                                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Initialize ball direction
        self.ball_dx = self.BALL_SPEED_X
        self.ball_dy = self.BALL_SPEED_Y

        # Initialize game variables (score, high score, etc.)
        self.score = 0
        self.high_score = 0

    def is_on_edge(self):
        return self.ball.y == 580 and self.ball_dy > 0

    def update_game_state(self):
        self.ball.x += self.ball_dx
        self.ball.y += self.ball_dy

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.iterations += 1

        if self.ball.colliderect(self.player_paddle) and self.ball_dy > 0:
            self.score += 1
            self.ball_dy *= -1
            if self.score > self.high_score:
                self.high_score = self.score
        if self.ball.x <= 0 or self.ball.x >= self.WIDTH:
            self.ball_dx *= -1
        if self.ball.y <= 0 and self.ball_dy < 0:
            self.ball_dy *= -1
        if self.ball.y > self.HEIGHT:
            self.Game_over = True
        self.render()
        return self.Game_over, self.score

    def play_step(self, action):
        if action[0] == 1:
            self.player_paddle.x = 0
        elif action[1] == 1:
            self.player_paddle.x = 200
        elif action[2] == 1:
            self.player_paddle.x = 400
        elif action[3] == 1:
            self.player_paddle.x = 600

        if self.ball.x in range(self.player_paddle.x , self.player_paddle.x + self.PADDLE_WIDTH):
            self.reward += 100
        else:
            self.reward -= 200

        return self.reward

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, self.WHITE, self.player_paddle)
        pygame.draw.circle(self.screen, self.WHITE,
                           self.ball.center, self.BALL_RADIUS)

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.WHITE)
        high_score_text = font.render(
            f"High Score: {self.high_score}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(high_score_text, (10, 50))
        pygame.display.flip()

    def reset(self):
        self.player_paddle = pygame.Rect((self.WIDTH) // 2, self.HEIGHT - self.PADDLE_HEIGHT,
                                         self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.ball = pygame.Rect(self.WIDTH // 2, self.BALL_RADIUS,
                                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        self.score = 0
        self.Game_over = False
        self.reward = 0
