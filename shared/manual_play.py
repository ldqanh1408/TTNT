# ============================================================
#  manual_play.py  –  Chơi tay game Chrome Dino
# ============================================================
import pygame
from shared.game_env import DinoEnv, Dinosaur


def manual_play():
    """Cho người dùng tự chơi game bằng bàn phím."""
    pygame.init()

    env  = DinoEnv(render=True)
    sprites = env.sprites
    dino = Dinosaur(sprites)
    env.reset()
    hi_score = 0

    running = True
    while running:
        running = env.handle_events()
        if not running:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_SPACE]:
            action = 1  # JUMP
        elif keys[pygame.K_DOWN]:
            action = 0  # DUCK
        else:
            action = 2  # RUN

        state, reward, done, info = env.step_single(dino, action)

        ai_info = {
            "name": "MANUAL",
            "generation": 0,
            "last_action": 0 if dino.is_ducking else (1 if dino.is_jumping else 2),
            "steps": dino.steps,
            "speed": env.game_speed,
        }
        env.render_frame(dino, hi_score, ai_info)

        if done:
            score = info["points"]
            if score > hi_score:
                hi_score = score
            print(f"\nGame over! Score: {score}  |  Best: {hi_score}")
            pygame.time.wait(1500)
            dino = Dinosaur(sprites)
            env.reset()

    env.close()
