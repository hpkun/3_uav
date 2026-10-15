from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import EnvironmentConfig, RenderConfig

if TYPE_CHECKING:
    from .env import UAVCombatEnv


TEAM_COLORS = ((231, 67, 64), (62, 70, 157))


class CombatRenderer:
    def __init__(
        self, environment: EnvironmentConfig, config: RenderConfig, project_root: Path
    ) -> None:
        self.environment = environment
        self.config = config
        source = Path(config.icon_source)
        if not source.is_absolute():
            source = project_root / source
        sprite = self._load_source_sprite(source)
        self.sprites = [self._tint_sprite(sprite, color) for color in TEAM_COLORS]
        self.font = ImageFont.load_default()

    @staticmethod
    def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components: list[list[tuple[int, int]]] = []
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(x, y)]
                visited[y, x] = True
                component: list[tuple[int, int]] = []
                while stack:
                    cx, cy = stack.pop()
                    component.append((cx, cy))
                    for nx, ny in (
                        (cx - 1, cy),
                        (cx + 1, cy),
                        (cx, cy - 1),
                        (cx, cy + 1),
                    ):
                        if (
                            0 <= nx < width
                            and 0 <= ny < height
                            and mask[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((nx, ny))
                if len(component) >= 3:
                    components.append(component)
        return components

    def _load_source_sprite(self, source: Path) -> Image.Image:
        if not source.exists():
            return self._fallback_sprite()
        image = Image.open(source).convert("RGB")
        pixels = np.asarray(image)
        mask = (
            (pixels[..., 0] > 145)
            & (pixels[..., 0] > pixels[..., 1] * 1.35)
            & (pixels[..., 0] > pixels[..., 2] * 1.25)
        )
        components = self._connected_components(mask)
        if not components:
            return self._fallback_sprite()
        component = max(components, key=len)
        xs = [point[0] for point in component]
        ys = [point[1] for point in component]
        pad = 4
        box = (
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            min(image.width, max(xs) + pad + 1),
            min(image.height, max(ys) + pad + 1),
        )
        crop = np.asarray(image.crop(box)).copy()
        redness = np.clip(
            crop[..., 0].astype(np.int16)
            - np.maximum(crop[..., 1], crop[..., 2]).astype(np.int16),
            0,
            255,
        ).astype(np.uint8)
        rgba = np.zeros((*crop.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = np.clip(redness.astype(np.float32) * 2.8, 0, 255).astype(
            np.uint8
        )
        sprite = Image.fromarray(rgba, mode="RGBA")
        if sprite.getbbox():
            sprite = sprite.crop(sprite.getbbox())
        width = 42
        height = max(18, round(sprite.height * width / max(sprite.width, 1)))
        return sprite.resize((width, height), Image.Resampling.LANCZOS)

    @staticmethod
    def _fallback_sprite() -> Image.Image:
        image = Image.new("RGBA", (48, 28), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.polygon(
            [
                (45, 14),
                (22, 10),
                (12, 1),
                (8, 2),
                (14, 12),
                (2, 11),
                (2, 17),
                (14, 16),
                (8, 26),
                (12, 27),
                (22, 18),
            ],
            fill=(255, 255, 255, 255),
        )
        return image

    @staticmethod
    def _tint_sprite(sprite: Image.Image, color: tuple[int, int, int]) -> Image.Image:
        tinted = Image.new("RGBA", sprite.size, (*color, 0))
        tinted.putalpha(sprite.getchannel("A"))
        return tinted

    def _world_to_pixel(self, point: np.ndarray) -> tuple[int, int]:
        half = self.environment.map_size / 2.0
        usable_w = self.config.width - 2 * self.config.margin
        usable_h = self.config.height - 2 * self.config.margin
        x = self.config.margin + (float(point[0]) + half) / self.environment.map_size * usable_w
        y = self.config.height - self.config.margin - (
            (float(point[1]) + half) / self.environment.map_size * usable_h
        )
        return int(round(x)), int(round(y))

    def render(self, env: "UAVCombatEnv") -> np.ndarray:
        cfg = self.config
        canvas = Image.new("RGBA", (cfg.width, cfg.height), (250, 250, 248, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            (cfg.margin, cfg.margin, cfg.width - cfg.margin, cfg.height - cfg.margin),
            outline=(92, 92, 92, 255),
            width=2,
        )
        if cfg.show_trails:
            for team in range(2):
                for agent in range(env.N_AGENTS):
                    points = [
                        self._world_to_pixel(point) for point in env._trails[team][agent]
                    ]
                    if len(points) >= 2:
                        draw.line(points, fill=(*TEAM_COLORS[team], 90), width=2)

        for team in range(2):
            for agent in range(env.N_AGENTS):
                center = self._world_to_pixel(env.positions[team, agent])
                if not env.alive[team, agent]:
                    color = (*TEAM_COLORS[team], 180)
                    draw.line(
                        (center[0] - 7, center[1] - 7, center[0] + 7, center[1] + 7),
                        fill=color,
                        width=2,
                    )
                    draw.line(
                        (center[0] - 7, center[1] + 7, center[0] + 7, center[1] - 7),
                        fill=color,
                        width=2,
                    )
                    continue
                sprite = self.sprites[team].rotate(
                    math.degrees(env.headings[team, agent]),
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                )
                canvas.alpha_composite(
                    sprite,
                    (center[0] - sprite.width // 2, center[1] - sprite.height // 2),
                )
                draw.text(
                    (center[0] + 11, center[1] + 8),
                    str(agent + 1),
                    fill=(*TEAM_COLORS[team], 255),
                    font=self.font,
                )

        status = (
            f"step={env.step_count:03d}  "
            f"RED {int(env.alive[0].sum())}/{env.N_AGENTS}  "
            f"BLUE {int(env.alive[1].sum())}/{env.N_AGENTS}"
        )
        draw.rectangle(
            (cfg.margin, 10, cfg.width - cfg.margin, 32), fill=(255, 255, 255, 220)
        )
        draw.text(
            (cfg.margin + 8, 15), status, fill=(40, 40, 40, 255), font=self.font
        )
        return np.asarray(canvas.convert("RGB"), dtype=np.uint8)
