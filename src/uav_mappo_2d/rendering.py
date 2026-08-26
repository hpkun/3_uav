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
        self.project_root = project_root
        source = Path(config.icon_source)
        if not source.is_absolute():
            source = project_root / source
        self.base_sprite = self._load_source_sprite(source)
        self.sprites = [self._tint_sprite(self.base_sprite, color) for color in TEAM_COLORS]
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
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
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
        red_mask = (
            (pixels[..., 0] > 145)
            & (pixels[..., 0] > pixels[..., 1] * 1.35)
            & (pixels[..., 0] > pixels[..., 2] * 1.25)
        )
        components = self._connected_components(red_mask)
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
        alpha = np.clip(redness.astype(np.float32) * 2.8, 0, 255).astype(np.uint8)
        rgba = np.zeros((*crop.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = alpha
        sprite = Image.fromarray(rgba, mode="RGBA")
        bbox = sprite.getbbox()
        if bbox:
            sprite = sprite.crop(bbox)
        target_width = 42
        target_height = max(18, round(sprite.height * target_width / max(sprite.width, 1)))
        return sprite.resize((target_width, target_height), Image.Resampling.LANCZOS)

    @staticmethod
    def _fallback_sprite() -> Image.Image:
        image = Image.new("RGBA", (48, 28), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.polygon(
            [(45, 14), (22, 10), (12, 1), (8, 2), (14, 12), (2, 11), (2, 17),
             (14, 16), (8, 26), (12, 27), (22, 18)],
            fill=(255, 255, 255, 255),
        )
        return image

    @staticmethod
    def _tint_sprite(sprite: Image.Image, color: tuple[int, int, int]) -> Image.Image:
        alpha = sprite.getchannel("A")
        tinted = Image.new("RGBA", sprite.size, (*color, 0))
        tinted.putalpha(alpha)
        return tinted

    def _world_to_pixel(self, point: np.ndarray) -> tuple[int, int]:
        cfg = self.config
        half = self.environment.map_size / 2.0
        usable_w = cfg.width - 2 * cfg.margin
        usable_h = cfg.height - 2 * cfg.margin
        x = cfg.margin + (float(point[0]) + half) / (2.0 * half) * usable_w
        y = cfg.height - cfg.margin - (float(point[1]) + half) / (2.0 * half) * usable_h
        return int(round(x)), int(round(y))

    def _range_to_pixels(self, distance: float) -> float:
        usable = min(
            self.config.width - 2 * self.config.margin,
            self.config.height - 2 * self.config.margin,
        )
        return distance / self.environment.map_size * usable

    def _draw_sector(
        self,
        canvas: Image.Image,
        center: tuple[int, int],
        heading: float,
        radius_world: float,
        fov_degrees: float,
        color: tuple[int, int, int, int],
    ) -> None:
        radius = self._range_to_pixels(radius_world)
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        points = [center]
        for offset in np.linspace(-fov_degrees / 2.0, fov_degrees / 2.0, 25):
            angle = heading + math.radians(offset)
            points.append(
                (
                    center[0] + radius * math.cos(angle),
                    center[1] - radius * math.sin(angle),
                )
            )
        draw.polygon(points, fill=color)
        canvas.alpha_composite(overlay)

    def render(self, env: "UAVCombatEnv") -> np.ndarray:
        cfg = self.config
        canvas = Image.new("RGBA", (cfg.width, cfg.height), (250, 250, 248, 255))
        draw = ImageDraw.Draw(canvas)
        left, top = cfg.margin, cfg.margin
        right, bottom = cfg.width - cfg.margin, cfg.height - cfg.margin
        draw.rectangle((left, top, right, bottom), outline=(92, 92, 92, 255), width=2)

        if cfg.show_center:
            half = self.environment.center_size / 2.0
            p1 = self._world_to_pixel(np.array([-half, half]))
            p2 = self._world_to_pixel(np.array([half, -half]))
            self._dashed_rectangle(draw, (*p1, *p2), fill=(202, 166, 35, 180), width=2)

        if cfg.show_trails:
            for team in range(2):
                color = (*TEAM_COLORS[team], 90)
                for agent in range(env.n_agents):
                    points = [self._world_to_pixel(point) for point in env._trails[team][agent]]
                    if len(points) >= 2:
                        draw.line(points, fill=color, width=2)

        for team in range(2):
            for agent in range(env.n_agents):
                if not env.alive[team, agent]:
                    continue
                center = self._world_to_pixel(env.positions[team, agent])
                if cfg.show_radar:
                    self._draw_sector(
                        canvas,
                        center,
                        env.headings[team, agent],
                        self.environment.radar_range,
                        self.environment.radar_fov_degrees,
                        (*TEAM_COLORS[team], 20),
                    )
                if cfg.show_weapon:
                    self._draw_sector(
                        canvas,
                        center,
                        env.headings[team, agent],
                        self.environment.weapon_range,
                        self.environment.weapon_fov_degrees,
                        (*TEAM_COLORS[team], 38),
                    )

        draw = ImageDraw.Draw(canvas)
        for team in range(2):
            for agent in range(env.n_agents):
                if not env.alive[team, agent]:
                    continue
                center = self._world_to_pixel(env.positions[team, agent])
                sprite = self.sprites[team].rotate(
                    math.degrees(env.headings[team, agent]),
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                )
                canvas.alpha_composite(
                    sprite, (center[0] - sprite.width // 2, center[1] - sprite.height // 2)
                )
                draw.text(
                    (center[0] + 11, center[1] + 8),
                    str(agent + 1),
                    fill=(*TEAM_COLORS[team], 255),
                    font=self.font,
                )

                target = int(env.lock_target[team, agent])
                if target >= 0 and env.alive[1 - team, target]:
                    target_px = self._world_to_pixel(env.positions[1 - team, target])
                    progress = env.lock_progress[team, agent] / max(
                        self.environment.lock_seconds, 1e-6
                    )
                    lock_color = (235, 159, 45, int(90 + 165 * min(progress, 1.0)))
                    draw.line((center, target_px), fill=lock_color, width=2)

        status = (
            f"t={env.elapsed_time:05.1f}s   "
            f"RED {int(env.alive[0].sum())}/{env.n_agents}   "
            f"BLUE {int(env.alive[1].sum())}/{env.n_agents}"
        )
        draw.rectangle((cfg.margin, 10, cfg.width - cfg.margin, 32), fill=(255, 255, 255, 220))
        draw.text((cfg.margin + 8, 15), status, fill=(40, 40, 40, 255), font=self.font)
        return np.asarray(canvas.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _dashed_rectangle(
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        *,
        fill: tuple[int, int, int, int],
        width: int,
        dash: int = 8,
    ) -> None:
        x1, y1, x2, y2 = box
        for x in range(x1, x2, 2 * dash):
            draw.line((x, y1, min(x + dash, x2), y1), fill=fill, width=width)
            draw.line((x, y2, min(x + dash, x2), y2), fill=fill, width=width)
        for y in range(y1, y2, 2 * dash):
            draw.line((x1, y, x1, min(y + dash, y2)), fill=fill, width=width)
            draw.line((x2, y, x2, min(y + dash, y2)), fill=fill, width=width)
