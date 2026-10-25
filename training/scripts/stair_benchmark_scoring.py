"""Isaac stair result scoring; independent of Isaac Sim for local verification."""

from __future__ import annotations

import math


class LandingScorer:
    """Geometry plus contact-force approximation of MuJoCo stair_landing_v1."""

    rule = "stair_landing_isaac_contact_proxy_v1"

    def __init__(self, landing_start_x: float, landing_end_x: float, center_y: float, top_z: float):
        self.start_x = landing_start_x
        self.end_x = landing_end_x
        self.center_y = center_y
        self.top_z = top_z
        self.last_contact = [-math.inf] * 4
        self.candidate_since = None
        self.fall_since = None
        self.achieved_at = None

    def update(self, t: float, base, feet, contacts, tilt_deg: float) -> tuple[str | None, dict]:
        if len(feet) != 4 or len(contacts) != 4:
            raise ValueError("Expected four foot poses and four contact flags")
        on_top = [
            self.start_x + 0.02 <= x <= self.end_x - 0.02
            and abs(y - self.center_y) <= 0.78
            and self.top_z - 0.03 <= z <= self.top_z + 0.07
            for x, y, z in feet
        ]
        for i, (inside, touching) in enumerate(zip(on_top, contacts)):
            if inside and touching:
                self.last_contact[i] = t
        recent = [t - last <= 0.5 + 1e-9 for last in self.last_contact]
        x, y, z = base
        candidate = (
            all(on_top) and all(recent)
            and self.start_x <= x <= self.end_x
            and abs(y - self.center_y) <= 0.8
            and z - self.top_z >= 0.16
            and tilt_deg <= 30.0
        )
        self.candidate_since = (t if self.candidate_since is None else self.candidate_since) if candidate else None
        fall = z < 0.16 or tilt_deg > 60.0
        self.fall_since = (t if self.fall_since is None else self.fall_since) if fall else None
        if candidate and t - self.candidate_since >= 0.2 - 1e-9:
            self.achieved_at = t
        # Match the MuJoCo order: fall, corridor exit, then landing success.
        outcome = None
        if self.fall_since is not None and t - self.fall_since >= 0.2 - 1e-9:
            outcome = "fall"
        elif abs(y - self.center_y) > 0.8:
            outcome = "left_course"
        elif self.achieved_at is not None:
            outcome = "success"
        return outcome, {
            "on_top_mask": sum(int(value) << i for i, value in enumerate(on_top)),
            "recent_contact_mask": sum(int(value) << i for i, value in enumerate(recent)),
            "candidate": candidate,
            "hold_s": t - self.candidate_since if candidate else 0.0,
        }
