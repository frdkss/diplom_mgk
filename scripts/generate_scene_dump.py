from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


SCENE_ROOT_CANDIDATES = [
    "/MGK_DIPLOM_CORRECT",
]
ROBOT_CANDIDATES = [
    "/RobotnikSummitXL",
]
ROOM_BORDERS_PATH = "/RoomBorders"
MAZE_PATH = "/Maze"
FLOORS_PATH = "/Floors"
SCORE_ROOT_PATH = "/Score"
SCORE_DUMMY_PATH = "/Score/Dummy"
SCORE_CUBOID_PATH = "/Score/Cuboid"
OUTPUT_PATH = Path("scene_dump.json")

REQUIRED_POINT_COUNT = 3
TARGET_NAME_ORDER = [
    "Start_point_dummy",
    "Point_dummy[0]",
    "Point_dummy[1]",
    "Point_dummy[2]",
    "End_point_dummy",
    "Fake_point_dummy",
]
VISUAL_TARGET_NAME_ORDER = [
    "Start_point_cuboid_score",
    "Point_cuboid_score[0]",
    "Point_cuboid_score[1]",
    "Point_cuboid_score[2]",
    "End_point_cuboid_score",
    "Fake_point_cuboid_score",
]

SIM_OBJECT_SHAPE_TYPE = 0
SIM_OBJECT_JOINT_TYPE = 1
SIM_OBJECT_DUMMY_TYPE = 4

EXCLUDED_LEAF_NAMES = {"visible"}
EXCLUDED_LEAF_SUBSTRINGS = ("ctrlpt", "script")
WALL_NAME_TOKENS = ("wall", "border")


@dataclass(slots=True)
class Pose:
    x: float
    y: float
    z: float
    alpha: float
    beta: float
    gamma: float


@dataclass(slots=True)
class SceneObject:
    name: str
    alias: str
    handle: int
    type: str
    pose: Pose


class SceneDumpGenerator:
    def __init__(self) -> None:
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self._path_cache: dict[int, str] = {}
        self._alias_cache: dict[int, str] = {}

    # ---------- low-level helpers ----------
    def obj_exists(self, path: str) -> bool:
        try:
            self.sim.getObject(path)
            return True
        except Exception:
            return False

    def first_existing_path(self, candidates: Iterable[str]) -> str | None:
        for path in candidates:
            if self.obj_exists(path):
                return path
        return None

    def get_handle(self, path: str) -> int:
        return int(self.sim.getObject(path))

    def get_parent(self, handle: int) -> int:
        return int(self.sim.getObjectParent(handle))

    def get_leaf_alias(self, handle: int) -> str:
        if handle in self._alias_cache:
            return self._alias_cache[handle]
        try:
            alias = str(self.sim.getObjectAlias(handle))
        except Exception:
            alias = ""
        self._alias_cache[handle] = alias
        return alias

    def get_full_path(self, handle: int) -> str:
        if handle in self._path_cache:
            return self._path_cache[handle]

        parts: list[str] = []
        cur = int(handle)
        visited: set[int] = set()

        while cur not in (-1, 0) and cur not in visited:
            visited.add(cur)
            alias = self.get_leaf_alias(cur)
            if alias:
                parts.append(alias)
            try:
                cur = self.get_parent(cur)
            except Exception:
                break

        full_path = "/" + "/".join(reversed(parts)) if parts else self.get_leaf_alias(handle)
        self._path_cache[handle] = full_path
        return full_path

    def get_type_id(self, handle: int) -> int:
        return int(self.sim.getObjectType(handle))

    def get_type_name(self, type_id: int) -> str:
        mapping = {
            SIM_OBJECT_SHAPE_TYPE: "shape",
            SIM_OBJECT_JOINT_TYPE: "joint",
            SIM_OBJECT_DUMMY_TYPE: "dummy",
        }
        return mapping.get(type_id, f"type_{type_id}")

    def get_pose(self, handle: int) -> Pose:
        pos = self.sim.getObjectPosition(handle, -1)
        ori = self.sim.getObjectOrientation(handle, -1)
        return Pose(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            alpha=float(ori[0]),
            beta=float(ori[1]),
            gamma=float(ori[2]),
        )

    @staticmethod
    def get_leaf_from_path(path: str) -> str:
        return str(path).rstrip("/").split("/")[-1]

    def make_scene_object(self, handle: int) -> SceneObject:
        type_id = self.get_type_id(handle)
        full_path = self.get_full_path(handle)
        alias = self.get_leaf_alias(handle) or self.get_leaf_from_path(full_path)
        return SceneObject(
            name=full_path,
            alias=alias,
            handle=int(handle),
            type=self.get_type_name(type_id),
            pose=self.get_pose(handle),
        )

    def children_in_tree(self, root_handle: int, object_type: int) -> list[int]:
        handles = list(self.sim.getObjectsInTree(root_handle, object_type, 0))
        return [int(h) for h in handles if int(h) != int(root_handle)]

    def sorted_handles(self, handles: Iterable[int]) -> list[int]:
        return sorted(handles, key=lambda h: (self.get_full_path(int(h)), int(h)))

    @staticmethod
    def _round6(value: float) -> float:
        return round(float(value), 6)

    def make_pose_fingerprint(self, obj: SceneObject) -> tuple[Any, ...]:
        p = obj.pose
        return (
            obj.alias,
            self._round6(p.x),
            self._round6(p.y),
            self._round6(p.z),
            self._round6(p.alpha),
            self._round6(p.beta),
            self._round6(p.gamma),
        )

    @staticmethod
    def _normalized(obj: SceneObject, root_path: str, new_leaf: str) -> SceneObject:
        return replace(obj, name=f"{root_path}/{new_leaf}", alias=new_leaf)

    def _collect_named_points(
        self,
        objects: list[SceneObject],
        prefix: str,
        root_path: str,
        expected_count: int,
    ) -> tuple[list[SceneObject], list[str]]:
        extras: list[str] = []
        exact: dict[str, SceneObject] = {}
        unindexed: list[SceneObject] = []

        for obj in objects:
            leaf = self.get_leaf_from_path(obj.name)
            alias = obj.alias
            if leaf.startswith("ctrlPt") or alias.startswith("ctrlPt"):
                continue
            if leaf.lower() in EXCLUDED_LEAF_NAMES or alias.lower() in EXCLUDED_LEAF_NAMES:
                continue

            if leaf.startswith(f"{prefix}[") and leaf.endswith("]"):
                exact[leaf] = self._normalized(obj, root_path, leaf)
                continue

            if alias == prefix or leaf == prefix or leaf.startswith(prefix):
                unindexed.append(obj)
                continue

            extras.append(obj.name)

        unindexed = sorted(unindexed, key=lambda o: (o.name, o.handle, o.pose.y, o.pose.x))
        for idx, obj in enumerate(unindexed):
            key = f"{prefix}[{idx}]"
            if key not in exact:
                exact[key] = self._normalized(obj, root_path, key)

        ordered: list[SceneObject] = []
        missing: list[str] = []
        for idx in range(expected_count):
            key = f"{prefix}[{idx}]"
            if key in exact:
                ordered.append(exact[key])
            else:
                missing.append(key)

        return ordered, missing + extras

    # ---------- filters ----------
    def is_excluded_shape(self, obj: SceneObject) -> bool:
        leaf_lower = obj.alias.lower()
        path_lower = obj.name.lower()

        if leaf_lower in EXCLUDED_LEAF_NAMES:
            return True
        if any(token in leaf_lower for token in EXCLUDED_LEAF_SUBSTRINGS):
            return True
        if "/score/" in path_lower:
            return True
        if "/floors/" in path_lower:
            return True
        return False

    def is_wall_shape(self, obj: SceneObject) -> bool:
        leaf_lower = obj.alias.lower()
        path_lower = obj.name.lower()

        if self.is_excluded_shape(obj):
            return False
        if not (
            path_lower.startswith(ROOM_BORDERS_PATH.lower() + "/")
            or path_lower.startswith(MAZE_PATH.lower() + "/")
        ):
            return False
        return any(token in leaf_lower for token in WALL_NAME_TOKENS)

    # ---------- semantic collectors ----------
    def collect_robot(self) -> SceneObject:
        robot_path = self.first_existing_path(ROBOT_CANDIDATES)
        if robot_path is None:
            raise RuntimeError(f"Robot not found. Tried: {', '.join(ROBOT_CANDIDATES)}")
        return self.make_scene_object(self.get_handle(robot_path))

    def infer_scene_root(self, robot: SceneObject, walls: list[SceneObject], floors: list[SceneObject]) -> str | None:
        candidate = self.first_existing_path(SCENE_ROOT_CANDIDATES)
        if candidate:
            return candidate

        prefixes: dict[str, int] = {}
        for obj in [robot, *walls, *floors]:
            parts = [p for p in obj.name.split("/") if p]
            if parts:
                prefixes[parts[0]] = prefixes.get(parts[0], 0) + 1
        if not prefixes:
            return None
        best = max(prefixes.items(), key=lambda kv: kv[1])[0]
        return f"/{best}"

    def collect_walls(self) -> tuple[list[SceneObject], dict[str, Any]]:
        roots: list[int] = []
        for path in (ROOM_BORDERS_PATH, MAZE_PATH):
            if self.obj_exists(path):
                roots.append(self.get_handle(path))

        if not roots:
            raise RuntimeError(
                f"Wall roots not found. Expected at least one of: {ROOM_BORDERS_PATH}, {MAZE_PATH}"
            )

        raw_shapes: list[SceneObject] = []
        skipped_non_walls: list[str] = []

        for root in roots:
            for handle in self.sorted_handles(self.children_in_tree(root, SIM_OBJECT_SHAPE_TYPE)):
                obj = self.make_scene_object(handle)
                if self.is_wall_shape(obj):
                    raw_shapes.append(obj)
                else:
                    skipped_non_walls.append(obj.name)

        deduped: list[SceneObject] = []
        seen_fingerprints: set[tuple[Any, ...]] = set()
        duplicate_paths: list[str] = []

        for obj in raw_shapes:
            fp = self.make_pose_fingerprint(obj)
            if fp in seen_fingerprints:
                duplicate_paths.append(obj.name)
                continue
            seen_fingerprints.add(fp)
            deduped.append(obj)

        if not deduped:
            raise RuntimeError("No wall shapes found under /RoomBorders or /Maze after filtering.")

        diagnostics = {
            "raw_wall_candidates": len(raw_shapes),
            "final_walls": len(deduped),
            "duplicate_walls_removed": len(duplicate_paths),
            "duplicate_wall_paths": duplicate_paths,
            "skipped_non_wall_shapes": skipped_non_walls,
        }
        return deduped, diagnostics

    def collect_floors(self) -> tuple[list[SceneObject], dict[str, Any]]:
        if not self.obj_exists(FLOORS_PATH):
            return [], {"raw_floor_shapes": 0, "final_floors": 0, "skipped_floor_shapes": []}

        floor_root = self.get_handle(FLOORS_PATH)
        raw: list[SceneObject] = []
        skipped: list[str] = []
        for handle in self.sorted_handles(self.children_in_tree(floor_root, SIM_OBJECT_SHAPE_TYPE)):
            obj = self.make_scene_object(handle)
            leaf_lower = obj.alias.lower()
            if leaf_lower in EXCLUDED_LEAF_NAMES or any(token in leaf_lower for token in EXCLUDED_LEAF_SUBSTRINGS):
                skipped.append(obj.name)
                continue
            raw.append(obj)

        normalized: list[SceneObject] = []
        for idx, obj in enumerate(raw):
            normalized.append(self._normalized(obj, FLOORS_PATH, f"Floor[{idx}]"))

        return normalized, {
            "raw_floor_shapes": len(raw),
            "final_floors": len(normalized),
            "skipped_floor_shapes": skipped,
        }

    def collect_target_dummies(self) -> tuple[list[SceneObject], dict[str, Any]]:
        if not self.obj_exists(SCORE_DUMMY_PATH):
            raise RuntimeError(f"Target root not found: {SCORE_DUMMY_PATH}")

        root = self.get_handle(SCORE_DUMMY_PATH)
        handles = self.sorted_handles(self.children_in_tree(root, SIM_OBJECT_DUMMY_TYPE))
        all_objects = [self.make_scene_object(h) for h in handles]

        start = None
        end = None
        fake = None
        point_candidates: list[SceneObject] = []
        extras: list[str] = []

        for obj in all_objects:
            leaf = self.get_leaf_from_path(obj.name)
            alias = obj.alias
            if leaf == "Start_point_dummy" or alias == "Start_point_dummy":
                start = self._normalized(obj, SCORE_DUMMY_PATH, "Start_point_dummy")
            elif leaf == "End_point_dummy" or alias == "End_point_dummy":
                end = self._normalized(obj, SCORE_DUMMY_PATH, "End_point_dummy")
            elif leaf == "Fake_point_dummy" or alias == "Fake_point_dummy":
                fake = self._normalized(obj, SCORE_DUMMY_PATH, "Fake_point_dummy")
            elif leaf.startswith("Point_dummy") or alias == "Point_dummy":
                point_candidates.append(obj)
            else:
                extras.append(obj.name)

        points, point_missing_or_extra = self._collect_named_points(
            point_candidates,
            prefix="Point_dummy",
            root_path=SCORE_DUMMY_PATH,
            expected_count=REQUIRED_POINT_COUNT,
        )
        extras.extend([x for x in point_missing_or_extra if not x.startswith("Point_dummy[")])
        missing = []
        if start is None:
            missing.append("Start_point_dummy")
        missing.extend([f"Point_dummy[{i}]" for i in range(REQUIRED_POINT_COUNT) if i >= len(points)])
        if end is None:
            missing.append("End_point_dummy")
        if fake is None:
            missing.append("Fake_point_dummy")

        # stricter missing detection based on actual normalized keys
        actual_names = {obj.alias for obj in points}
        for i in range(REQUIRED_POINT_COUNT):
            key = f"Point_dummy[{i}]"
            if key not in actual_names and key not in missing:
                missing.append(key)

        if missing:
            raise RuntimeError("Missing required target dummies: " + ", ".join(sorted(missing)))

        ordered = [start, *points, end, fake]
        diagnostics = {
            "missing_required_targets": [],
            "extra_dummies": extras,
        }
        return ordered, diagnostics

    def collect_visual_targets(self) -> tuple[list[SceneObject], dict[str, Any]]:
        if not self.obj_exists(SCORE_CUBOID_PATH):
            return [], {
                "missing_visual_targets": VISUAL_TARGET_NAME_ORDER.copy(),
                "extra_visual_shapes": [],
            }

        root = self.get_handle(SCORE_CUBOID_PATH)
        handles = self.sorted_handles(self.children_in_tree(root, SIM_OBJECT_SHAPE_TYPE))
        all_objects = [self.make_scene_object(h) for h in handles]

        start = None
        end = None
        fake = None
        point_candidates: list[SceneObject] = []
        extras: list[str] = []

        for obj in all_objects:
            leaf = self.get_leaf_from_path(obj.name)
            alias = obj.alias
            lower_leaf = leaf.lower()
            lower_alias = alias.lower()
            if lower_leaf == "visible" or lower_alias == "visible":
                continue
            if leaf == "Start_point_cuboid_score" or alias == "Start_point_cuboid_score":
                start = self._normalized(obj, SCORE_CUBOID_PATH, "Start_point_cuboid_score")
            elif leaf == "End_point_cuboid_score" or alias == "End_point_cuboid_score":
                end = self._normalized(obj, SCORE_CUBOID_PATH, "End_point_cuboid_score")
            elif leaf == "Fake_point_cuboid_score" or alias == "Fake_point_cuboid_score":
                fake = self._normalized(obj, SCORE_CUBOID_PATH, "Fake_point_cuboid_score")
            elif leaf.startswith("Point_cuboid_score") or alias == "Point_cuboid_score":
                point_candidates.append(obj)
            else:
                extras.append(obj.name)

        points, point_missing_or_extra = self._collect_named_points(
            point_candidates,
            prefix="Point_cuboid_score",
            root_path=SCORE_CUBOID_PATH,
            expected_count=REQUIRED_POINT_COUNT,
        )
        extras.extend([x for x in point_missing_or_extra if not x.startswith("Point_cuboid_score[")])

        missing: list[str] = []
        if start is None:
            missing.append("Start_point_cuboid_score")
        actual_names = {obj.alias for obj in points}
        for i in range(REQUIRED_POINT_COUNT):
            key = f"Point_cuboid_score[{i}]"
            if key not in actual_names:
                missing.append(key)
        if end is None:
            missing.append("End_point_cuboid_score")
        if fake is None:
            missing.append("Fake_point_cuboid_score")

        ordered = [obj for obj in [start, *points, end, fake] if obj is not None]
        diagnostics = {
            "missing_visual_targets": missing,
            "extra_visual_shapes": extras,
        }
        return ordered, diagnostics

    def build_summary(self, targets: list[SceneObject], visual_targets: list[SceneObject]) -> dict[str, Any]:
        target_names = [obj.alias for obj in targets]
        visual_names = [obj.alias for obj in visual_targets]
        return {
            "start_target": next((name for name in target_names if name == "Start_point_dummy"), None),
            "point_targets": [name for name in target_names if name.startswith("Point_dummy[")],
            "end_target": next((name for name in target_names if name == "End_point_dummy"), None),
            "fake_target": next((name for name in target_names if name == "Fake_point_dummy"), None),
            "visual_start": next((name for name in visual_names if name == "Start_point_cuboid_score"), None),
            "visual_points": [name for name in visual_names if name.startswith("Point_cuboid_score[")],
            "visual_end": next((name for name in visual_names if name == "End_point_cuboid_score"), None),
            "visual_fake": next((name for name in visual_names if name == "Fake_point_cuboid_score"), None),
        }

    def validate_scene_roots(self, scene_root: str | None) -> dict[str, Any]:
        return {
            "scene_root": scene_root,
            "has_room_borders": self.obj_exists(ROOM_BORDERS_PATH),
            "has_maze": self.obj_exists(MAZE_PATH),
            "has_floors": self.obj_exists(FLOORS_PATH),
            "has_score": self.obj_exists(SCORE_ROOT_PATH),
            "has_score_dummy": self.obj_exists(SCORE_DUMMY_PATH),
            "has_score_cuboid": self.obj_exists(SCORE_CUBOID_PATH),
        }

    # ---------- export ----------
    def generate(self) -> dict[str, Any]:
        robot = self.collect_robot()
        walls, wall_diag = self.collect_walls()
        floors, floor_diag = self.collect_floors()
        targets, target_diag = self.collect_target_dummies()
        visual_targets, visual_diag = self.collect_visual_targets()
        scene_root = self.infer_scene_root(robot, walls, floors)
        roots_info = self.validate_scene_roots(scene_root)

        dump = {
            "meta": {
                "format_version": 4,
                "scene_root": scene_root,
                "source": "CoppeliaSim via ZMQ Remote API",
                "counts": {
                    "walls": len(walls),
                    "floors": len(floors),
                    "targets": len(targets),
                    "visual_targets": len(visual_targets),
                },
                "roots": roots_info,
                "summary": self.build_summary(targets, visual_targets),
                "diagnostics": {
                    "walls": wall_diag,
                    "floors": floor_diag,
                    "targets": target_diag,
                    "visual_targets": visual_diag,
                },
            },
            "robot": asdict(robot),
            "targets": [asdict(x) for x in targets],
            "visual_targets": [asdict(x) for x in visual_targets],
            "walls": [asdict(x) for x in walls],
            "floors": [asdict(x) for x in floors],
        }
        return dump

    def save(self, output_path: Path = OUTPUT_PATH) -> Path:
        dump = self.generate()
        output_path = Path(output_path)
        output_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[OK] scene_dump.json written to: {output_path.resolve()}")
        print("[INFO] Counts:")
        print(json.dumps(dump["meta"]["counts"], ensure_ascii=False, indent=2))

        visual_missing = dump["meta"]["diagnostics"]["visual_targets"]["missing_visual_targets"]
        if visual_missing:
            print("[WARN] Missing visual target cuboids:", ", ".join(visual_missing))

        dup_removed = dump["meta"]["diagnostics"]["walls"]["duplicate_walls_removed"]
        if dup_removed:
            print(f"[WARN] Duplicate walls removed: {dup_removed}")

        return output_path


if __name__ == "__main__":
    SceneDumpGenerator().save(OUTPUT_PATH)
