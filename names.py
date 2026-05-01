from dataclasses import dataclass, field

@dataclass
class Face:
    name: str = "-"
    id: int = 0
    bbox: list[float] = field(default_factory=list)
    face_center: tuple[int, int] = (0, 0)
    gesture_score: float = 0.0

    def is_new(self) -> bool:
        return self.name == "-"


@dataclass
class KnownFaces:
    faces: list[Face] = field(default_factory=list)

    def add(self, face: Face) -> None:
        self.faces.append(face)

    def get_by_id(self, face_id: int) -> Face | None:
        return next((f for f in self.faces if f.id == face_id), None)

    def get_by_name(self, name: str) -> Face | None:
        return next((f for f in self.faces if f.name == name), None)