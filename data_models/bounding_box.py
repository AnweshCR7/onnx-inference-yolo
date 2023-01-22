from typing import Union, List, Dict


class BoundingBox:
    """Class to store License plate character"""

    def __init__(self, position: list) -> None:
        assert len(position) == 4, "position can only contain exactly 4 values"
        self.x1, self.y1, self.x2, self.y2 = position

    def get_center_coordinates(self) -> dict:
        return {'x': (self.x1 + self.x2) / 2, 'y': (self.y1 + self.y2) / 2}

    def get_coordinates(self, astype: str = "list") -> Union[List, Dict[str, int]]:
        if astype == "list":
            return [self.x1, self.y1, self.x2, self.y2]
        else:
            return dict(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)
