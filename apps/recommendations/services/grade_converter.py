import numpy as np


class GradeConverter:
    def __init__(self):
        self.grade_weights = {
            'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
            'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0,
            'D-': 0.7, 'E': 0.3, np.nan: 0
        }

    def grade_to_weight(self, grade):
        if isinstance(grade, str):
            return self.grade_weights.get(grade.upper(), 0)
        return 0
