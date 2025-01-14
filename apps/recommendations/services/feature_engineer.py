import pandas as pd


class FeatureEngineer:
    def __init__(self):
        self.subject_mapping = {
            'english': ['english'],
            'mathematics': ['mathematics'],
            'physics': ['physics'],
            'chemistry': ['chemistry'],
            'computers': ['computer studies'],
            'business': ['business studies'],
            'history': ['history and government'],
            'geography': ['geography'],
            'religious_education': ['christian religious education', 'islamic religious education'],
            'language': ['french', 'german', 'kiswahili'],
            'other_technicals': ['home science', 'power mechanics', 'electricity', 'aviation technology', 'metal work',
                                 'wood work', 'agriculture', 'drawing and design', 'art and design', 'music'],
        }

    def feature_vectorization(self, student_grades):
        feature_vector = {}

        for std_subject, subjects in self.subject_mapping.items():
            valid_grades = []
            for sub in subjects:
                if sub in student_grades and pd.notna(student_grades[sub]):
                    valid_grades.append(student_grades[sub])

            feature_vector[std_subject] = max(valid_grades) if valid_grades else 0

        return feature_vector
