import os
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
from .grade_converter import GradeConverter
from .feature_engineer import FeatureEngineer
from ...core.models import Subject


class CourseRecommender:
    def __init__(self):
        self.model_components = self._load_model()
        self.grade_converter = GradeConverter()
        self.feature_engineer = FeatureEngineer()
        self.course_data = self._load_course_data()
        self.subject_importance = self._load_subject_importance()

    def _load_model(self):
        model_path = os.path.join(settings.TRAINED_MODELS_DIR, 'recommendation_model.pkl')
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def _load_course_data(self):
        course_data_path = os.path.join(settings.COURSE_DATA, 'course_data.xlsx')
        return pd.read_excel(course_data_path, sheet_name='grade_requirements')

    def _load_subject_importance(self):
        course_data_path = os.path.join(settings.COURSE_DATA, 'course_data.xlsx')
        return pd.read_excel(course_data_path, sheet_name='subject_importance')

    def _cosine_similarity_calc(self, student_vector, course_row, subject_value):
        """Copied from your model_trainer's cosine_similarity_calc method"""
        course_name = course_row['course_name']
        required_subjects = []
        student_grades = []
        required_grades = []
        weights = []

        for subject in course_row.index:
            if subject not in ['course_name', 'mean_grade'] and pd.notna(course_row[subject]):
                required_subjects.append(subject)

                importance_row = subject_value[subject_value['course_name'] == course_name]
                if not importance_row.empty:
                    specific_weight = importance_row[subject].iloc[0]
                    weight = 0.0 if pd.isna(specific_weight) else specific_weight
                else:
                    weight = 0.0

                student_grade = student_vector.get(subject, 0.0)
                required_grade = self.grade_converter.grade_to_weight(course_row[subject])

                student_grades.append(student_grade if pd.notna(student_grade) else 0.0)
                required_grades.append(required_grade)
                weights.append(weight)

        student_grades = np.array(student_grades)
        required_grades = np.array(required_grades)
        weights = np.array(weights)

        weighted_student = student_grades * weights
        weighted_course = required_grades * weights

        similarity = np.dot(weighted_student, weighted_course) / (
                np.linalg.norm(weighted_student) * np.linalg.norm(weighted_course)
        ) if np.linalg.norm(weighted_student) * np.linalg.norm(weighted_course) != 0 else 0

        return similarity

    def calculate_similarity(self, student_vector, student_mean):
        """Calculate cosine similarity scores for each course"""
        similarities = {}

        for _, course_row in self.course_data.iterrows():
            required_mean = self.grade_converter.grade_to_weight(course_row['mean_grade'])

            if student_mean >= required_mean:
                similarity = self._cosine_similarity_calc(
                    student_vector,
                    course_row,
                    self.subject_importance
                )
                similarities[course_row['course_name']] = similarity
            else:
                similarities[course_row['course_name']] = 0.0

        return similarities

    def predict_courses(self, student_grades):
        """
        Predict courses based on student grades, returning both similarity scores
        and model confidence predictions separately.

        Args:
            student_grades (dict): Dictionary of subject-grade pairs

        Returns:
            tuple: (similarities_list, confidence_list, recommendations) where each is a
                  list of dictionaries containing course predictions and scores
        """
        try:
            # Convert grades to weights
            weighted_grades = {
                subject.lower(): self.grade_converter.grade_to_weight(grade)
                for subject, grade in student_grades.items()
            }

            # Create feature vector
            subject_vector = {subject: 0.0 for subject in self.feature_engineer.subject_mapping.keys()}

            # Update with actual grades
            for subject, grade in weighted_grades.items():
                for std_subject, variants in self.feature_engineer.subject_mapping.items():
                    if any(variant in subject.lower() for variant in variants):
                        subject_vector[std_subject] = grade
                        break

            # Calculate student's mean grade
            valid_grades = [g for g in weighted_grades.values() if pd.notna(g) and g > 0]
            student_mean = np.mean(valid_grades) if valid_grades else 0.0

            # Calculate similarities
            similarities = {}
            for _, course_row in self.course_data.iterrows():
                required_mean = self.grade_converter.grade_to_weight(course_row['mean_grade'])

                if student_mean >= required_mean:
                    similarity = self._cosine_similarity_calc(
                        subject_vector,
                        course_row,
                        self.subject_importance
                    )
                else:
                    similarity = 0.0

                similarities[course_row['course_name']] = similarity

            # Create similarity feature array for model input
            similarity_features = []
            for course in self.model_components['course_labels']:
                similarity_features.append(similarities.get(course, 0.0))

            # Scale features and get model predictions
            X = np.array([similarity_features])
            X_scaled = self.model_components['scaler'].transform(X)
            probabilities = self.model_components['model'].predict_proba(X_scaled)[0]

            # Combine all results
            recommendations = []
            for idx, course in enumerate(self.model_components['course_labels']):
                similarity = similarities.get(course, 0.0)
                confidence = float(probabilities[idx])

                recommendations.append({
                    'course': course,
                    'similarity_score': similarity,
                    'confidence_score': confidence,
                    'combined_score': (similarity + confidence) / 2
                })

            # Sort by combined score
            recommendations.sort(key=lambda x: x['combined_score'], reverse=True)

            return recommendations

        except Exception as e:
            print(f"Error in predict_courses: {str(e)}")
            raise
