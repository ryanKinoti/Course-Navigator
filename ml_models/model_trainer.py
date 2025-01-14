import os
import pickle
import warnings
import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, precision_score,
    recall_score, f1_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CourseModelTrainer:
    def __init__(self):
        self.grade_weights = {
            'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
            'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0,
            'D-': 0.7, 'E': 0.3, np.nan: 0
        }

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
            'other_technicals': ['home science', 'power mechanics', 'electricity', 'aviation technology',
                                 'metal work', 'wood work', 'agriculture', 'drawing and design',
                                 'art and design', 'music'],
        }

        warnings.filterwarnings('ignore')

    def load_data(self):
        """Load and prepare initial data"""
        logger.info("Loading course and student data...")

        try:
            course_data = 'data/data_results/course_data.xlsx'
            student_data = 'data/data_results/student_data.xlsx'

            all_courses = pd.read_excel(course_data, sheet_name='grade_requirements')
            subject_value = pd.read_excel(course_data, sheet_name='subject_importance')
            weight_classes = pd.read_excel(course_data, sheet_name='weight_importance')
            all_students = pd.read_excel(student_data)

            subject_value = subject_value.fillna(0)
            weight_classes = weight_classes.set_index('classification')['weight'].to_dict()

            logger.info("Data loading completed successfully")
            return all_courses, subject_value, weight_classes, all_students

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def grade_to_weight(self, grade):
        """Convert grade to numerical weight"""
        if pd.isna(grade):
            return np.nan
        return self.grade_weights.get(grade, 0)

    def student_grade_conversion(self, student_row):
        """Convert student grades to weights"""
        converted_data = {'student_id': student_row['student_id']}

        for col in student_row.index:
            if col != 'student_id' and pd.notna(student_row[col]):
                converted_data[col.lower()] = self.grade_to_weight(student_row[col])

        return converted_data

    def feature_vectorization(self, student_row):
        """Create feature vector from student data"""
        feature_vector = {}
        try:
            for std_subject, student_subjects in self.subject_mapping.items():
                valid_grades = []
                for sub in student_subjects:
                    if sub in student_row and pd.notna(student_row[sub]):
                        valid_grades.append(student_row[sub])

                feature_vector[std_subject] = max(valid_grades) if valid_grades else 0

            return feature_vector

        except Exception as e:
            logger.error(f"Error in feature vectorization: {str(e)}")
            raise

    def cosine_similarity_calc(self, student_vector, course_row, subject_value, weight_classes):
        """Calculate cosine similarity between student and course vectors"""
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
                required_grade = self.grade_to_weight(course_row[subject])

                student_grades.append(student_grade if pd.notna(student_grade) else 0.0)
                required_grades.append(required_grade)
                weights.append(weight)

        student_grades = np.array(student_grades)
        required_grades = np.array(required_grades)
        weights = np.array(weights)

        weighted_student = student_grades * weights
        weighted_course = required_grades * weights

        similarity = cosine_similarity(
            weighted_student.reshape(1, -1),
            weighted_course.reshape(1, -1)
        )[0][0]

        return similarity

    def generate_similarity_data(self):
        """Generate similarity data for all students"""
        logger.info("Starting similarity calculations...")

        course_data, subject_importance, classification_weights, student_data = self.load_data()
        output_data = {}
        total_students = len(student_data)

        for idx, student_row in student_data.iterrows():
            if idx % 100 == 0:  # Progress update every 100 students
                logger.info(f"Processing student {idx + 1}/{total_students}")

            student_id = student_row['student_id']
            converted_grades = self.student_grade_conversion(student_row)
            student_vector = self.feature_vectorization(converted_grades)

            valid_grades = [g for g in student_vector.values() if pd.notna(g) and g > 0]
            student_mean = np.mean(valid_grades) if valid_grades else 0.0

            student_recommendations = {'student_id': student_id}

            for _, course_row in course_data.iterrows():
                required_mean = self.grade_to_weight(course_row['mean_grade'])
                if student_mean >= required_mean:
                    similarity = self.cosine_similarity_calc(
                        student_vector, course_row, subject_importance, classification_weights
                    )
                    student_recommendations[course_row['course_name']] = similarity
                else:
                    student_recommendations[course_row['course_name']] = 0.0

            output_data[student_id] = student_recommendations

        output_df = pd.DataFrame(output_data).T.reset_index(drop=True)

        # Save similarity data
        output_path = 'data/data_results/recommendation_data.xlsx'
        output_df.to_excel(output_path, index=False)
        logger.info(f"Similarity calculations completed and saved to {output_path}")

        if idx == total_students - 1:
            logger.info(f"Completed processing all {total_students} students!")

        return output_df

    def prepare_model_data(self, similarity_data):
        """Prepare data for model training"""
        logger.info("Preparing data for model training...")

        def softmax(x):
            x = np.array(x, dtype=np.float64)
            exp_x = np.exp(x - np.max(x))
            return exp_x / exp_x.sum()

        similarity_cols = similarity_data.columns[1:]
        # probabilities = similarity_data[similarity_cols].apply(softmax, axis=1)
        probabilities = pd.DataFrame(
            [softmax(row) for row in similarity_data[similarity_cols].values],
            columns=similarity_cols
        )

        N = 6  # Number of top recommendations
        top_recommendations = probabilities.apply(
            lambda x: x.nlargest(N).index.tolist(), axis=1
        )

        X = similarity_data[similarity_cols].values
        all_courses = similarity_cols.tolist()

        y = np.zeros((len(similarity_data), len(all_courses)))
        for i, recommendations in enumerate(top_recommendations):
            for course in recommendations:
                y[i, all_courses.index(course)] = 1

        return X, y, all_courses

    def train_and_evaluate_model(self, X, y, course_labels):
        """Train and evaluate the model"""
        logger.info("Starting model training and evaluation...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and train model
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],        # Reduced max_depth
            'min_samples_split': [5, 10, 15],  # Increased min_samples_split
            'min_samples_leaf': [2, 4, 6],     # Increased min_samples_leaf
            'max_features': ['sqrt', 'log2'],   # Changed feature selection
            'class_weight': ['balanced'],
            'max_samples': [0.7, 0.8, 0.9]     # Added bootstrap sample size
        }
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        base_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            max_features='sqrt',
            class_weight='balanced'
        )
        logger.info("Performing Grid Search for Hyperparameter Tuning...")
        grid_search.fit(X_train_scaled, y_train)

        best_rf_model = grid_search.best_estimator_
        logger.info(f"Best Parameters from Grid Search: {grid_search.best_params_}")

        model = OneVsRestClassifier(best_rf_model)
        logger.info("Training the final OneVsRestClassifier model...")
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        logger.info("Evaluating model performance...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)

        # Print metrics
        logger.info("Model Performance Metrics:")
        logger.info("-" * 50)
        logger.info(f"Overall Accuracy: {accuracy_score(y_test, y_pred)}")
        logger.info("Detailed Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=course_labels))

        # Create evaluation plots
        self._create_evaluation_plots(y_test, y_pred, y_pred_proba, course_labels)

        return model, scaler

    def _create_evaluation_plots(self, y_test, y_pred, y_pred_proba, course_labels):
        """Create and save evaluation plots"""
        logger.info("Generating evaluation plots...")

        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))

        # Confusion Matrix
        plt.subplot(2, 2, 1)
        cm_avg = np.zeros((2, 2))
        for i in range(len(course_labels)):
            cm = confusion_matrix(y_test[:, i], y_pred[:, i])
            cm_avg += cm
        cm_avg = cm_avg / len(course_labels)

        sns.heatmap(cm_avg, annot=True, fmt='.2f',
                    xticklabels=['Not Recommended', 'Recommended'],
                    yticklabels=['Not Recommended', 'Recommended'])
        plt.title('Average Confusion Matrix Across All Courses')

        # ROC Curve
        plt.subplot(2, 2, 2)
        self._plot_roc_curve(y_test, y_pred_proba, course_labels)

        # Precision-Recall Curve
        plt.subplot(2, 2, 3)
        self._plot_pr_curve(y_test, y_pred_proba, course_labels)

        # Performance Metrics
        plt.subplot(2, 2, 4)
        self._plot_performance_metrics(y_test, y_pred, course_labels)

        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f'evaluation/model_evaluation_{timestamp}.png'
        os.makedirs('evaluation', exist_ok=True)
        plt.savefig(plot_path)
        logger.info(f"Evaluation plots saved to {plot_path}")
        plt.close()

    def _plot_roc_curve(self, y_test, y_pred_proba, course_labels):
        """Plot ROC curve"""
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)

        for i in range(len(course_labels)):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr /= len(course_labels)

        plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {auc(mean_fpr, mean_tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Average ROC Curve Across All Courses')
        plt.legend()

    def _plot_pr_curve(self, y_test, y_pred_proba, course_labels):
        """Plot Precision-Recall curve"""
        mean_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(mean_recall)

        for i in range(len(course_labels)):
            precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
            mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])
        mean_precision /= len(course_labels)

        plt.plot(mean_recall, mean_precision, label='Mean PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average Precision-Recall Curve Across All Courses')
        plt.legend()

    def _plot_performance_metrics(self, y_test, y_pred, course_labels):
        """Plot per-course performance metrics"""
        metrics = []
        for i, course in enumerate(course_labels):
            precision = precision_score(y_test[:, i], y_pred[:, i], average='binary')
            recall = recall_score(y_test[:, i], y_pred[:, i], average='binary')
            f1 = f1_score(y_test[:, i], y_pred[:, i], average='binary')
            metrics.append([precision, recall, f1])

        metrics_df = pd.DataFrame(metrics,
                                  columns=['Precision', 'Recall', 'F1-Score'],
                                  index=course_labels)
        metrics_df.plot(kind='bar', ax=plt.gca())
        plt.title('Per-Course Performance Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        return metrics_df

    def save_model(self, model, scaler, course_labels):
        """Save the trained model and components"""
        logger.info("Saving model and components...")

        model_components = {
            'model': model,
            'scaler': scaler,
            'course_labels': course_labels,
        }

        # Save in ml_models directory
        os.makedirs('trained_models', exist_ok=True)
        model_path = 'trained_models/recommendation_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_components, f)
        logger.info(f"Model saved to {model_path}")

    def train(self):
        """Main training pipeline"""
        try:
            logger.info("Starting model training pipeline...")

            # Generate similarity data
            similarity_data = self.generate_similarity_data()

            # Prepare data for modeling
            X, y, course_labels = self.prepare_model_data(similarity_data)

            # Train and evaluate model
            model, scaler = self.train_and_evaluate_model(X, y, course_labels)

            # Save model
            self.save_model(model, scaler, course_labels)

            logger.info("Model training pipeline completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            return False

    def test_student_recommendation(self, student_grades):
        """
        Test the model with a single student's grades and get both similarity scores
        and model confidence predictions separately.

        Args:
            student_grades (dict): Dictionary of subject-grade pairs

        Returns:
            tuple: (similarity_scores, model_predictions) where each is a sorted list
                  of (course, score) tuples
        """
        logger.info("Testing model with provided student grades...")

        try:
            # Load the trained model and components
            model_path = 'trained_models/recommendation_model.pkl'
            with open(model_path, 'rb') as f:
                model_components = pickle.load(f)

            model = model_components['model']
            scaler = model_components['scaler']
            course_labels = model_components['course_labels']

            # Convert student grades to feature vector
            student_row = {'student_id': 'test_student'}
            for subject, grade in student_grades.items():
                student_row[subject.lower()] = grade

            converted_grades = self.student_grade_conversion(pd.Series(student_row))
            student_vector = self.feature_vectorization(converted_grades)

            # Calculate mean grade
            valid_grades = [g for g in student_vector.values() if pd.notna(g) and g > 0]
            student_mean = np.mean(valid_grades) if valid_grades else 0.0
            logger.info(f"\nStudent Mean Grade: {student_mean:.2f}")

            # Load course data for similarity calculation
            course_data, subject_importance, classification_weights, _ = self.load_data()

            # Calculate similarities
            similarities = []
            for _, course_row in course_data.iterrows():
                required_mean = self.grade_to_weight(course_row['mean_grade'])
                if student_mean >= required_mean:
                    similarity = self.cosine_similarity_calc(
                        student_vector, course_row, subject_importance, classification_weights
                    )
                    similarities.append((course_row['course_name'], similarity))
                else:
                    similarities.append((course_row['course_name'], 0.0))

            # Sort similarities by score
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Prepare input for model prediction
            similarity_dict = dict(similarities)
            X = np.array([[similarity_dict[course] for course in course_labels]])
            X_scaled = scaler.transform(X)

            # Get model predictions
            probabilities = model.predict_proba(X_scaled)

            # Create sorted list of model predictions
            model_predictions = []
            for i, course in enumerate(course_labels):
                confidence = probabilities[0][i]
                model_predictions.append((course, confidence))

            # Sort predictions by confidence
            model_predictions.sort(key=lambda x: x[1], reverse=True)

            # Print Similarity Scores
            logger.info("\nTop Courses by Similarity Score:")
            logger.info("-" * 50)
            for course, similarity in similarities[:6]:
                logger.info(f"{course}: {similarity:.4f}")

            # Print Model Confidence Predictions
            logger.info("\nTop Courses by Model Confidence:")
            logger.info("-" * 50)
            for course, confidence in model_predictions[:6]:
                logger.info(f"{course}: {confidence:.4f}")

            return similarities, model_predictions

        except Exception as e:
            logger.error(f"Error in testing model: {str(e)}")
            return None


def main():
    """Main execution function"""
    trainer = CourseModelTrainer()
    success = trainer.train()

    if success:
        logger.info("Model training completed successfully!")
        # Test student grades
        test_grades = {
            'English': 'D+',
            'Mathematics': 'C',
            'Geography': 'B+',
            'History and Government': 'B+',
            'Kiswahili': 'B',
            'Business Studies': 'B',
            'Christian Religious Education': 'B+',
        }

        logger.info("\nTesting model with sample student grades...")
        similarities, model_predictions = trainer.test_student_recommendation(test_grades)

        if similarities and model_predictions:
            logger.info("\nTest completed successfully!")
        else:
            logger.error("Test failed!")
    else:
        logger.error("Model training failed!")


if __name__ == "__main__":
    main()
