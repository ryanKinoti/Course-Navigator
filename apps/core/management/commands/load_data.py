# apps/core/management/commands/load_data.py
import pandas as pd
from django.core.management.base import BaseCommand
from apps.core.models import Subject, Course, CourseRequirement
from django.conf import settings
import os


class Command(BaseCommand):
    help = 'Load initial course and subject data from Excel file'

    def __init__(self):
        super().__init__()
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
        # Create reverse mapping for easier lookup
        self.reverse_subject_mapping = {}
        for main_subject, sub_subjects in self.subject_mapping.items():
            for sub in sub_subjects:
                self.reverse_subject_mapping[sub] = main_subject

    def handle(self, *args, **kwargs):
        self.stdout.write('Starting data loading process...')

        # Define path to Excel file
        excel_path = os.path.join(settings.COURSE_DATA, 'course_data.xlsx')

        try:
            # Read all sheets
            self.stdout.write('Reading Excel file...')
            all_subjects = pd.read_excel(excel_path, sheet_name='all_subjects')
            all_courses = pd.read_excel(excel_path, sheet_name='all_courses')
            grade_requirements = pd.read_excel(excel_path, sheet_name='grade_requirements')
            subject_importance = pd.read_excel(excel_path, sheet_name='subject_importance')

            # Create subjects from all_subjects sheet
            self.stdout.write('Creating subjects from all_subjects sheet...')
            created_subjects = {}
            for _, row in all_subjects.iterrows():
                subject_name = row['subject_name'].lower()  # Convert to lowercase for consistency
                code = row['subject_code']

                # Find the main subject category for this subject
                main_subject = self.reverse_subject_mapping.get(subject_name)
                if main_subject:
                    subject, created = Subject.objects.get_or_create(
                        code=code,
                        defaults={
                            'name': row['subject_name'],  # Keep original case for display
                            'is_active': True
                        }
                    )
                    # Store both mappings
                    created_subjects[main_subject] = subject  # Store by main category
                    created_subjects[subject_name] = subject  # Store by specific name
                    action = 'Created' if created else 'Already exists'
                    self.stdout.write(f'{action}: Subject {row["subject_name"]} with code {code}')
                else:
                    self.stdout.write(
                        self.style.WARNING(f'Warning: No mapping found for subject {subject_name}')
                    )

            # Create courses from all_courses sheet
            self.stdout.write('\nCreating courses...')
            created_courses = {}
            for _, row in all_courses.iterrows():
                course_code = row['course_code']
                course_name = row['course_full_name']

                # Find corresponding grade requirement
                grade_req = grade_requirements[
                    grade_requirements['course_name'].str.contains(course_code, na=False)
                ]
                mean_grade = grade_req['mean_grade'].iloc[0] if not grade_req.empty else 'C+'

                course, created = Course.objects.get_or_create(
                    code=course_code,
                    defaults={
                        'name': course_name,
                        'description': row['description'],
                        'university': row['university'],
                        'course_url': row['link'],
                        'mean_grade': mean_grade
                    }
                )
                created_courses[course_code] = course
                action = 'Created' if created else 'Already exists'
                self.stdout.write(f'{action}: Course {course_name}')

            # Create course requirements
            self.stdout.write('\nCreating course requirements...')
            for _, row in grade_requirements.iterrows():
                course_code = row['course_name']  # Assuming contains course code
                if course_code not in created_courses:
                    self.stdout.write(
                        self.style.WARNING(f'Warning: Course {course_code} not found in all_courses')
                    )
                    continue

                course = created_courses[course_code]

                # Get subject columns (excluding course_name and mean_grade)
                subject_columns = [col for col in row.index if col not in ['course_name', 'mean_grade']]

                # Get importance weights for this course
                importance_row = subject_importance[
                    subject_importance['course_name'].str.contains(course_code, na=False)
                ]

                for subject_col in subject_columns:
                    if pd.isna(row[subject_col]):
                        continue

                    subject_key = subject_col.lower()
                    if subject_key not in created_subjects:
                        self.stdout.write(
                            self.style.WARNING(f'Warning: Subject {subject_key} not found for {course_code}')
                        )
                        continue

                    subject = created_subjects[subject_key]

                    # Get importance weight
                    importance_weight = (
                        importance_row[subject_col].iloc[0]
                        if not importance_row.empty and subject_col in importance_row.columns
                        else 1.0
                    )

                    # Create requirement
                    requirement, req_created = CourseRequirement.objects.get_or_create(
                        course=course,
                        subject=subject,
                        defaults={
                            'minimum_grade': row[subject_col],
                            'importance_weight': importance_weight
                        }
                    )
                    if req_created:
                        self.stdout.write(
                            f'Created requirement: {subject.name} (grade: {row[subject_col]}, '
                            f'weight: {importance_weight}) for {course_code}'
                        )

            self.stdout.write(self.style.SUCCESS('Successfully loaded all data from Excel file'))

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error loading data: {str(e)}')
            )
