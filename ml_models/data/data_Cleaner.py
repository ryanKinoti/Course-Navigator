from openpyxl import Workbook
import re
import json
import os


def clean_grade(grade):
    # Remove text in parentheses and clean up the grade
    grade = re.sub(r'\s*\([^)]*\)', '', grade).strip()
    return grade


def parse_exam_results(results):
    # Initialize dictionary to store student processed_data
    students_data = {}
    current_student_id = None

    for line in results:
        # Check for new student record (first record uses '2023 KCSE', others use '.....2023 KCSE')
        if line.startswith('2023 KCSE') or line.startswith('.....2023 KCSE'):
            # Reset for new student processing
            current_student_id = None
            continue

        # Extract student ID and name
        match = re.match(r'(\d{10,}) - (.+)', line)  # Match IDs with at least 10 digits followed by ' - Name'
        if match:
            current_student_id = match.group(1).strip()
            student_name = match.group(2).strip()

            # Initialize student processed_data
            students_data[current_student_id] = {
                'name': student_name,
                'subjects': {},  # Changed to dictionary for easier access
                'mean_grade': None
            }
            continue

        # Extract mean grade
        if line.startswith('Mean Grade:'):
            if current_student_id:
                students_data[current_student_id]['mean_grade'] = clean_grade(line.split(': ')[1].strip())
            continue

        # Extract subject information
        if len(line.split()) >= 4 and line.split()[0].isdigit():
            if current_student_id:
                parts = line.split()

                # Clean up subject code
                subject_code = parts[1]

                # Extract subject name and grade
                subject_line = ' '.join(parts[2:])
                match = re.search(r'(.+?)\s+([A-Z][+-]?)\s*(\(.+\))?$', subject_line)
                if match:
                    subject_name = match.group(1).strip()
                    subject_grade = match.group(2).strip()
                else:
                    subject_name = subject_line  # Fallback if the pattern doesn't match
                    subject_grade = ""

                # Store subjects in a dictionary with subject code as key
                students_data[current_student_id]['subjects'][subject_code] = {
                    'name': subject_name,
                    'grade': subject_grade
                }

    return students_data


def save_results_to_json(students_data, filename='results.json'):
    try:
        with open(filename, 'w') as json_file:
            json.dump(students_data, json_file, indent=4)
        print(f"Results successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


def save_combined_to_excel_file(combined_data, all_subjects, filename='combined_results.xlsx'):
    # Prepare the header
    header = ['Student ID', 'Name'] + all_subjects

    # Initialize the workbook and worksheet
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Combined Exam Results"

    # Write the header row
    worksheet.append(header)

    # Prepare rows for all students
    for student_id, student_data in combined_data.items():
        row = [student_id, student_data['name']]
        for subject_name in all_subjects:
            # Retrieve the grade for each subject, or leave blank if not present
            grade = ""
            for subject in student_data['subjects'].values():
                if subject['name'] == subject_name:
                    grade = subject['grade']
                    break
            row.append(grade)
        worksheet.append(row)

    # Save the workbook to a file
    try:
        workbook.save(filename)
        print(f"Combined XLSX file saved at {filename}")
    except Exception as e:
        print(f"An error occurred while saving the combined XLSX: {e}")


def combine_all_results(input_dir):
    combined_data = {}
    all_subjects = set()
    # print(os.listdir(input_dir))

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):  # Process only .txt files
            input_path = os.path.join(input_dir, filename)
            print(f"Processing file: {input_path}")

            # Parse the file
            with open(input_path, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
                parsed_results = parse_exam_results(lines)

            # Add parsed results to the combined dataset
            for student_id, student_data in parsed_results.items():
                if student_id not in combined_data:
                    combined_data[student_id] = {
                        'name': student_data['name'],
                        'subjects': {},
                        'mean_grade': student_data['mean_grade']
                    }
                combined_data[student_id]['subjects'].update(student_data['subjects'])

            # Update the set of all unique subjects
            for student_data in parsed_results.values():
                all_subjects.update(subject['name'] for subject in student_data['subjects'].values())

    return combined_data, sorted(all_subjects)


def process_and_combine_files(input_dir, output_dir, combined_filename ='original_student_data.xlsx'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Combine all parsed results
    combined_data, all_subjects = combine_all_results(input_dir)

    # Save the combined processed_data to a single Excel file
    combined_filepath = os.path.join(output_dir, combined_filename)
    save_combined_to_excel_file(combined_data, all_subjects, combined_filepath)


# Example usage:
input_directory = 'data_source'
output_directory = 'raw_results'
process_and_combine_files(input_directory, output_directory)
