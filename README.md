# Course Navigator

Course Navigator is a comprehensive recommendation system designed to help students make informed decisions about their
academic paths. The system leverages high school results to recommend suitable university courses, combining
collaborative and content-based filtering techniques.

## Features

- **User Management**: Authentication system for students and administrators.
- **Course Recommendations**: Suggests university courses based on student grades.
- **Interactive Dashboard**: Allows users to view recommendations, explore courses, and track their preferences.
- **Custom Filters**: Supports advanced filtering options for tailored recommendations.

## Technologies Used

- **Backend**: Django, Python
- **Frontend**: HTML, CSS, Tailwind CSS
- **Machine Learning**: Collaborative and content-based filtering models
- **Database**: SQLite (default, easily configurable for PostgreSQL or MySQL)
- **Others**: Docker, TailwindCSS, Git

## Project Structure

```
Course-Navigator-main/
    .gitignore
    README.md
    manage.py
    requirements.txt
    apps/
        core/
            management/commands/
            migrations/
            templatetags/
        recommendations/
            services/
            migrations/
        users/
            migrations/
    config/
    ml_models/
        data/
        evaluation/
        trained_models/
    templates/
        auth/
        core/
        recommendations/
    theme/
        static/
        static_src/
        templates/
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Node.js (for Tailwind CSS)
- Docker (optional, for containerized deployment)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ryanKinoti/Course-Navigator.git
   cd Course-Navigator-main
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Database Migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

4. **Start the Development Server**
   ```bash
   python manage.py runserver
   ```
   The server will be available at `http://127.0.0.1:8000/`.

5. **Compile Tailwind CSS**
   ```bash
   python manage.py tailwind start
   ```

6. **Load Base Data**
   ```bash
   python manage.py load_data
   ```

## Usage

### Running Locally

1. Start the Django development server.
2. Access the application at `http://127.0.0.1:8000/`.
3. Use the dashboard to explore recommendations and manage profiles.

### Key Endpoints

- **Homepage**: `/`
- **Login**: `/auth/login/`
- **Course Recommendations**: `/recommendations/`

### Automated Setup (Optional)

You can set up the entire project (dependencies installation, database migrations, loading base data, and starting the
server) using the `project_script.py` file:

1. Make sure you have Python 3.8 or higher installed.
2. Navigate to the project directory.
3. Run the following command:

   ```bash
   python project_script.py
   ```

This script will:

1. Install all the required dependencies.
2. Perform database migrations.
3. Load sample data into the database.
4. Start the Django development server and Tailwind CSS compiler.

### Further Notes

- **Concurrent Processes**: Automatically compiles Tailwind CSS and starts the Django server concurrently for a faster
  setup.
- You can either use the automated script (`python project_script.py`) to set up and run the server or execute each step
  manually as previously described in the [Setup and Installation](#setup-and-installation) section. This flexibility
  allows you to customize your workflow based on your needs.

### Troubleshooting

- If `load_data` fails, ensure the data loading script exists in the `management/commands/` directory and is error-free.
- If Tailwind CSS compilation fails, check that Node.js is installed and properly configured.
- For any unexpected errors, consult the console logs and verify installation prerequisites (e.g., Python, Node.js,
  Django, and Tailwind CSS dependencies).

## ML Models

The recommendation system integrates trained models located in `ml_models/trained_models/recommendation_model.pkl`.
These models are responsible for predicting suitable courses based on input data.

### Training and Evaluation

Scripts for training and evaluating the models are in the `ml_models/` directory. To retrain the model:

```bash
  python ml_models/model_trainer.py
```

Evaluation outputs are stored in `ml_models/evaluation/` as images and metrics.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add feature name'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
