import os
import subprocess
import threading


def install_dependencies():
    """Install required dependencies for the project."""
    print("Installing dependencies...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    print("Dependencies installed successfully.")


def migrate_database():
    """Run Django migrations to set up the database."""
    print("Running database migrations...")
    subprocess.run(["python", "manage.py", "makemigrations"], check=True)
    subprocess.run(["python", "manage.py", "migrate"], check=True)
    print("Database migrations completed.")


def load_sample_data():
    """Load sample data into the database if needed."""
    print("Loading sample data...")
    try:
        subprocess.run(["python", "manage.py", "load_data"], check=True)
        print("Sample data loaded successfully.")
    except Exception as e:
        print(f"Error loading sample data: {e}")


def compile_static_files():
    """Compile static files using Tailwind CSS."""
    print("Compiling Tailwind CSS...")
    try:
        subprocess.run(["python", "manage.py", "tailwind", "start"], check=True)
        print("Tailwind CSS compiled successfully.")
    except Exception as e:
        print(f"Error compiling Tailwind CSS: {e}")


def start_server():
    """Start the Django development server."""
    print("Starting the development server...")
    subprocess.run(["python", "manage.py", "runserver"], check=True)


def main():
    """Run the entire project setup and start the server and Tailwind concurrently."""
    try:
        # Step 1: Install dependencies
        install_dependencies()

        # Step 2: Perform database migrations and load data
        migrate_database()
        load_sample_data()

        # Step 3: Start Tailwind CSS and Django server in concurrent threads
        tailwind_thread = threading.Thread(target=compile_static_files)
        server_thread = threading.Thread(target=start_server)

        # Start both threads
        tailwind_thread.start()
        server_thread.start()

        # Wait for both threads to complete (optional, or join later if needed)
        tailwind_thread.join()
        server_thread.join()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
