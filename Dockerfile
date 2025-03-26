FROM python:3.11
LABEL maintainer="JanneK"

# Set working directory for our application
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure gunicorn is installed (if it's not already in requirements.txt)
RUN pip install gunicorn

# Copy the main application file (renaming to app.py)
COPY UPBEAT_learning_assistant_GUI.py app.py

# Create necessary directories relative to /app
RUN mkdir -p data learning_plans user_data

# Copy data files from the build context into corresponding directories under /app
COPY learning_plans/study_plans_data.pickle learning_plans/study_plans_data.pickle
COPY data/curated_additional_materials.txt data/curated_additional_materials.txt
COPY logo.png logo.png

# Create a non-root user and adjust ownership:
# - Change owner of /app (which includes all our files and directories)
# - Set study_plans_data.pickle to be readable by any user (644)
# - Ensure the 'user_data' directory is writable by any user (777) so subfolders can be created
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    chmod 644 learning_plans/study_plans_data.pickle && \
    chmod 777 user_data

# Switch to the non-root user for enhanced security
USER appuser

# Expose the port your app uses
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
