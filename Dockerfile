FROM python:3.11
LABEL maintainer="JanneK"

# Set working directory for our application
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application file (renaming to app.py)
COPY UPBEAT_learning_assistant_GUI.py app.py

# Create necessary directories relative to /app
RUN mkdir -p data learning_plans user_data

# Copy data files from the build context into the corresponding directories under /app
COPY learning_plans/study_plans_data.pickle learning_plans/study_plans_data.pickle
COPY data/curated_additional_materials.txt data/curated_additional_materials.txt
COPY logo.png logo.png

# Create a non-root user and adjust ownership for security:
# - Change owner of /app (which includes our app and data directories)
# - Restrict study_plans_data.pickle so only the owner can read it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    chmod 600 learning_plans/study_plans_data.pickle

# Switch to the non-root user for enhanced security
USER appuser

# Expose the port your app uses
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
