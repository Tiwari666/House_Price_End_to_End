# Use official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files from local project to /app in container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port for Flask (default: 5000)
EXPOSE 5000

# Command to run the app
CMD ["python", "FlaskApp/app.py"]


####$####################################STEP1 
## Code to Build the Docker image named loan-predictor-app
# docker build -t loan-predictor-app .


############################################STEP 2
## Run the Docker container:

 # docker run -p 5000:5000 loan-predictor-app 
 ## Note ( Choose both "public" and "private" option)
 ## you will see :* Serving Flask app 'app'
                #:* Debug mode: on

#In Docker, the Flask server runs inside a container, so it won’t print a clickable link. But as long as you see:

# :* Serving Flask app 'app'
# :* Debug mode: on

# That means it's running and accessible.

# Just manually open your browser and go to: http://127.0.0.1:5000/

# Then try entering values and check the output. If you see your form → you're DONE! 

# For entering/checking list , see the setup10_dockerization: or enter them manually.


#######################STEP 3#################

##Log in DOCKER  via terminal: docker login


## : Tag your image :docker tag loan-predictor-app narendratiwari17/loan-predictor-app


## : Push the image: docker push narendratiwari17/loan-predictor-app

####################STEP 4##################################

# Create .dockerignore (to clean up future builds) on the root of the project
## Right-click on the empty space in the Explorer (left sidebar).
## Click “New File”.
## Name it exactly:.dockerignore
## write the required code content for .dockerignore
# Save the file
# Once done, your Docker builds will be faster and cleaner, since it skips unnecessary files.


#######NEXT.................: Cloud deployment######################### 