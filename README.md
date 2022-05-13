## Overview of what each file does

`app.py` where the dash app lives <br>
`requirements.txt` python modules that will be installed onto the Heroku Dyno process (your linux webserver) at build <br>
`runtime.txt` simply tells Heroku (the Gunicorn HTTP server) which Python version to use <br>
`Procfile` tells Heroku what type of process is going to run (Gunicorn web process) and the Python app entrypoint (app.py) <br>
`/assets` this directory is ONLY to serve the favicon icon. It cannot be used to serve any other static files <br>
`functions.py` this file stores the helper functions for the application, including all of the code implementing the +4 spanner<br>
`data.csv` stores the runtime data so that we do not have to rerun the algorithm on graphs of different sizes<br>
`styles.json` creates css styling for the application
`.gitignore` file telling git which files and directories to ignore when pushing to the remote repositories <br>

## Instructions

## 1. Clone this repo to your local machine and install modules

First clone the repository into your local machine using `git clone`

Next Install modules from the requirements.txt using the commands <br>

`pip3 install -r requirements.txt` <br> or <br>
`conda install --file requirements.txt`
<br><br>

## 2. Run the app locally

Run the app from your IDE direct, or like a boss from the terminal: `python3 app.py`

If it runs, it should be visible on a browser via `http://0.0.0.0:8050`
<br><br>
