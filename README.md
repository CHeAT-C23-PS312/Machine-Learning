# Machine Learning
Machine Learning Service for CHeAT-C23-PS312

## Contributor
Contributor to this repostory:

- [Firly Nuraulia Rahmah](https://github.com/firlynuraulia)
- [Tedja Diah Rani Octavia](https://github.com/ranioc)
- [Amalia Hasanah](https://github.com/amaliahsn14)

## Tech Stacks

This project was built on top of:

- [FastAPI](https://fastapi.tiangolo.com/)
- [Google App Engine](https://cloud.google.com/appengine)
- [Cloud Storage](https://console.cloud.google.com/storage)


## YAML Configuration 

Google Cloud Platform allows App Engine to perform deployment based on a configuration defined in a yaml file. You have followed the configuration

   ```
   runtime: python39
   instance_class: F2
   entrypoint: uvicorn main:app --host=0.0.0.0 --port=8080

   runtime_config:
   python_version: 3.9
   ```

### Building a New Project

Ensure to create a new project in GCP. For the new project being created, give Project name, Project ID, Billing account and click Create.

### Clone CHeAT Repository from Github

Enter the command on the cloud shell.

```bash
$ git clone -b https://github.com/CHeAT-C23-PS312/Machine-Learning.git
```

### Create and Activate Virtual Environment

```bash
$ virtualenv env
```

```bash
$ source env/bin/activate
```

### Install Requirements for FastAPI

```bash
$ pip install -r requirements.txt
```

## Create APP Engine on GCP

on the cloud shell, type the following command.

```bash
$ gcloud app create
```

## Deploying Model ML on App Engine

Run the following command in Google Cloud Shell to deploy Machine Learning Model to Google App Engine.

```bash
$ gcloud app deploy
```

To proceed with deployment, simply press Y and hit enter when prompted. Typically, your app will be deployed on a URL that follows this format: your-project-id.appspot.com.
