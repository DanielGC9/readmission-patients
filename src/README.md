<div id="top"></div> 

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/chiper-inc/ml_amplitudeCalculation">
    <img src="docs/images/chiperLogo.png" alt="Logo" width="80" height="80">
  </a>
<h3 align="center">Customer Churn</h3>

  <p align="center">
    The development of this model was done in search of estimating the probability of Churn of the users and understanding which variables are the ones that affect the making of this decision.
    <br />
    <a href="https://a_url_that_has_documentation"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/chiper-inc/your_repo_name/pulls">Make a Pull Request</a>
    ·
    <a href="https://github.com/chiper-inc/your_repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/chiper-inc/your_repo_name/issues">Request Feature</a>
  </p>

</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#outputs">Outputs</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

This project is carried out from the Global-Data Science cell with the objective of predicting from characteristic, service and geographical variables of each StoreId the probability of Churn, having as definition of Churn those stores that have not placed an order in the last thirty days.
Once the data is loaded from BigQuery, it is preprocessed and then the LightGBM algorithm is implemented to make the prediction, extracting the Churn and retention probabilities in addition to the Shap-Values.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

* [Python](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [Shap](https://shap.readthedocs.io/en/latest/)
* [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

To run this code, it's recommended that you start a new python environment (Using [venv](https://help.dreamhost.com/hc/en-us/articles/115000695551-Installing-and-using-virtualenv-with-Python-3)
or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)) and install the
packages in the requirements.txt file

  ```sh
  pip install -r requirements.txt
  ```

### Environment variables

To run this project, you will need to set environment variables, this can be done using a `.env` file and
a plugin for your IDE, or if you're using conda or venv, you can set the environment variables right into the
terminal.  
These are the environment variables needed to run the code:
```
CREDENTIALS_PATH=path/to/credentials.json 
ENVIRONMENT 
LOGGING_ENV
add the variables you need, remember that must be in upper case and not use special characters
```


The `CREDENTIALS_PATH` variable is the path to the credentials file for the Google Cloud Platform, if you don't have it
ask for it to your leader or the owner of this repo, is important to set this variable as the global path in your
machine. The `LOGGING_ENV` variable is used to determine the type of logging your code is using, if you are running on any
cloud environment use `gcp` otherwise use `develop`.

The environment variables for develop and staging are [here](https://github.com/chiper-inc/deployments-beta-stag/tree/main/cronjobs) 
and the environment variables for production are [here](https://github.com/chiper-inc/deployments-prod/tree/main/cronjobs/data-science), 
search for the repo/cronjob name and look the `variables.properties` file.

### Running the code

Finally, you can run the code of each component independently:

   ```sh
   python src/main/example_component.py
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- OUTPUTS EXAMPLES -->

## Outputs

The outputs of the cron are the following:

The result of this model is a DataFrame containing the following columns:
- storeId: identifier of each user
- Retention: Retention probability
- Churn: Chance of Churn
- Predict: Prediction made by the model
The following columns represent the percentage of weight that each variable has in the model's prediction.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

- **Creator:** Daniel Guzman Cuellar
- **e-mail:** daniel.guzman@chiper.co

Project Link: [https://github.com/chiper-inc/your_repo_name](https://github.com/chiper-inc/your_repo_name)

<!-- Template developed by the ML Team :D-->

<p align="right">(<a href="#top">back to top</a>)</p>
