# Predicting Project Risk
A machine learning model to predict project risk.

The objective of this work is design a [machine learning model](/project-risk-model.ipynb) to predict the probability of a project having issues worth being featured in the project management risk report. The project risk report elaboration requires a significant effort, as the analysts have to peruse many reports and related documents to determine if a project has a high risk. So, by training a discriminative model, we will be able to prioritize the projects that statistically present a high risk profile and reduce the cost of report elaboration. We also want to explore the most significant factors that contribute to project risk, like managers, scope, seasonality, etc.

<img src="/model.png?raw=true" width="700">

## The data

This work uses project data extracted from the Microsoft Project Server database, where the Chamber of Deputies corporate projects are stored. More specifically, we analyse data from  IT projects from March 2015 to August 2016.

The dataset `project-data.csv` used in this model is generated in the notebook `project-risk-features.ipynb`. This dataset has the following attributes:

* `project`: project identifier
* `risk`: label: 1 - high risk, 0 - low risk
* `status`: project status: {'tramitando para contratação', 'em andamento', 'não iniciado', 'sem relatório', 'atrasado', 'dependência externa', 'suspenso', 'em dependência externa', 'cancelado', 'em fase de encerramento', 'atividade', 'sem informação'}
* `compliance`: index for compliance with project management process
* `report_count`: number of reports available for the project
* `has_schedule`: 1 - project has schedule, 0 - otherwise
* `scope`: project scope: {"Corporativo", "Setorial", "Estruturante"}
* `office`: project sponsor office: {'Corporativo', 'CENIN', 'SECOM', 'DILEG', 'DG', 'DIRAD', 'DRH'}
* `month`: month of project report publication
* `year`: year of project report publication
* `day`: day of project report publication
* `risk_previous1`: project appeared in the risk report in the last month
* `risk_previous2`: project appeared in the risk report in the last two months
* `risk_previous3`: project appeared in the risk report in the last three months
* `project_risk_likelihood`: maximum likelihood risk probability estimation (with Laplace smoothing)
* `report_word_count`: number of words in report
* `poa_word_count`: number of words in "points of attention" section
* `estimated_days_finish`: estimated days to finish project
* `manager_risk_likelihood`: maximum likelihood risk probability estimation for managers (with Laplace smoothing)
* `manager_project_count`: number of projects the manager is responsible for in a given month
