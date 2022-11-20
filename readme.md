---
title: "README"
output: html_document
---

# Outlier Detection Project README

This is the README file to help with the understanding and running of the code behind 'NEEDLE IN A HAYSTACK: Using deep learning to detect outliers in regulatory insurance data' by Andrew Calver. This is the final project for a Data Science MSc at Birkbeck College.

# Main pipelines

## Helper functions

2 scripts contain the necessary custom functions to run the pipelines: `dataProcessing.py` (dP) and `MLfunctions.py` (ML). dP contains various data manipulation functions like data loading and formatting. ML contains the machine learning functions 

## Pipelines

The 2 pipelines are contained in 2 scripts: `_AllFirmPipeline.py` and `_ClusterPipeline.py`. Intuitively, `_AllFirmPipeline.py` is the pipeline using an autoencoder on all firms, and `_ClusterPipeline.py` is the pipeline using seperate autoencoders on the 2 clusters

## Project justifications
In the project, justifications are given for choosing various configurations of machine learning algorithms and hyperparameters. 3 scripts contain the code behind these justifications: `AE hyperparameter tune.py`, `AE scaling comparions.py` and `lstm grid search.py`. These are not required for the running of the main pipelines, but included to demonstarte the steps taken to justify decisions outlined in the report.

# Saved files
As the report discusses, there were software issues with getting the pipelines to run from start to finish without crashing due to the limited system capability. In order to produce results, the pipelines were run using a subset of firms, the results saved and the pipelines rerun using the second subset of firms. The `lstm saved files.py` script was used for the saving and loading of pipeline results. This script is a consequence of software limitations and on a system with the sufficient requirements for this code base, this script would be redundant.

# Confidentiality
Due to the confidential nature of the data, none of the input files were available for upload to GitHub, raw data or saved files