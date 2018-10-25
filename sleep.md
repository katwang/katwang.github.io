---
layout: page
title:  Sleep Habits & Mental Health
---


**Dataset**: Sleep Heart Health Study
<br/>
**Language**: Python


Sleeping problems are a prevalent issue that may be tied to cardiovascular and other adverse health problems. This project aims to identify how sleep habits and sleep outcomes may affect one’s mental health. In particular, we are looking at how self-reported quality and length of sleep affect participants’ mental well-being, as captured by the scoring from the RAND 36-Item Short Form Survey (SF-36) questionnaire. The data used for this project comes from the <a href="https://sleepdata.org/datasets/shhs">Sleep Heart Health Study</a> (SHHS), a multi-center cohort study that was formed to study the adverse effects of sleep-disordered breathing. 6,441 participants were enrolled in the study between November 1, 1995 and January 31, 1998 [1,2,3,4]. Regression methods such as linear regression, random forest regression, and XGBoost were used to model individuals' SF-36 Mental Health Index Standardized Score. These methods were also used to determine particular sleep habits that were more influential in determining one's Mental Health Index. Associated behaviors such as drinking and caffeine intake were included in the model to adjust for confounders, but our models picked up on that and the number of naps taken per week as the most influential predictors for mental health. This provides a good starting point for future analyses and tells us that future projects should adjust for these predictors to find other influential sleep habits (if there are any).

My Jupyter Notebook can be found <a href="https://github.com/katwang/Examples/blob/master/shhs_mentalhealth.ipynb">here.

