---
layout: page
title:  Interactive Map - US Opioid Prescribing Rate
---

**Dataset**: Medicare Part D Prescription Drug Program
<br/>
**Language**: R

An interactive map showing the opioid prescribing rate across the US in 2015.

The map was built using the <code>leaflet</code> package in R, with <a href="https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data">data</a> from the CDC. The data wrangling was performed using <code>sparklyr</code>, an R interface to Apache Spark.


<iframe src="leafletmap" frameborder="0" width="99%" height="510" marginwidth="0" marginheight="0" scrolling="no" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

